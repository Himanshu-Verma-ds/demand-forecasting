#!/usr/bin/env bash
# Full pipeline: dataset -> tuning -> training -> comparison -> promotion.
#
# Every stage logs to logs/<step>.log and to MLflow. Tuning creates one nested MLflow run per
# Optuna trial, so every hyperparameter set ever evaluated is recorded with its metrics.
#
# Usage:
#   ./scripts/run_all.sh                          # config defaults
#   METRIC=mape ./scripts/run_all.sh              # rank and tune on validation MAPE
#   CONFIG=configs/config.yaml ./scripts/run_all.sh
set -euo pipefail

CONFIG="${CONFIG:-configs/config.yaml}"
METRIC="${METRIC:-}"          # empty = use config training.selection_metric
SKIP_DARTS="${SKIP_DARTS:-0}" # set to 1 to skip the slow Darts models
PYTHON="${PYTHON:-python}"    # override if the venv is not activated, e.g. PYTHON=./venv/bin/python

export PYTHONPATH="${PYTHONPATH:-}:src"

# Artifact directory comes from the config so alternate configs stay self-contained.
SAVE_DIR="$("$PYTHON" -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['training']['save_dir'])" "$CONFIG")"

metric_args=()
if [ -n "$METRIC" ]; then
  metric_args=(--metric "$METRIC")
fi

step() { echo; echo "=================== $* ==================="; }

step "1/5  Build the point-in-time feature table"
"$PYTHON" -m demand_forecasting.prepare_dataset --config "$CONFIG"

step "2/6  Bayesian tuning - tree models (every trial logged to MLflow)"
for model in lightgbm xgboost catboost; do
  echo "--- tuning $model ---"
  "$PYTHON" -m demand_forecasting.tune_bayesian --config "$CONFIG" --model "$model" "${metric_args[@]}"
done

step "3/6  Bayesian tuning - sequence models (searches lookback / input chunk length)"
dl_models=(lstm transformer)
if [ "$SKIP_DARTS" != "1" ]; then
  dl_models+=(tide tsmixer)
fi

for model in "${dl_models[@]}"; do
  echo "--- tuning $model ---"
  "$PYTHON" -m demand_forecasting.tune_dl --config "$CONFIG" --model "$model" "${metric_args[@]}"
done

step "4/6  Train and evaluate every model on its tuned hyperparameters"
for model in lightgbm xgboost catboost; do
  echo "--- training $model ---"
  "$PYTHON" -m demand_forecasting.train_ml --config "$CONFIG" --model "$model" \
    --params-json "${SAVE_DIR}/tuning/${model}_best_params.json"
done

for model in lstm transformer; do
  echo "--- training $model ---"
  "$PYTHON" -m demand_forecasting.train_dl --config "$CONFIG" --model "$model" \
    --params-json "${SAVE_DIR}/tuning/${model}_best_params.json"
done

if [ "$SKIP_DARTS" != "1" ]; then
  for model in tide tsmixer; do
    echo "--- training $model ---"
    "$PYTHON" -m demand_forecasting.train_darts --config "$CONFIG" --model "$model" \
      --params-json "${SAVE_DIR}/tuning/${model}_best_params.json"
  done
fi

step "5/6  Rank every model, and report the best per family with its test inference"
"$PYTHON" -m demand_forecasting.evaluate_compare --config "$CONFIG" "${metric_args[@]}"
"$PYTHON" -m demand_forecasting.report_best_models --config "$CONFIG" "${metric_args[@]}"

step "6/6  Promote the validation champion"
"$PYTHON" - "$CONFIG" "${METRIC}" <<'EOF'
"""Read the ranked comparison and promote the top model with the right family/paths."""
import subprocess
import sys

import pandas as pd
import yaml

config_path, metric = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
metric = metric or cfg["training"].get("selection_metric", "wape")

comparison = pd.read_csv("reports/model_comparison.csv")
champion = comparison.sort_values(f"val_{metric}").iloc[0]["model"]

FAMILY = {
    "lightgbm": ("ml", "model.joblib"), "xgboost": ("ml", "model.joblib"), "catboost": ("ml", "model.joblib"),
    "lstm": ("dl", "model.pt"), "transformer": ("dl", "model.pt"),
    "tide": ("darts", "model.pt"), "tsmixer": ("darts", "model.pt"),
}

family, artifact_name = FAMILY[champion]
save_dir = cfg["training"]["save_dir"]

cmd = [
    sys.executable, "-m", "demand_forecasting.select_model",
    "--config", config_path,
    "--metric", metric,
    "--name", champion,
    "--family", family,
    "--model-type", champion,
    "--artifact", f"{save_dir}/{champion}/{artifact_name}",
    "--metrics", f"{save_dir}/{champion}/metrics.json",
]

if family == "darts":
    cmd += ["--metadata-path", f"{save_dir}/{champion}/metadata.joblib"]

print(f"Promoting champion by val_{metric}: {champion}")
subprocess.run(cmd, check=True)
EOF

step "Done"
echo "Comparison : reports/model_comparison.csv"
echo "Best/family: reports/best_models.csv (+ per-series and per-horizon breakdowns)"
echo "Registry   : configs/model_registry.yaml"
echo "Logs       : logs/*.log"
echo "MLflow     : mlflow ui --backend-store-uri sqlite:///mlflow.db"
