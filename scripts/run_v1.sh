#!/usr/bin/env bash
# v1 production run on the full dataset, tracked in DagsHub MLflow.
#
# Trial counts are per family because the cost profile differs by an order of magnitude:
# a tree trial is ~1.5 min (dominated by the 14 causal feature rebuilds inside
# recursive_forecast), while a sequence-model trial trains a network.
#
#   PYTHON=./venv/bin/python bash scripts/run_v1.sh
set -uo pipefail

CONFIG="${CONFIG:-configs/config.yaml}"
PYTHON="${PYTHON:-python}"
TREE_TRIALS="${TREE_TRIALS:-12}"
TORCH_TRIALS="${TORCH_TRIALS:-4}"
DARTS_TRIALS="${DARTS_TRIALS:-3}"
FORCE="${FORCE:-0}"   # 1 = redo stages whose outputs already exist

export PYTHONPATH="${PYTHONPATH:-}:src"

SAVE_DIR="$("$PYTHON" -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['training']['save_dir'])" "$CONFIG")"
STATUS="reports/v1_run_status.txt"
mkdir -p reports

: > "$STATUS"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Never let one model failure abort the whole run - record it and continue.
run() {
  local name="$1"; shift
  local start=$SECONDS
  echo "[$(stamp)] START $name" | tee -a "$STATUS"

  if "$@"; then
    echo "[$(stamp)] PASS  $name ($((SECONDS - start))s)" | tee -a "$STATUS"
  else
    echo "[$(stamp)] FAIL  $name ($((SECONDS - start))s)" | tee -a "$STATUS"
  fi
}

# Skip a stage whose output already exists, so an interrupted run resumes cheaply.
# Optuna studies resume from their own sqlite storage, so tuning is always re-entered.
skip_if() {
  local target="$1"
  [ "$FORCE" = "0" ] && [ -e "$target" ]
}

echo "=== v1 run: config=$CONFIG save_dir=$SAVE_DIR trees=$TREE_TRIALS ===" | tee -a "$STATUS"

if skip_if "data/processed/features.parquet"; then
  echo "[$(stamp)] SKIP  prepare_dataset (features.parquet exists)" | tee -a "$STATUS"
else
  run prepare_dataset "$PYTHON" -m demand_forecasting.prepare_dataset --config "$CONFIG"
fi

# ---- tuning -----------------------------------------------------------------
for m in lightgbm xgboost catboost; do
  run "tune_$m" "$PYTHON" -m demand_forecasting.tune_bayesian --config "$CONFIG" --model "$m" --n-trials "$TREE_TRIALS"
done

for m in lstm transformer; do
  run "tune_$m" "$PYTHON" -m demand_forecasting.tune_dl --config "$CONFIG" --model "$m" --n-trials "$TORCH_TRIALS"
done

for m in tide tsmixer; do
  run "tune_$m" "$PYTHON" -m demand_forecasting.tune_dl --config "$CONFIG" --model "$m" --n-trials "$DARTS_TRIALS"
done

# ---- final training ---------------------------------------------------------
# Fall back to defaults when a tuning stage failed and left no params file.
params_arg() {
  local f="${SAVE_DIR}/tuning/$1_best_params.json"
  [ -f "$f" ] && echo "--params-json $f" || echo ""
}

train_stage() {
  local m="$1" module="$2"
  if skip_if "${SAVE_DIR}/${m}/metrics.json"; then
    echo "[$(stamp)] SKIP  train_$m (metrics.json exists)" | tee -a "$STATUS"
  else
    run "train_$m" "$PYTHON" -m "demand_forecasting.$module" --config "$CONFIG" --model "$m" $(params_arg "$m")
  fi
}

for m in lightgbm xgboost catboost; do train_stage "$m" train_ml; done
for m in lstm transformer;          do train_stage "$m" train_dl; done
for m in tide tsmixer;              do train_stage "$m" train_darts; done

# ---- comparison, reporting, promotion --------------------------------------
run evaluate_compare "$PYTHON" -m demand_forecasting.evaluate_compare --config "$CONFIG"
run report_best      "$PYTHON" -m demand_forecasting.report_best_models --config "$CONFIG"

run promote "$PYTHON" - "$CONFIG" <<'EOF'
import subprocess
import sys

import pandas as pd
import yaml

config_path = sys.argv[1]
cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
metric = cfg["training"].get("selection_metric", "wape")
save_dir = cfg["training"]["save_dir"]

comparison = pd.read_csv("reports/model_comparison.csv")
champion = comparison.sort_values(f"val_{metric}").iloc[0]["model"]

FAMILY = {
    "lightgbm": ("ml", "model.joblib"), "xgboost": ("ml", "model.joblib"), "catboost": ("ml", "model.joblib"),
    "lstm": ("dl", "model.pt"), "transformer": ("dl", "model.pt"),
    "tide": ("darts", "model.pt"), "tsmixer": ("darts", "model.pt"),
}
family, artifact = FAMILY[champion]

cmd = [
    sys.executable, "-m", "demand_forecasting.select_model",
    "--config", config_path, "--metric", metric,
    "--name", champion, "--family", family, "--model-type", champion,
    "--artifact", f"{save_dir}/{champion}/{artifact}",
    "--metrics", f"{save_dir}/{champion}/metrics.json",
]
if family == "darts":
    cmd += ["--metadata-path", f"{save_dir}/{champion}/metadata.joblib"]

print(f"Promoting {champion} (val_{metric})")
subprocess.run(cmd, check=True)
EOF

# ---- 14-day forecast from the champion --------------------------------------
run make_future_template "$PYTHON" -m demand_forecasting.make_future_template --config "$CONFIG"

echo "[$(stamp)] RUN COMPLETE" | tee -a "$STATUS"
