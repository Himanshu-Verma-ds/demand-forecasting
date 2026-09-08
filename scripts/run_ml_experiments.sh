#!/usr/bin/env bash
# Tune, train and compare all three tree models.
# Every step writes its own rotating log file under logs/.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

for m in lightgbm xgboost catboost; do
  python -m demand_forecasting.tune_bayesian --model "$m"
  python -m demand_forecasting.train_ml --model "$m" --params-json "artifacts/tuning/${m}_best_params.json"
done

python -m demand_forecasting.evaluate_compare
