#!/usr/bin/env bash
cd /mnt/d/Himanshu/code/fmcg_demand_forecasting_project/ml_assignment_project || exit 1
export PYTHONPATH=src
PY=./venv/bin/python
C=_audit/config.yaml

run() {
  local name="$1"; shift
  echo "### $name"
  if "$@" >_audit/out.txt 2>&1; then
    echo "PASS $name"
  else
    echo "FAIL $name"
    tail -20 _audit/out.txt
  fi
}

run prepare_dataset $PY -m demand_forecasting.prepare_dataset --config $C --output _audit/features.parquet
run tune_lightgbm   $PY -m demand_forecasting.tune_bayesian --config $C --model lightgbm
run train_lightgbm  $PY -m demand_forecasting.train_ml --config $C --model lightgbm --params-json _audit/artifacts/tuning/lightgbm_best_params.json
run train_xgboost   $PY -m demand_forecasting.train_ml --config $C --model xgboost
run train_catboost  $PY -m demand_forecasting.train_ml --config $C --model catboost
run train_lstm      $PY -m demand_forecasting.train_dl --config $C --model lstm
run train_transformer $PY -m demand_forecasting.train_dl --config $C --model transformer
run train_tide      $PY -m demand_forecasting.train_darts --config $C --model tide
run evaluate_compare $PY -m demand_forecasting.evaluate_compare --config $C --artifacts _audit/artifacts --output _audit/model_comparison.csv

echo "=== comparison ==="
cat _audit/model_comparison.csv 2>/dev/null | cut -d, -f1-6
echo "=== artifact tree ==="
find _audit/artifacts -maxdepth 2 -type f | sort
