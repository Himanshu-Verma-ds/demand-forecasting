# Inference bundle — v2 champion models

Everything needed to load the promoted models and produce a 14-day Store-SKU forecast, without
re-running training. The raw history is **not** copied here (212 MB, DVC-tracked); point the script
at `data/raw/data.csv`.

## Contents

```text
config_v2.yaml                        the exact config these models were trained under
model_registry_v2.yaml                the promoted champion (what the API would serve)
generate_forecast.py                  standalone script - loads a model and forecasts

model_registry_all_v2.yaml            all six models registered, ranked, with metrics

models/                               ALL SIX trained models, not just the per-family winners
├── lightgbm/     model.joblib + best_params.json               12 MB   ml adapter
├── xgboost/      model.joblib + best_params.json               16 MB   ml adapter
├── catboost/     model.joblib + best_params.json              4.1 MB   ml adapter
├── lstm/         model.pt     + best_params.json              2.9 MB   dl adapter
├── transformer/  model.pt     + best_params.json               11 MB   dl adapter
└── tide/         model.pt + model.pt.ckpt + metadata.joblib   3.2 MB   darts adapter

data/
├── future_covariates_2024-01-01_to_2024-01-14.csv    14,056 known-future rows (1,004 series x 14 days)
├── api_forecast_request.json                          the same rows as a POST /forecast body
├── forecast_all_models_2024-01-01_to_2024-01-14.csv   84,336 rows - ALL SIX models
├── forecast_output_2024-01-01_to_2024-01-14.csv       42,168 rows - the three per-family champions
└── store_sku_forecast_and_metrics.csv                 forecast + that series' test accuracy

metrics/
├── all_models_metrics.csv   every model, train/val/test x 6 metrics
├── best_models.csv          champion per family
├── model_comparison.csv     all six models ranked
└── <model>_metrics.json     train / validation / test metrics + breakdowns  (all six)
```

`tide/model.pt.ckpt` is required — Darts splits a saved model into a small config file
(`model.pt`) and the weights (`model.pt.ckpt`). Without the `.ckpt`, loading raises
*"The model must be fit before calling predict()"*.

## Reproduce the forecast

```bash
export PYTHONPATH=src          # from the project root, venv active

python final_conclusion/inference_bundle/generate_forecast.py \
    --history data/raw/data.csv \
    --model all \
    --output my_forecast.csv
```

Verified output — these totals reproduce `data/forecast_all_models_*.csv` exactly:

```text
history ends 2023-12-31
forecasting  2024-01-01 .. 2024-01-14 for 1004 Store-SKU series
  lightgbm     -> 14,056 rows, 790,566 total units
  xgboost      -> 14,056 rows, 779,889 total units
  catboost     -> 14,056 rows, 804,418 total units
  lstm         -> 14,056 rows, 761,973 total units
  transformer  -> 14,056 rows, 808,023 total units
  tide         -> 14,056 rows, 762,232 total units
  wrote 84,336 rows
```

`--model` accepts any of `lightgbm`, `xgboost`, `catboost`, `lstm`, `transformer`, `tide`, or `all`.

### Which model to load

| model | val WAPE | test WAPE | test bias | note |
|---|---|---|---|---|
| transformer | **0.2654** | 0.2747 | **+7.32%** | promoted champion, but over-forecasts badly |
| lstm | 0.2662 | **0.2673** | −1.10% | best test WAPE/MAPE/MAE |
| catboost | 0.2719 | 0.2683 | +0.99% | **recommended** — stable, cheap, explainable |
| xgboost | 0.2721 | 0.2714 | −1.27% | most conservative (under-forecasts) |
| lightgbm | 0.2722 | 0.2704 | +0.86% | **least biased of all six** (+0.06% on validation) |
| tide | 0.2762 | 0.2694 | −0.00% | near-perfect test bias, weakest WAPE |

## Forecasting a different horizon

The script rebuilds the known-future covariates from history by default, so the horizon always
starts the day *after* history ends. To forecast a later window, extend `data/raw/data.csv` with the
newer actuals and re-run — no retraining needed, though accuracy will drift as the data moves away
from the training period (see the drift section of the report).

To supply your own promo calendar rather than the default of "no promotion planned", edit
`data/future_covariates_*.csv` and pass it with `--future`. The required columns are driven by the
`*_known_future` contract in `config_v2.yaml`; under the current settings they are:

```text
date, store_id, sku_id, is_holiday, promo_flag
```

## Serving through the API instead

```bash
export MODEL_REGISTRY_PATH=final_conclusion/inference_bundle/model_registry_v2.yaml
export CONFIG_PATH=final_conclusion/inference_bundle/config_v2.yaml
export HISTORY_CSV=data/raw/data.csv

uvicorn demand_forecasting.api:app --port 8000

curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d @final_conclusion/inference_bundle/data/api_forecast_request.json
```

Note the registry points at `artifacts_v2/...`; change `artifact_path` to the bundled
`models/<name>/...` paths if you move this folder elsewhere.

## Caveat worth reading before using these numbers

The Transformer is the *promoted* model because it won validation, but it over-forecast the held-out
test window by **7.3%** (941,432 predicted vs 877,244 actual units). Every other model is within
±1.3%. For inventory decisions, prefer **CatBoost**, **LightGBM** or **TiDE**.

Note also that the six models disagree by **6%** on the out-of-sample total (808,023 vs 761,973
units) despite WAPEs within 1% of each other. There is no ground truth for that window, so consider
reporting the range — or an ensemble — rather than a single number. Reasoning in
`../final_report.md` §6, §7 and §11.
