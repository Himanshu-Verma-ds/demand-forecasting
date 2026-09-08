# FMCG Multi-Store Demand Forecasting

A production-oriented, reproducible 14-day demand-forecasting system built for the assignment in
`problem_statement/problem_statement.pdf`.

Forecasts daily `units_sold` for every active `(store_id, sku_id)` pair over a 14-day horizon,
using global tree models, from-scratch PyTorch sequence models and Darts global models — with
leakage-safe evaluation, MLflow/DagsHub tracking, DVC data versioning, rotating per-step logs,
drift monitoring, and both batch and API inference.

| Document | Purpose |
|---|---|
| `README.md` | how to install, run, log, inspect and analyse (this file) |
| `architecture_flow.md` | what every file does, how data flows, worked examples |
| `explanation.md` | design rationale and mapping to each assignment requirement |
| `notebooks/01_eda.ipynb` | decision-oriented Plotly EDA |

---

## Dataset

1,100,000 rows, 2021-01-01 to 2023-12-31 (1,095 days), 13 stores, 102 SKUs, 1,005 observed
store-SKU series, 5 categories, 17 subcategories, 4 channels. 1,004 pairs stay active through the
final date; one ends in 2022. Promotions on ~8.0% of rows, stockouts on ~3.0%.

Place the supplied CSV at `data/raw/data.csv`. **Do not commit it to Git** — track it with DVC.

---

## Project structure

```text
configs/config.yaml              every runtime setting (single source of truth)
configs/model_registry.yaml      the currently promoted champion model
data/raw/data.csv                supplied dataset (DVC-tracked)
data/processed/features.parquet  generated feature table (DVC output)
logs/                            one midnight-rotating log file per step
artifacts/                       models, predictions, metrics, tuning results
reports/                         model comparison, drift reports, profiles
notebooks/01_eda.ipynb           EDA
src/demand_forecasting/          the package
tests/                           leakage, split and metric tests
dvc.yaml, params.yaml            DVC pipeline
Dockerfile                       packaged inference service
```

Inside `src/demand_forecasting/`:

```text
config.py           load_config()
logging_utils.py    setup_logging() + rotating file handlers
tracking.py         MLflow configuration and artifact helpers
data.py             read_raw(), prepare_features()
stockout.py         censored-demand target and sample weights
features.py         calendar / lag / rolling / promo features, blocked-column policy
splits.py           chronological train / validation / test boundaries
metrics.py          WAPE, MAE, RMSE, SMAPE, bias, business proxy, breakdowns
hierarchy.py        bottom-up aggregation, middle-out allocation
models/ml.py        MLBundle + LightGBM / XGBoost / CatBoost pipelines
models/dl_data.py   windowed dataset, scaling metadata, static encoding
models/lstm.py      from-scratch encoder-decoder LSTM
models/transformer.py  from-scratch temporal Transformer (direct 14-step output)
prepare_dataset.py  step: build the feature table
tune_bayesian.py    step: Optuna TPE search on recursive validation WAPE
train_ml.py         step: train + recursively evaluate tree models
train_dl.py         step: train + evaluate LSTM / Transformer
train_darts.py      step: train TiDE / TSMixer
evaluate_compare.py step: rank all models by validation WAPE
select_model.py     step: promote the champion to the registry
inference_ml.py     step: recursive leakage-safe tree forecasting
inference_dl.py     step: direct multi-horizon DL forecasting
inference_darts.py  step: TiDE / TSMixer forecasting
inference_router.py dispatch to the promoted model's family
api.py              step: FastAPI POST /forecast
drift.py            step: PSI / KS / promo-regime monitoring
strategy_analysis.py step: operational size of each modelling strategy
```

---

## 1. Setup

Python 3.11+ (developed and verified on 3.13). GPU is used automatically by the PyTorch and Darts
models when CUDA is available.

```bash
python -m venv venv
source venv/bin/activate          # Windows PowerShell: venv\Scripts\Activate.ps1
pip install -r requirements.txt

export PYTHONPATH=$PWD/src        # Windows PowerShell: $env:PYTHONPATH="$PWD\src"
```

Copy `.env.example` to `.env` and fill it in if you are using DagsHub. `.env` is gitignored —
never commit a token.

Run the tests to confirm the install:

```bash
pytest tests -q                   # 15 tests
```

---

## 2. Configuration

Everything is driven by `configs/config.yaml`. Nothing is hard-coded in a training script, so
changing a setting there changes it consistently across feature building, training, all inference
paths and API request validation.

The settings you are most likely to change:

```yaml
data:
  horizon: 14              # forecast horizon
  lookback: 56             # DL history window

features:
  stockout_target_mode: none   # none | rolling_median | percentage
  promo_known_future: true     # promo calendar is committed before the origin
  price_known_future: false    # realized future prices NOT assumed known
  weather_known_future: false  # 14-day weather forecast NOT assumed available

training:
  max_train_rows: null     # null = use everything; set a number for fast debugging
  mlflow_experiment: fmcg-demand-forecasting

bayes:
  n_trials: 25             # raise for a serious tuning run

logging:
  dir: logs
  level: INFO
  console: true
  backup_count: 14
```

---

## 3. EDA

```bash
jupyter lab notebooks/01_eda.ipynb
```

Twelve sections, each ending with what the signal implies for model design: schema and
completeness, hierarchy integrity, target distribution, trend and weekly/monthly seasonality,
interactive series examples, promo lift and discount depth, the stockout censored-demand
diagnostic, price and weather, sparse and cold-start diagnostics, the leakage-safe split, the
global-vs-segmented-vs-local strategy comparison, and what to run next.

Sizing the modelling strategies is also available as a script:

```bash
python -m demand_forecasting.strategy_analysis     # → reports/strategy_analysis.csv
```

---

## 4. Build the feature table

```bash
python -m demand_forecasting.prepare_dataset \
  --config configs/config.yaml \
  --output data/processed/features.parquet
```

Produces 1,100,000 rows × 72 columns, split-labelled as train 1,071,888 / validation 14,056 /
test 14,056. Every demand-derived feature is shifted before rolling, and `gross_sales` / `net_sales`
are permanently excluded.

Split boundaries (derived from the last date, so they update automatically if the data grows):

```text
train      2021-01-01 .. 2023-12-03
validation 2023-12-04 .. 2023-12-17    model selection and tuning
test       2023-12-18 .. 2023-12-31    evaluated once, for final reporting
```

---

## 5. Tune and train

### Tree models

```bash
python -m demand_forecasting.tune_bayesian --model lightgbm
python -m demand_forecasting.tune_bayesian --model xgboost
python -m demand_forecasting.tune_bayesian --model catboost
```

Each trial is scored by the **true recursive 14-day validation WAPE** — the same way the model is
deployed. Writes `artifacts/tuning/<model>_best_params.json` and `<model>_trials.csv`.

```bash
python -m demand_forecasting.train_ml --model lightgbm --params-json artifacts/tuning/lightgbm_best_params.json
python -m demand_forecasting.train_ml --model xgboost  --params-json artifacts/tuning/xgboost_best_params.json
python -m demand_forecasting.train_ml --model catboost --params-json artifacts/tuning/catboost_best_params.json
```

Or all three end to end:

```bash
./scripts/run_ml_experiments.sh
```

### Deep learning

```bash
python -m demand_forecasting.train_dl --model lstm
python -m demand_forecasting.train_dl --model transformer
```

Validation drives early stopping; the model is then refit on train + validation for the
validation-selected epoch count, so the test window never influences training.

### Darts global models

```bash
python -m demand_forecasting.train_darts --model tide
python -m demand_forecasting.train_darts --model tsmixer
```

Every training run writes `artifacts/<model>/` (model, `validation_predictions.csv`,
`test_predictions.csv`, `metrics.json`), one MLflow run, and one entry in `logs/<step>.log`.

---

## 6. Compare and promote

```bash
python -m demand_forecasting.evaluate_compare       # → reports/model_comparison.csv
```

Ranks every model in `artifacts/*/metrics.json` by **validation WAPE**. Then inspect the validation
breakdowns by channel, category and promo, plus operational properties (training time, serving
complexity). Test is for final unbiased reporting, not repeated model selection.

```bash
python -m demand_forecasting.select_model \
  --name lightgbm \
  --family ml \
  --artifact artifacts/lightgbm/model.joblib \
  --metrics artifacts/lightgbm/metrics.json
```

This refuses to promote anything that is not the current validation champion, and writes
`configs/model_registry.yaml`. For a Darts model add `--metadata-path artifacts/tide/metadata.joblib
--model-type tide`; for a DL model use `--family dl`.

---

## 7. Logs

Every step writes its own log file under `logs/`, rotating at midnight.

```text
logs/prepare_dataset.log       logs/train_darts.log       logs/inference_ml.log
logs/tune_bayesian.log         logs/evaluate_compare.log  logs/inference_dl.log
logs/train_ml.log              logs/select_model.log      logs/inference_darts.log
logs/train_dl.log              logs/drift.log             logs/api.log
```

Yesterday's file is kept as `logs/train_ml.log.2026-09-07`, retained for `logging.backup_count`
days (default 14).

Each run records the run context (timestamp, Python and platform, package version, working
directory, CLI arguments), **the entire effective configuration**, the settings actually used
(hyperparameters and whether they came from a tuned JSON or the defaults, split boundaries, device,
feature counts, MLflow run id), progress milestones, the resulting metrics, and every artifact path
written.

```bash
tail -f logs/train_ml.log                  # follow a run live
grep "Validation metrics" logs/train_ml.log
grep "Best trial" -A 20 logs/tune_bayesian.log
ls logs/                                   # rotated history
```

Turn off console mirroring (file only) or change verbosity in `configs/config.yaml`:

```yaml
logging:
  level: DEBUG
  console: false
```

MLflow remains the experiment system of record; the log file is the chronological narrative you
read when a run fails or behaves unexpectedly, and it stays readable without a server.

---

## 8. MLflow — local or DagsHub

With no environment variables set, MLflow writes to `./mlruns`:

```bash
mlflow ui --backend-store-uri ./mlruns     # http://localhost:5000
```

For DagsHub, create a repository, obtain a token, then:

```bash
export DAGSHUB_USER=<user>
export DAGSHUB_REPO=<repo>
export DAGSHUB_TOKEN=<token>
source scripts/setup_dagshub.sh
```

That exports `MLFLOW_TRACKING_URI` / `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`, which
`tracking.configure_mlflow()` picks up automatically. Never commit the token.

Each run logs hyperparameters, the feature-contract flags, the stockout strategy, train/validation
/test metrics, business-proxy metrics, split-boundary tags, the model artifact, and both prediction
CSVs. Compare runs by sorting on `val_wape` in the MLflow UI.

---

## 9. DVC

DagsHub is the recommended remote here because the same project hosts both DVC storage and the
MLflow endpoint. Google Drive works too (needs `dvc-gdrive` and OAuth).

```bash
git init
dvc init
dvc add data/raw/data.csv
git add data/raw/data.csv.dvc .gitignore dvc.yaml params.yaml
git commit -m "Track forecasting dataset with DVC"
```

Then add the remote using the exact command generated in your DagsHub repository UI and run
`dvc push`. A collaborator runs `git clone` followed by `dvc pull`.

The `prepare` stage is defined in `dvc.yaml`, so the feature table can be rebuilt reproducibly:

```bash
dvc repro          # re-runs prepare only if data, config or the relevant code changed
dvc dag
```

---

## 10. Inference

### Batch

Build a future covariates file containing exactly 14 contiguous days per active store-SKU, with the
known-future fields (`date`, `store_id`, `sku_id`, `is_holiday`, and `promo_flag` while
`promo_known_future: true`), then:

```bash
python -m demand_forecasting.inference_ml \
  --model artifacts/lightgbm/model.joblib \
  --history data/raw/data.csv \
  --future reports/future_covariates.csv \
  --output artifacts/forecast_ml.csv

python -m demand_forecasting.inference_dl \
  --model artifacts/lstm/model.pt \
  --history data/raw/data.csv --future reports/future_covariates.csv \
  --output artifacts/forecast_dl.csv

python -m demand_forecasting.inference_darts \
  --model-type tide \
  --model artifacts/tide/model.pt --metadata artifacts/tide/metadata.joblib \
  --history data/raw/data.csv --future reports/future_covariates.csv \
  --output artifacts/forecast_darts.csv
```

All three write `date, store_id, sku_id, prediction`.

### Hierarchy

Bottom-level forecasts aggregate upward coherently with `hierarchy.aggregate_bottom_up()`; use
`trailing_shares()` + `middle_out_allocate()` when planners need stable parent totals allocated down
to SKUs.

### API

```bash
docker build -t fmcg-forecast-api .

docker run --rm -p 8000:8000 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/artifacts:/app/artifacts" \
  -v "$PWD/configs:/app/configs" \
  -v "$PWD/logs:/app/logs" \
  fmcg-forecast-api
```

Mount data, artifacts and configs rather than baking them into the image; mount `logs` so the
service's rotating log survives the container.

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"future_covariates":[
        {"date":"2024-01-01","store_id":"STORE0001","sku_id":"SKU0086","is_holiday":1,"promo_flag":1},
        {"date":"2024-01-02","store_id":"STORE0001","sku_id":"SKU0086","is_holiday":0,"promo_flag":0}
      ]}'
```

The request must contain exactly `horizon` contiguous days per series, starting the day after the
history ends, with no duplicates; anything else is rejected with a 422 explaining why. Static
attributes are joined from each series' latest history row. Locally, run it with
`uvicorn demand_forecasting.api:app --reload`.

---

## 11. Drift monitoring

Use a stable reference period (for example the last 8-12 training weeks) and the current scoring
window:

```bash
python -m demand_forecasting.drift \
  --reference reports/reference_window.csv \
  --current reports/current_window.csv \
  --output reports/drift_report.json
```

Reports numeric PSI/KS/mean shift/missingness, categorical PSI, and an explicit promo-regime check.
Thresholds live in `configs/config.yaml` (`psi_warning: 0.10`, `psi_alert: 0.25`,
`promo_rate_relative_change_alert: 0.30`). `logs/drift.log` ends with a
`N warnings, M alerts` summary.

Monitor three layers: input drift (price, discount, promo rate, stockout rate, category mix),
prediction drift before labels arrive (forecast mean and quantiles, zero-forecast rate, promo
uplift distribution), and performance drift once labels arrive (WAPE/MAE/RMSE overall and by
breakdown, bias, error by horizon day). Trigger retraining on sustained signals rather than daily.

---

## 12. Analysing results

| Question | Where to look |
|---|---|
| Which model wins? | `reports/model_comparison.csv`, sorted by `val_wape` |
| How did a specific run behave? | `logs/<step>.log` — full config, settings, progress, metrics |
| Compare many runs | MLflow UI, sort on `val_wape` |
| Where is the error concentrated? | `metrics.json` → `breakdowns` (channel, category, promo) |
| Per-row errors | `artifacts/<model>/validation_predictions.csv`, `test_predictions.csv` |
| Business exposure | `business_proxy` in `metrics.json` (under-forecast margin risk vs over-forecast cost) |
| What is currently served? | `configs/model_registry.yaml` |
| Has the input distribution moved? | `reports/drift_report.json` |
| Would a different strategy be cheaper? | `reports/strategy_analysis.csv` |

Select on **validation WAPE**, then check the breakdowns before promoting — WAPE is volume-weighted,
so a good aggregate number can hide poor performance on low-volume or promo-heavy segments.

---

## The central production rule

Validation and test are **never** evaluated by precomputing lags from their own true targets.
Tree models use recursive inference from a fixed origin, where each horizon day's lag features come
from earlier *predictions*. Deep models consume only the lookback window plus genuinely-known
future covariates. Scaling statistics and category maps are fitted on training rows only. Bayesian
tuning optimises validation WAPE, and the test window is evaluated exactly once.

`tests/test_leakage.py` asserts the feature-level half of this mechanically on every test run.
