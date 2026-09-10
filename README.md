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

## 0. The v1 production run

The full-dataset run, tuned for every model and tracked in DagsHub MLflow:

```bash
PYTHON=./venv/bin/python bash scripts/run_v1.sh
```

Then generate the deliverable Store-SKU forecast from each family's champion:

```bash
python -m demand_forecasting.run_final_inference
```

A second, larger experiment (25 tree trials, 4 sequence trials, 50 epochs) runs from its own config
and writes to its own experiment and artifact directory, so v1 is never overwritten:

```bash
PYTHON=./venv/bin/python CONFIG=configs/config_v2.yaml bash scripts/run_v1.sh
```

Key deliverables:

| File | Contents |
|---|---|
| `reports/v1/forecast_store_sku.csv` | the 14-day forecast, one row per Store-SKU-date |
| `reports/v1/store_sku_forecast_and_metrics.csv` | forecast **and** that series' test accuracy, worst first |
| `reports/v1/best_models.csv` | the champion of each family with train/val/test metrics |
| `configs/model_registry.yaml` | the promoted model the API serves |

`run_v1.sh` differs from `run_all.sh` in three ways: per-family trial counts (tree models are ~20×
cheaper per trial than sequence models), a `PASS`/`FAIL` status trail in
`reports/v1_run_status.txt` so one model failing cannot abort the other six, and a fallback to
default hyperparameters when a tuning stage produced no params file.

Results, artifact locations and the v1 post-mortem are in `architecture_flow.md` §9.

---

## 0b. Run everything (generic)

The whole pipeline — dataset, tuning for every model, training, evaluation, comparison and
promotion — with every experiment logged to MLflow and to `logs/`:

```bash
export PYTHONPATH=$PWD/src
./scripts/run_all.sh
```

Rank and tune on validation MAPE instead of the default WAPE:

```bash
METRIC=mape ./scripts/run_all.sh
```

Skip the slow Darts models:

```bash
SKIP_DARTS=1 ./scripts/run_all.sh
```

What it produces:

| Output | Contents |
|---|---|
| `data/processed/features.parquet` | the point-in-time feature table |
| `artifacts/tuning/<model>_best_params.json` | winning hyperparameters per model |
| `artifacts/tuning/<model>_trials.csv` | every trial with its params and metrics |
| `artifacts/<model>/model.*` | the trained model (refit on train+validation) |
| `artifacts/<model>/{validation,test}_predictions.csv` | per-row predictions |
| `artifacts/<model>/metrics.json` | train/val/test metrics + breakdowns |
| `reports/v1/model_comparison.csv` | every model ranked by the validation metric |
| `configs/model_registry.yaml` | the promoted champion |
| `logs/<step>.log` | full run narrative per step |
| `mlflow.db` | every run: tuning trials and final models |

The equivalent step-by-step commands are in sections 4–6 below; the sections after that cover
logging, MLflow, DVC, inference and monitoring.

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
pytest tests -q                   # 25 tests
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
  n_jobs: -1               # -1 = auto (half the CPUs). See the warning below.
  mlflow_experiment: fmcg-demand-forecasting

bayes:
  n_trials: 25             # raise for a serious tuning run

logging:
  dir: logs
  level: INFO
  console: true
  backup_count: 14
```

> **Do not set `training.n_jobs` to your full CPU count.** LightGBM, XGBoost and CatBoost all use
> OpenMP, and when the thread count saturates the machine they collapse into spin-wait contention.
> Measured on this 16-CPU WSL2 box, one 100-tree LightGBM fit on 20,780 rows takes **0.31s at 8
> threads and 235s at 16** — a ~750x difference; XGBoost showed 0.45s vs 39s. `-1` now resolves to
> half the available CPUs (`resolve_n_jobs()` in `models/ml.py`), and an explicit saturating value
> logs a warning. Before this was fixed, `tune_bayesian` could not finish a single trial in 15
> minutes on 20 series; afterwards the whole tune-plus-train sequence finished in about 90 seconds.

> **Memory.** `recursive_forecast` trims history to the longest feature window
> (`required_history_days()`, 70 days) before rebuilding features, which cut peak RSS from ~10 GB
> (OOM-killed on a 11.9 GB box) to 3.47 GB and made it 3.1× faster. The trim is pinned by an
> equivalence test asserting bit-identical predictions. If you add a feature with a window longer
> than 28 days, extend `required_history_days()` or it will be silently starved of history.

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
test 14,056.

### What features the models actually use

Roughly 52 numeric + 9 categorical columns reach the tree models. Built by
`features.build_causal_features()`, every target-derived value shifted **before** rolling:

| Group | Features |
|---|---|
| Demand history | `demand_lag_{1,7,14,28}`; shifted rolling mean/std/max over 7/14/28 days |
| Calendar | year, month, day, weekday, weekofyear, `is_weekend`, `is_holiday`, plus cyclical `dow_sin/cos`, `month_sin/cos`, `doy_sin/cos` |
| Promotions | `promo_flag` (known future), `promo_prev_1`, `promo_rate_28` |
| Price | `list_price_lag_1`, `discount_pct_lag_1`, `list_price_mean_28`, `discount_pct_mean_28` |
| Inventory history | `stockout_lag_{1,7,14}`, `stockout_rate_28`, `stock_on_hand_lag_1`, `stock_on_hand_mean_7` |
| Cross-series | `store_demand_mean_lag_1`, `sku_demand_mean_lag_1` (day-lagged group means) |
| Series identity | `store_id`, `sku_id`, `store_sku_id`, `channel`, `category`, `subcategory`, `brand`, `country`, `city`, `series_age_days` |

**Excluded on purpose:** `gross_sales` and `net_sales` (deterministic functions of the target);
same-day `stock_out_flag` and `stock_on_hand` (unknown at forecast time); `purchase_cost` /
`margin_pct` (kept only for the business-cost proxy); `supplier_id`, `sku_name`, `latitude`,
`longitude`; and — under the current contract — same-day `list_price`, `discount_pct`,
`temperature`, `rain_mm`, because `price_known_future` and `weather_known_future` are `false`.

The deep models consume the same information in tensor form: a 56-day lookback of demand and
dynamic covariates, 14 days of known-future covariates, and six static embedding IDs.

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

# or optimise a different validation metric
python -m demand_forecasting.tune_bayesian --model lightgbm --metric mape
```

Each trial is scored by a **true recursive 14-day validation forecast** — the same way the model is
deployed. Writes `artifacts/tuning/<model>_best_params.json` and `<model>_trials.csv`.

**Every trial is logged to MLflow.** Each study opens a parent run (`<model>-tuning`,
`stage=tuning_parent`) with one nested run per trial (`<model>-tune-trial-000`, `stage=tuning`)
recording the model name, that trial's hyperparameters, and the full validation metric set
(`val_wape`, `val_mape`, `val_mae`, `val_rmse`, `val_smape`, `val_bias`). The parent additionally
logs the best params, `best_val_*` metrics, and both tuning files as artifacts.

Trial **model binaries** are off by default because they are large (25 trials × 3 models can reach
several GB). Params and metrics for every trial are always logged. To persist the models too:

```yaml
# configs/config.yaml
bayes:
  log_trial_models: true
```

Each trial then also writes `artifacts/tuning/<model>_trial_<n>.joblib` and logs it to its run.

```bash
python -m demand_forecasting.train_ml --model lightgbm --params-json artifacts/tuning/lightgbm_best_params.json
python -m demand_forecasting.train_ml --model xgboost  --params-json artifacts/tuning/xgboost_best_params.json
python -m demand_forecasting.train_ml --model catboost --params-json artifacts/tuning/catboost_best_params.json
```

Or all three end to end:

```bash
./scripts/run_ml_experiments.sh
```

### Sequence models — tuning searches the input chunk length

`tune_dl.py` runs the same Optuna TPE search for `lstm`, `transformer`, `tide` and `tsmixer`, and
searches **`lookback` (input chunk length)** from one horizon up to 112 days in weekly steps,
alongside learning rate, dropout, batch size and the per-architecture sizes:

```bash
python -m demand_forecasting.tune_dl --model lstm
python -m demand_forecasting.tune_dl --model transformer
python -m demand_forecasting.tune_dl --model tide --n-trials 5
python -m demand_forecasting.tune_dl --model tsmixer
```

Trial count is `bayes.n_trials_dl` (default 10, lower than the tree models because each trial trains
a network). Every trial is a nested MLflow run, same as the tree tuner.

### Deep learning

```bash
python -m demand_forecasting.train_dl --model lstm --params-json artifacts/tuning/lstm_best_params.json
python -m demand_forecasting.train_dl --model transformer --params-json artifacts/tuning/transformer_best_params.json
```

Validation drives early stopping; the model is then refit on train + validation for the
validation-selected epoch count, so the test window never influences training.

### Darts global models

```bash
python -m demand_forecasting.train_darts --model tide --params-json artifacts/tuning/tide_best_params.json
python -m demand_forecasting.train_darts --model tsmixer --params-json artifacts/tuning/tsmixer_best_params.json
```

Like the other trainers, this evaluates validation from the train-end origin, refits on
train+validation, scores the test window once, and writes `metrics.json` plus both prediction CSVs —
so TiDE/TSMixer take part in `evaluate_compare` and can be promoted.

Every training run writes `artifacts/<model>/` (model, `validation_predictions.csv`,
`test_predictions.csv`, `metrics.json`), one MLflow run, and one entry in `logs/<step>.log`.

---

## 6. Compare and promote

```bash
python -m demand_forecasting.evaluate_compare                  # ranks by config selection_metric
python -m demand_forecasting.evaluate_compare --metric mape    # rank by validation MAPE
```

Ranks every model in `artifacts/*/metrics.json` and writes `reports/v1/model_comparison.csv` with
`train_wape`, `train_mape`, and the full `val_*` / `test_*` metric set for each. Then inspect the
validation breakdowns by channel, category and promo, plus operational properties (training time,
serving complexity). Test is for final unbiased reporting, not repeated model selection.

```bash
python -m demand_forecasting.select_model \
  --name lightgbm \
  --family ml \
  --artifact artifacts/lightgbm/model.joblib \
  --metrics artifacts/lightgbm/metrics.json

# promoting on MAPE instead (must match how you ranked)
python -m demand_forecasting.select_model --metric mape --name lstm --family dl --model-type lstm \
  --artifact artifacts/lstm/model.pt --metrics artifacts/lstm/metrics.json
```

This refuses to promote anything that is not the current validation champion **under the metric you
pass**, and writes `configs/model_registry.yaml` recording both `selection_metric` and the winning
`selection_value`. For a Darts model add `--metadata-path artifacts/tide/metadata.joblib
--model-type tide`; for a DL model use `--family dl`.

### Best model per family, with test inference vs actuals

```bash
python -m demand_forecasting.report_best_models                  # uses config selection_metric
python -m demand_forecasting.report_best_models --metric mape    # best by lowest val MAPE
```

Picks the lowest `val_<metric>` model **within each family** (ml / dl / darts) and scores its test
predictions against actuals:

| File | Contents |
|---|---|
| `reports/v1/all_models_metrics.csv` | every model, train/val/test × wape, mape, mae, rmse, smape, bias |
| `reports/v1/best_models.csv` | the champion of each family, ranked overall |
| `reports/v1/best_models_test_predictions.csv` | per-row test predictions vs actuals, with `error` and `horizon_day` |
| `reports/v1/best_models_per_series_metrics.csv` | metrics per Store-SKU, worst first |
| `reports/v1/best_models_per_horizon_metrics.csv` | metrics per horizon day D+1 … D+14 |
| `reports/v1/best_models_pooled_vs_macro.csv` | pooled vs macro WAPE, best/worst series |

The per-horizon table is the one to read first — it shows error growth across the 14 days that the
single pooled number hides.

### Choosing between WAPE and MAPE

Both are always computed and logged — `selection_metric` only decides the ranking key. But the
choice is not cosmetic: **it can change which model gets promoted.** From an actual run on a
12-series subset:

| model | val_wape | val_mape | test_wape | test_mape |
|---|---|---|---|---|
| lstm | 0.2552 | **0.3151** | 0.3024 | 0.8697 |
| lightgbm | **0.2511** | 0.3336 | 0.2821 | 0.9060 |
| xgboost | 0.2737 | 0.3513 | 0.3121 | 0.8865 |
| tide | 0.3742 | 0.4576 | 0.3781 | 1.0791 |

WAPE picks **lightgbm**; MAPE picks **lstm**. Note also how MAPE nearly triples from validation
(~0.32) to test (~0.87–1.08) while WAPE moves modestly (~0.25 → ~0.28). That is MAPE's known
instability: it divides by each actual, and ~1.5% of rows have demand below 5 units, where a
2-unit miss reads as a 40–200% error. Zero-demand rows (0.28% of the data) are excluded entirely —
`val_mape_coverage` in the comparison reports the fraction actually scored.

**Recommendation: keep `selection_metric: wape`** for promotion decisions, and read MAPE alongside
it. WAPE is volume-weighted, always defined, and far more stable across windows. Use `--metric mape`
when a stakeholder specifically asks to be ranked on it.

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

With no environment variables set, **MLflow 3.x defaults to a local sqlite store at
`./mlflow.db`** (not `./mlruns`). The legacy `./mlruns` file store is in maintenance mode and now
raises unless you set `MLFLOW_ALLOW_FILE_STORE=true`, so use sqlite locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
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
CSVs. Compare runs by sorting on `val_wape` (or `val_mape`) in the MLflow UI.

Runs are tagged so you can filter them:

| Tag | Values | Meaning |
|---|---|---|
| `stage` | `tuning_parent` | one per tuning study |
| | `tuning` | one per Optuna trial (nested under the parent) |
| | `final` | the trained, evaluated, promotable model |
| `model` | `lightgbm`, `xgboost`, `catboost`, `lstm`, `transformer`, `tide`, `tsmixer` | |
| `family` | `ml`, `dl`, `darts` | which inference adapter serves it |

Useful MLflow UI filters:

```text
tags.stage = 'final'                       # just the comparable models
tags.stage = 'tuning' and tags.model = 'lightgbm'   # every LightGBM trial
```

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

Generate the future covariates rather than hand-writing them — the required fields are derived from
the `*_known_future` contract, so the request can never drift from the model:

```bash
python -m demand_forecasting.make_future_template \
  --output-csv reports/future_covariates.csv \
  --output-json reports/forecast_request.json
# --series-limit 2 for a small smoke test
```

This produces exactly 14 contiguous days per active store-SKU starting the day after history ends.
Then:

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

### Batch inference from every model

`run_final_inference.py` runs one forecast per model listed in a manifest CSV and writes a single
combined output, so all six models can be compared side by side on the same 14-day horizon.

Note the shipped `reports/v2/best_models.csv` holds only the **three per-family winners**
(transformer, catboost, tide). To forecast with all six, use the full manifest:

```bash
export PYTHONPATH=src

python -m demand_forecasting.run_final_inference \
  --config configs/config.yaml \
  --best-models reports/v2/all_models_for_inference.csv \
  --outdir reports/v2_all
```

The manifest needs only three columns — `model`, `family`, `artifact_path`. It is ordered
cheapest-first, because the `ml` adapter never imports torch or darts:

| model | family | artifact |
|---|---|---|
| catboost | ml | `artifacts_v2/catboost/model.joblib` |
| lightgbm | ml | `artifacts_v2/lightgbm/model.joblib` |
| xgboost | ml | `artifacts_v2/xgboost/model.joblib` |
| lstm | dl | `artifacts_v2/lstm/model.pt` |
| transformer | dl | `artifacts_v2/transformer/model.pt` |
| tide | darts | `artifacts_v2/tide/model.pt` (+ `metadata.joblib` alongside) |

Outputs land in `--outdir`:

| File | Shape |
|---|---|
| `forecast_store_sku.csv` | long — `date, store_id, sku_id, prediction, model, family` |
| `forecast_store_sku_wide.csv` | wide — one row per Store-SKU-date, one column per model |
| `store_sku_forecast_and_metrics.csv` | per Store-SKU per model: forecast totals joined to held-out test accuracy |

The third file joins against `best_models_per_series_metrics.csv` **inside `--outdir`**. That file
ships covering only the three champions, so a six-model run needs the six-model version generated
first or the test-accuracy columns come back empty for lightgbm, xgboost and lstm. It is derived
from each model's `artifacts_v2/<model>/test_predictions.csv`.

A failure in one model is logged and skipped rather than aborting the run, so a broken adapter never
costs you the other five forecasts.

#### Verified run

Six models, 1,004 series, horizon `2024-01-01..2024-01-14`: 84,336 forecast rows, zero nulls, zero
negatives. Wall clock ~6.5 min total.

| model | family | time | 14-day total units |
|---|---|---|---|
| catboost | ml | 40s | 804,418 |
| lightgbm | ml | 33s | 790,566 |
| xgboost | ml | 32s | 779,889 |
| lstm | dl | 61s | 761,973 |
| transformer | dl | 4s | 808,023 |
| tide | darts | 3m 27s | 762,232 |

The LSTM's 61s is almost entirely the one-time `torch` import — the transformer, running straight
after it, took 4s. Order the manifest so the tree models run first and that import cost is paid once.

The forward forecasts corroborate the held-out bias finding independently: the transformer projects
808,023 units against the LSTM's 761,973, a 6.0% spread on the same 14 days with no ground truth
involved. That is the same over-forecasting the test window showed at +7.32%, which is why CatBoost
rather than the validation winner is the production recommendation.

Models disagree by a mean of 8.84 units per Store-SKU-day (median 7.01, p95 20.75, max 67.08).
`forecast_store_sku_wide.csv` puts them side by side for exactly this comparison.

### API

The service loads whichever model the **serving registry** names. Two ready-made registries exist:

| Registry | Serves | Start-up | When to use |
|---|---|---|---|
| `configs/serving_registry.yaml` | CatBoost (`family: ml`) | ~7s | Default. Near-unbiased on test (+0.99%), no torch import. |
| `configs/model_registry_v2.yaml` | Transformer (`family: dl`) | ~145s | Best validation WAPE, but +7.32% test bias. |

`configs/model_registry_all_v2.yaml` is **not** a serving registry — it is the audit record of all
six models and has a `models:` list rather than a `selected_model:` entry. Pointing the API at it
fails with `No selected_model entry found`.

#### 1. Build the request body

The API needs one row per Store-SKU per future date carrying only genuinely-known-future fields.
Generate it from history rather than writing it by hand:

```bash
export PYTHONPATH=src

# Full horizon: 1,004 series x 14 days = 14,056 rows
python -m demand_forecasting.make_future_template \
  --config configs/config.yaml \
  --output-csv  reports/future_covariates.csv \
  --output-json reports/forecast_request.json

# Small body for a smoke test: 3 series x 14 days = 42 rows
python -m demand_forecasting.make_future_template \
  --config configs/config.yaml --series-limit 3 \
  --output-csv  reports/future_covariates_smoke.csv \
  --output-json reports/forecast_request_smoke.json
```

The fields emitted are driven by the `*_known_future` contract in the config, so the body changes
automatically if that contract changes. With the current settings it is
`date, store_id, sku_id, is_holiday, promo_flag` — price and weather are deliberately absent
because they are not known at the forecast origin.

#### 2. Start the server

Environment variables are read by the **server** process, so they must be exported in the terminal
running uvicorn — not in the terminal running curl.

```bash
export PYTHONPATH=src
export MODEL_REGISTRY_PATH=configs/serving_registry.yaml
export CONFIG_PATH=configs/config.yaml
export HISTORY_CSV=data/raw/data.csv

python -m uvicorn demand_forecasting.api:app --host 127.0.0.1 --port 8000
```

Note the module path is `demand_forecasting.api:app` with `PYTHONPATH=src` — **not**
`src.demand_forecasting.api:app`. The saved model artifacts were pickled against the
`demand_forecasting.*` module path, so importing the package under a different name makes
`joblib.load` fail with `No module named 'demand_forecasting'`.

Wait for the health check before posting:

```bash
until curl -s http://127.0.0.1:8000/health; do sleep 1; done
```

#### 3. Call it

```bash
# Smoke test: 3 series, returns 42 rows
curl -s -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d @reports/forecast_request_smoke.json \
  -o reports/forecast_response_smoke.json \
  -w "HTTP %{http_code}  %{time_total}s\n"

# Full horizon: all 1,004 series
curl -s -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d @reports/forecast_request.json \
  -o reports/forecast_response.json \
  -w "HTTP %{http_code}  %{time_total}s\n"

# Or by hand - one row per series per future date
curl -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"future_covariates":[
        {"date":"2024-01-01","store_id":"STORE0001","sku_id":"SKU0086","is_holiday":1,"promo_flag":1},
        {"date":"2024-01-02","store_id":"STORE0001","sku_id":"SKU0086","is_holiday":0,"promo_flag":0}
      ]}'
```

Response shape:

```json
{
  "forecast_horizon": 14,
  "forecast_count": 42,
  "forecasts": [
    {"date":"2024-01-01","store_id":"STORE0001","sku_id":"SKU0001","prediction":101.93},
    {"date":"2024-01-01","store_id":"STORE0001","sku_id":"SKU0002","prediction":69.81}
  ]
}
```

A 3-series call takes ~55s: the cost is dominated by reading the 1.07M-row history and running the
14 recursive feature-rebuild passes, both of which are near-constant regardless of how many series
are requested. A full 1,004-series call is not much slower.

#### Docker

```bash
docker build -t fmcg-forecast-api .

docker run --rm -p 8000:8000 \
  -e MODEL_REGISTRY_PATH=configs/serving_registry.yaml \
  -v "$PWD/data:/app/data" \
  -v "$PWD/artifacts_v2:/app/artifacts_v2" \
  -v "$PWD/configs:/app/configs" \
  -v "$PWD/logs:/app/logs" \
  fmcg-forecast-api
```

Mount data, artifacts and configs rather than baking them into the image; mount `logs` so the
service's rotating log survives the container. The Dockerfile already sets `PYTHONPATH=src`.

#### Failure modes

| Response | Cause | Fix |
|---|---|---|
| `Model registry not found: <path>` | `MODEL_REGISTRY_PATH` names a file that does not exist | Use `configs/serving_registry.yaml`; check for typos |
| `No selected_model entry found` | Pointed at an audit registry (`model_registry_all_v2.yaml`) | Use a registry with a `selected_model:` block |
| `No module named 'demand_forecasting'` | Started as `src.demand_forecasting.api:app` | Use `PYTHONPATH=src` + `demand_forecasting.api:app` |
| 422 with `Field required` | Body is not wrapped in `{"future_covariates": [...]}` | Regenerate with `make_future_template` |
| 422 about dates or duplicates | Not exactly `horizon` contiguous days per series from the day after history ends | Regenerate with `make_future_template` |

To serve a different model, point `MODEL_REGISTRY_PATH` at another registry — the API is
model-agnostic and `inference_router.py` dispatches on `family` (`ml` / `dl` / `darts`). Note the
Darts adapter requires **every trained series** in the request, so a single-series call fails when a
Darts model is promoted; the ML and DL adapters accept any subset.

The request must contain exactly `horizon` contiguous days per series, starting the day after the
history ends, with no duplicates; anything else is rejected with a 422 explaining why. Static
attributes are joined from each series' latest history row.

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
| Which model wins? | `reports/v1/model_comparison.csv`, sorted by `val_wape` |
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
