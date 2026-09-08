# Architecture and Flow

This document is the map of the repository. It explains what every file does, how data moves
through the system, what each stage reads and writes, and what the logs and tracking artifacts
look like in practice.

For *why* each design decision was made, see `explanation.md`.
For *how to run* each stage, see `README.md`.

---

## 1. One-paragraph summary

The system forecasts daily `units_sold` for every active `(store_id, sku_id)` pair for the next
14 days. A single raw CSV (1.1M rows, 2021-01-01 to 2023-12-31, 13 stores, 102 SKUs, 1,005
store-SKU series) is turned into a point-in-time-correct feature table, split chronologically into
train / validation / test, and used to train four families of global models: tree models
(LightGBM, XGBoost, CatBoost), from-scratch PyTorch sequence models (LSTM, Transformer), and
Darts global models (TiDE, TSMixer). Models are selected on **validation WAPE only**, promoted to
a YAML registry, and served either as a batch job or through a FastAPI `POST /forecast` endpoint.
Every stage writes a rotating log file under `logs/`, and every training run writes params,
metrics and artifacts to MLflow (local `./mlruns` or DagsHub).

---

## 2. Repository layout

```text
ml_assignment_project/
├── configs/
│   ├── config.yaml                  # single source of truth for every runtime setting
│   └── model_registry.yaml          # the currently promoted champion model
├── data/
│   ├── raw/data.csv                 # supplied dataset (DVC-tracked, not in Git)
│   └── processed/features.parquet   # generated feature table (DVC output)
├── logs/                            # one rotating log file per pipeline step
├── artifacts/                       # models, predictions, metrics, tuning results
├── reports/                         # comparison tables, drift reports, profiles
├── notebooks/01_eda.ipynb           # decision-oriented Plotly EDA
├── problem_statement/               # the assignment PDF
├── scripts/                         # convenience shell entry points
├── src/demand_forecasting/          # the package (see section 4)
├── tests/                           # leakage, split and metric unit tests
├── dvc.yaml, params.yaml            # DVC pipeline definition
├── Dockerfile, .dockerignore        # packaged inference service
├── requirements.txt, .env.example
├── README.md                        # how to run everything
├── explanation.md                   # design rationale and assignment mapping
└── architecture_flow.md             # this file
```

---

## 3. End-to-end flow

```text
                       configs/config.yaml
                                │  (every stage loads this first)
                                ▼
  data/raw/data.csv ──► read_raw ──► add_stockout_target ──► build_causal_features
        (1.1M rows)      data.py        stockout.py              features.py
                                                                    │
                                    ┌───────────────────────────────┤
                                    │                               │
                                    ▼                               ▼
                        make_temporal_split                 data/processed/
                            splits.py                       features.parquet
                                    │                    (prepare_dataset.py, DVC out)
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
  tune_bayesian.py            train_dl.py                 train_darts.py
  (Optuna TPE, 25 trials)     (LSTM / Transformer)        (TiDE / TSMixer)
        │  best_params.json         │                           │
        ▼                           │                           │
   train_ml.py                      │                           │
  (LGBM/XGB/CatBoost)               │                           │
        │                           │                           │
        └──────────► artifacts/<model>/{model.*, metrics.json, *_predictions.csv}
                                    │            + MLflow run  + logs/<step>.log
                                    ▼
                          evaluate_compare.py
                    (ranks every model by validation WAPE)
                                    │  reports/model_comparison.csv
                                    ▼
                            select_model.py
                    (writes configs/model_registry.yaml)
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
          inference_router.py                    drift.py
       (routes to ml / dl / darts)        (PSI, KS, promo regime)
                    │                                │
        ┌───────────┴───────────┐            reports/drift_report.json
        ▼                       ▼
   batch CSV forecast     api.py POST /forecast
   (inference_*.py)       (Docker, port 8000)
                                    │
                                    ▼
                            hierarchy.py
                (bottom-up aggregation / middle-out allocation)
```

---

## 4. File-by-file reference

### 4.1 Configuration and cross-cutting concerns

#### `configs/config.yaml`
The single source of truth. Every script takes `--config` and reads its settings from here, so no
behaviour is hard-coded in a training script. Sections:

| Section | Controls |
|---|---|
| `logging` | log directory, level, console mirroring, rotation retention, UTC |
| `project` | run name and `random_seed` (42) used by every model and sampler |
| `data` | raw path, `date_col`, `target_col`, `series_cols`, `horizon: 14`, `lookback: 56` |
| `split` | `validation_days: 14`, `test_days: 14` |
| `features` | lag/rolling windows, stockout strategy, which covariates are known in the future |
| `training` | row caps, `n_jobs`, MLflow experiment name, artifact directory |
| `bayes` | Optuna trial count, timeout, which models to tune |
| `ml` / `dl` | model families and neural hyperparameters |
| `hierarchy` | share window and whether stockout days are excluded from shares |
| `monitoring` | PSI warning/alert thresholds and promo-regime alert threshold |

The three `*_known_future` flags are the **forecast-time contract** and are respected identically in
feature building, training, all three inference paths and the API request validator:

```yaml
promo_known_future: true     # the promo calendar is committed before the forecast origin
price_known_future: false    # realized future prices are NOT assumed known
weather_known_future: false  # a 14-day weather forecast is NOT assumed available
```

Flipping `price_known_future` to `true` simultaneously adds `list_price` / `discount_pct` /
`price_vs_28d_mean` to the tree feature set, to the DL future-covariate tensor, to the Darts future
covariates, and to the fields the API requires in a request. That single-switch consistency is the
point of routing everything through config.

#### `src/demand_forecasting/config.py`
One function, `load_config(path)`, returning the parsed YAML as a plain dict. Deliberately thin so
that config is just data and can be logged verbatim.

#### `src/demand_forecasting/logging_utils.py`
Shared logging setup. Every stage calls `setup_logging(step, cfg)` immediately after loading config.

- Creates `logs/` if missing.
- Attaches a `TimedRotatingFileHandler(when="midnight", backupCount=14)` writing `logs/<step>.log`.
  At midnight the current file becomes `logs/<step>.log.YYYY-MM-DD` and a fresh file starts.
- Mirrors the same records to stdout when `logging.console` is true.
- Is idempotent: calling it twice for the same step does not duplicate handlers.

Helpers:

| Function | Purpose |
|---|---|
| `setup_logging(step, cfg)` | returns the configured `demand_forecasting.<step>` logger |
| `log_run_context(logger, step, cfg, **extra)` | records timestamp, Python/platform, package version, working dir, the full effective config, plus any stage-specific values |
| `log_settings(logger, title, payload)` | writes a labelled JSON block (hyperparameters, split boundaries, saved artifact paths) |
| `log_metrics(logger, title, metrics)` | writes a metric block; MLflow stays the system of record, the log keeps a local copy |

Example of what lands in `logs/train_ml.log`:

```text
2026-09-08 14:46:40 | INFO | demand_forecasting.train_ml | Run context:
{
  "config_path": "configs/config.yaml",
  "model": "lightgbm",
  "package_version": "0.1.0",
  "params_json": "artifacts/tuning/lightgbm_best_params.json",
  "python": "3.13.11",
  "started_at_utc": "2026-09-08T14:46:40.112233+00:00",
  "step": "train_ml"
}
2026-09-08 14:46:40 | INFO | demand_forecasting.train_ml | Effective configuration:
{ ... entire config.yaml ... }
2026-09-08 14:46:41 | INFO | demand_forecasting.train_ml | Model hyperparameters:
{
  "mlflow_experiment": "fmcg-demand-forecasting",
  "mlflow_run_id": "0f3c9d1e5a7b4c2f8d6e1a9b3c5d7e2f",
  "model": "lightgbm",
  "params": {"colsample_bytree": 0.88, "learning_rate": 0.037, "n_estimators": 940, ...},
  "source": "tuned_json"
}
2026-09-08 14:47:02 | INFO | demand_forecasting.train_ml | Fitting lightgbm on 1,041,120 training rows with 61 features (52 numeric, 9 categorical)
2026-09-08 14:52:18 | INFO | demand_forecasting.train_ml | Running recursive validation forecast for 2023-12-04..2023-12-17
2026-09-08 14:56:44 | INFO | demand_forecasting.train_ml | Validation metrics (model selection):
{ "bias": -0.0121, "mae": 9.87, "rmse": 14.02, "smape": 0.1904, "wape": 0.1673 }
```

Because each step has its own file, `logs/tune_bayesian.log` contains one line per Optuna trial,
`logs/train_dl.log` contains the per-epoch loss curve, and `logs/api.log` contains one line per
request — they never interleave.

#### `src/demand_forecasting/tracking.py`
MLflow wiring. `configure_mlflow(experiment)` uses local `./mlruns` unless `MLFLOW_TRACKING_URI` is
set (DagsHub). `flatten_dict()` turns nested config into MLflow's flat `a.b.c` param keys.
`log_json_artifact()` writes a JSON blob and logs it as a run artifact.

---

### 4.2 Data preparation

#### `src/demand_forecasting/data.py`
Two functions that define the canonical load path so training, tuning and inference never disagree:

- `read_raw(path, cfg)` — reads the CSV, parses `date`, sorts by `(store_id, sku_id, date)`.
- `prepare_features(df, cfg)` — `add_stockout_target()` then `build_causal_features()`.

The 33 raw columns are: calendar (`date`, `year`…`is_holiday`), weather (`temperature`, `rain_mm`),
store (`store_id`, `country`, `city`, `channel`, `latitude`, `longitude`), product (`sku_id`,
`sku_name`, `category`, `subcategory`, `brand`), commercial (`units_sold`, `list_price`,
`discount_pct`, `promo_flag`, `gross_sales`, `net_sales`), and supply (`stock_on_hand`,
`stock_out_flag`, `lead_time_days`, `supplier_id`, `purchase_cost`, `margin_pct`).

#### `src/demand_forecasting/stockout.py`
Creates the training target and confidence weights for stockout days, where `units_sold` is
censored sales rather than true demand. `config.features.stockout_target_mode` selects the strategy:

| Mode | `demand_target` on a stockout day | `sample_weight` |
|---|---|---|
| `none` (current default) | observed `units_sold`, unchanged | 1.0 |
| `rolling_median` | `max(observed, prior 56-day non-stockout rolling median)` | 0.5 |
| `percentage` | `observed × (1 + 0.50)` | 0.5 |

Always emits `demand_target`, `sample_weight` and `stockout_imputed_amount` (the audit trail), so
downstream code is identical regardless of mode. Every baseline is `shift(1)`-ed, so today's target
can never estimate itself.

Example under `rolling_median`: a series whose prior non-stockout median is 50 records
`units_sold = 20` with `stock_out_flag = 1`. The row becomes `demand_target = 50`,
`sample_weight = 0.5`, `stockout_imputed_amount = 30`.

#### `src/demand_forecasting/features.py`
Owns all feature logic so training and inference apply identical transformations. Three exports:

- `add_calendar_features(df, date_col)` — year/month/day/weekday/weekofyear/`is_weekend` plus
  cyclical `dow_sin/cos`, `month_sin/cos`, `doy_sin/cos`.
- `build_causal_features(df, cfg, demand_col)` — the point-in-time feature table.
- `ml_feature_columns(df, cfg)` — returns `(numeric, categorical)` after removing every blocked column.

Everything derived from the target is shifted **before** rolling:

```python
out[f"demand_lag_{lag}"] = g[demand_col].shift(lag)                       # lags 1, 7, 14, 28
out["demand_roll_mean_7"] = g[demand_col].transform(
    lambda s: s.shift(1).rolling(7, min_periods=3).mean()                 # windows 7, 14, 28
)
```

Also produced: lagged stockout flags (1/7/14) and a 28-day stockout rate, lagged/rolling
`stock_on_hand`, lagged price and discount plus their 28-day means, `promo_prev_1` and
`promo_rate_28`, day-lagged store-level and SKU-level mean demand, `series_age_days`, and a
`store_sku_id` key.

`LEAKAGE_COLUMNS = ["gross_sales", "net_sales"]` are permanently blocked because they are
deterministic functions of `units_sold × price`. Also blocked from the model input: `demand_target`,
`sample_weight`, `stockout_imputed_amount`, same-day `stock_out_flag` and `stock_on_hand`,
`purchase_cost`, `margin_pct` (kept for business evaluation only), `supplier_id`, `sku_name`,
`latitude`, `longitude` — plus whatever the `*_known_future` flags disable.

#### `src/demand_forecasting/splits.py`
`make_temporal_split(df, cfg)` returns a frozen `TemporalSplit` computed backwards from the last
date. With `validation_days: 14` and `test_days: 14` on data ending 2023-12-31:

```text
train      2021-01-01 .. 2023-12-03
validation 2023-12-04 .. 2023-12-17   (model selection, tuning objective)
test       2023-12-18 .. 2023-12-31   (touched once, for final reporting)
```

`label_split(df, split, cfg)` returns a `"train" / "validation" / "test" / "ignore"` Series.

#### `src/demand_forecasting/prepare_dataset.py` — **step `prepare_dataset`**
CLI that materialises the feature table. This is the DVC `prepare` stage.

```bash
PYTHONPATH=src python -m demand_forecasting.prepare_dataset \
  --config configs/config.yaml --output data/processed/features.parquet
```

Logs the run context, the split boundaries, the row count per split label, and the output shape.
Observed output: 1,100,000 rows × 72 columns; train 1,071,888 / validation 14,056 / test 14,056.

---

### 4.3 Models

#### `src/demand_forecasting/models/ml.py`
`MLBundle` is the serialised unit: model name, fitted sklearn `Pipeline`, and the numeric and
categorical feature lists. `predict()` clips at zero because demand cannot be negative.

`build_ml_bundle()` assembles `ColumnTransformer(median impute → numeric, most-frequent impute +
OrdinalEncoder(unknown = -1) → categorical)` followed by the estimator. Ordinal encoding is used for
all three libraries so the inference contract is identical across model families, and unseen
categories at inference map to `-1` rather than raising.

Objectives: LightGBM `regression_l1`, CatBoost `MAE`, XGBoost `reg:squarederror` — L1 matches the
WAPE/MAE decision metrics on a right-skewed target.

#### `src/demand_forecasting/models/dl_data.py`
The bridge between the dataframe and PyTorch tensors.

- `get_dl_feature_columns(cfg)` — returns `(past_cols, future_cols, static_cols)`, honouring the
  `*_known_future` flags. Past always includes demand, promo, price, weather, stock and calendar;
  future includes calendar plus whatever is genuinely known.
- `fit_metadata(train, cfg)` — computes standardisation statistics and categorical index maps
  **from training rows only**, returned as a `DLMetadata` dataclass that is saved inside the
  checkpoint. Index 0 is reserved for unseen categories.
- `MultiSeriesWindowDataset` — slides a `lookback + horizon` window over each series. Per sample:

```text
past_x      [56, P]   scaled history (demand + dynamic covariates + calendar)
future_x    [14, F]   known-future covariates for each horizon day
static_ids  [6]       store_id, sku_id, channel, category, subcategory, brand
y           [14]      scaled demand_target
weight      [14]      sample_weight (0.5 on imputed stockout days)
```

  `require_target=False` is used at inference, where future targets do not exist.

#### `src/demand_forecasting/models/lstm.py`
From-scratch encoder-decoder LSTM with shared weights across all 1,005 series.

```text
past_x [B,56,P] ─► encoder LSTM ─► (h, c)
static_ids [B,6] ─► embeddings ─► static_vec [B, 6*16]
decoder step h: concat(future_x[:,h,:], static_vec, prev_y) ─► LSTM ─► head ─► ŷ_h
outputs [B,14]
```

The first autoregressive token is the last normalised demand value, `past_x[:, -1, 0:1]`. Scheduled
teacher forcing (probability 0.2) is applied during training only.

#### `src/demand_forecasting/models/transformer.py`
From-scratch temporal Transformer with **direct** 14-step output, avoiding recursive error
accumulation.

```text
past_x  [B,56,P] ─► linear ─► +positional ─► TransformerEncoder ─► memory [B,56,D]
future_x [B,14,F] ─► linear ─┐
static embeddings ─► linear ─┴─► queries [B,14,D] ─► TransformerDecoder(cross-attends memory)
head ─► [B,14]
```

Each horizon day forms its own query from that day's known promo/calendar, so day D+14 is
conditioned on day D+14's promotion, not on a chain of predictions.

---

### 4.4 Training and tuning

#### `src/demand_forecasting/tune_bayesian.py` — **step `tune_bayesian`**
Optuna TPE (`multivariate=True`, seeded) search over LightGBM / XGBoost / CatBoost. The objective is
the **true recursive 14-day validation WAPE**, not a row-wise score on pre-computed lags — a trial is
scored exactly the way the model will be deployed. Training rows are capped (default 400,000) for
search speed; the final fit in `train_ml.py` uses all eligible rows.

```bash
PYTHONPATH=src python -m demand_forecasting.tune_bayesian --model lightgbm
```

Writes `artifacts/tuning/lightgbm_best_params.json` and `artifacts/tuning/lightgbm_trials.csv`.
`logs/tune_bayesian.log` gets the setup block, one line per trial, and the best-trial summary.

#### `src/demand_forecasting/train_ml.py` — **step `train_ml`**
The tree-model training and evaluation stage. Sequence:

1. Build features and drop rows without the longest warm-up lag (`demand_lag_28`).
2. Fit on train (`≤ 2023-12-03`) with `sample_weight`.
3. **Recursively** forecast validation from a fixed origin (`recursive_forecast`).
4. Refit a fresh model on train + validation (`≤ 2023-12-17`), asserting the feature list is unchanged.
5. Recursively forecast the test window once.
6. Log params/metrics/tags/artifacts to MLflow and write `artifacts/<model>/`.

```bash
PYTHONPATH=src python -m demand_forecasting.train_ml \
  --model lightgbm --params-json artifacts/tuning/lightgbm_best_params.json
```

Outputs `artifacts/lightgbm/{model.joblib, validation_predictions.csv, test_predictions.csv,
metrics.json}` plus one MLflow run named `lightgbm-final`.

#### `src/demand_forecasting/train_dl.py` — **step `train_dl`**
Trains the from-scratch LSTM or Transformer. Loss is a weighted L1 on the scaled target, so imputed
stockout days contribute at half weight. Validation drives early stopping (`patience: 5`); the
selected `best_epoch` is then used to refit on train + validation for a **fixed** epoch count, so the
test window never influences stopping. Saves `model.pt` containing the state dict, the `DLMetadata`,
the `dl` config block, lookback/horizon and `best_epoch` — everything inference needs to rebuild the
exact architecture. Per-epoch losses go to both MLflow (`step=epoch`) and `logs/train_dl.log`.

#### `src/demand_forecasting/train_darts.py` — **step `train_darts`**
Trains TiDE or TSMixer as global models over the list of series, with static covariates, past
covariates (`stock_out_flag`, `stock_on_hand`), future covariates from `get_future_cols(cfg)`, and
per-timestep sample weights. Saves `model.pt` plus a `metadata.joblib` holding the series keys,
column lists, static maps and the feature-contract flags — `inference_darts.py` refuses to run if
the current config disagrees with that metadata.

---

### 4.5 Evaluation, comparison and promotion

#### `src/demand_forecasting/metrics.py`
| Function | Meaning |
|---|---|
| `wape` | `Σ|y−ŷ| / Σ|y|` — the primary decision metric; defined even when rows are zero |
| `smape` | symmetric percentage error, safe denominator |
| `forecast_bias` | `Σ(ŷ−y) / Σ|y|`; positive = systematic overforecasting |
| `regression_metrics` | wape, mae, rmse, smape, bias |
| `business_proxy` | underforecast units × unit margin, overforecast units × purchase cost |
| `grouped_metrics` | the same metrics per group, sorted worst-first |
| `full_evaluation` | overall + business proxy + breakdowns by channel, category, promo_flag |

`business_proxy` is a directional cost comparison, not an inventory simulation — lead time, safety
stock and replenishment policy are not modelled.

#### `src/demand_forecasting/evaluate_compare.py` — **step `evaluate_compare`**
Scans `artifacts/*/metrics.json`, builds one row per model, ranks by validation WAPE and writes
`reports/model_comparison.csv`. The ranked table and the selected champion are written to
`logs/evaluate_compare.log`. Test columns are carried for reporting but are never the sort key.

#### `src/demand_forecasting/select_model.py` — **step `select_model`**
Promotes a model to `configs/model_registry.yaml`. It re-reads the comparison file and **refuses to
promote anything that is not the current validation champion**, which prevents accidental promotion
on a good-looking test number.

```bash
PYTHONPATH=src python -m demand_forecasting.select_model \
  --name lightgbm --family ml \
  --artifact artifacts/lightgbm/model.joblib \
  --metrics artifacts/lightgbm/metrics.json
```

Resulting registry:

```yaml
selected_model:
  name: lightgbm
  family: ml
  artifact_path: artifacts/lightgbm/model.joblib
  selected_at: '2026-09-08T14:58:31.204915+00:00'
  selection_metric: validation_wape
  validation_metrics: {wape: 0.1673, mae: 9.87, rmse: 14.02, smape: 0.1904, bias: -0.0121}
  test_metrics: {wape: 0.1711, ...}
  notes: Selected using validation performance only. ...
```

Darts entries additionally need `--metadata-path` and `--model-type`.

---

### 4.6 Inference

All three inference paths enforce the same rule: **no future label, and no covariate the
`*_known_future` flags say is unknown, may enter a feature.**

#### `src/demand_forecasting/inference_ml.py` — **step `inference_ml`**
`recursive_forecast()` is the leakage control for tree models. For each of the 14 horizon days in
order: blank the target and unavailable columns, append the day to history, rebuild causal features,
predict, then append the **prediction** (not the truth) as synthetic history so day D+2's `lag_1`
is day D+1's forecast.

```text
origin = 2023-12-03
D+1 2023-12-04 ← features from real history through 12-03            → ŷ = 47.2
D+2 2023-12-05 ← lag_1 = 47.2 (the prediction, never the actual)     → ŷ = 51.8
...
D+14 2023-12-17
```

#### `src/demand_forecasting/inference_dl.py` — **step `inference_dl`**
Loads the checkpoint, rebuilds the architecture from the stored `dl_config`, and calls
`validate_checkpoint()` to fail loudly if the current config would produce a different feature
schema or a different lookback/horizon. Then builds one `lookback + horizon` window per series and
emits all 14 days directly, de-scaling with the training-time mean/std and clipping at zero.

#### `src/demand_forecasting/inference_darts.py` — **step `inference_darts`**
Validates the config against `metadata.joblib`, rebuilds per-series `TimeSeries` objects with static
covariates, requires a complete contiguous horizon for every trained series, and calls
`model.predict(n=horizon, ...)`.

#### `src/demand_forecasting/inference_router.py`
Reads `configs/model_registry.yaml` and dispatches to the right inference function based on
`family` / `name`. This is the only place that knows how the three families differ, which keeps
`api.py` model-agnostic.

#### `src/demand_forecasting/api.py` — **step `api`**
FastAPI service. `GET /health` returns `{"status": "ok"}`. `POST /forecast` accepts the known-future
covariates and returns one prediction per series-date.

Validation before any model is touched: required fields present (driven by the `*_known_future`
flags), dates parseable, no duplicate `(store, SKU, date)`, exactly `horizon` rows per series, and
the dates exactly contiguous from the day after the history end. Static attributes (channel,
category, brand, cost…) are joined from each series' latest history row, and an unknown store-SKU is
rejected with a 422.

Request:

```json
{
  "future_covariates": [
    {"date": "2024-01-01", "store_id": "STORE0001", "sku_id": "SKU0086",
     "is_holiday": 1, "promo_flag": 1},
    {"date": "2024-01-02", "store_id": "STORE0001", "sku_id": "SKU0086",
     "is_holiday": 0, "promo_flag": 0}
  ]
}
```

Response:

```json
{
  "forecast_horizon": 14,
  "forecast_count": 14,
  "forecasts": [
    {"date": "2024-01-01", "store_id": "STORE0001", "sku_id": "SKU0086", "prediction": 61.4},
    {"date": "2024-01-02", "store_id": "STORE0001", "sku_id": "SKU0086", "prediction": 43.9}
  ]
}
```

Startup context and one line per request go to `logs/api.log`; rejections log at WARNING, unexpected
failures log a full traceback.

---

### 4.7 Hierarchy, monitoring and analysis

#### `src/demand_forecasting/hierarchy.py`
Kept deliberately separate from the forecast models.

- `aggregate_bottom_up(forecasts, levels, cfg)` — sums store-SKU forecasts to any parent (store,
  category, subcategory, channel, country). Coherent by construction.
- `trailing_shares(history, child_cols, parent_cols, cfg)` — allocation shares from the trailing
  56-day window, excluding stockout days by default, falling back to equal shares when a parent has
  no recent demand. Asserts shares sum to 1 within each parent.
- `middle_out_allocate(parent_forecast, shares, ...)` — splits a parent-level forecast down to
  children and verifies the children sum back to the parent.

#### `src/demand_forecasting/drift.py` — **step `drift`**
Compares a reference window against a current window: numeric PSI + KS + mean shift + missingness
(`units_sold`, `list_price`, `discount_pct`, `temperature`, `rain_mm`, `stock_on_hand`,
`lead_time_days`), categorical PSI (`channel`, `category`, `subcategory`, `brand`, `promo_flag`,
`stock_out_flag`), and an explicit promo-regime check. Thresholds come from `config.monitoring`
(PSI ≥ 0.10 warning, ≥ 0.25 alert; promo rate relative change ≥ 0.30 alert). Writes
`reports/drift_report.json` and logs a `N warnings, M alerts` summary line.

#### `src/demand_forecasting/strategy_analysis.py` — **step `strategy_analysis`**
Quantifies the operational cost of each modelling strategy — how many models or series
global / per-subcategory / per-SKU / local / middle-out would create, and the row counts behind
each. Writes `reports/strategy_analysis.csv`.

#### `notebooks/01_eda.ipynb`
Twelve decision-oriented sections: schema and completeness, hierarchy integrity, target distribution
and heterogeneity, aggregate trend and weekly/monthly seasonality, interactive series examples,
promotion lift and discount depth, the stockout censored-demand diagnostic, price and weather,
sparse/zero/cold-start diagnostics, the leakage-safe split, the global-vs-segmented-vs-local
strategy summary, and what to run next. Each section states what the signal implies for the model,
not just what the chart shows.

#### `tests/`
`conftest.py` holds a minimal config fixture. `test_leakage.py` proves that mutating `y_t` does not
change the features for row `t`, and that changing future targets does not change past features.
`test_splits.py` pins the exact boundary dates and labels. `test_metrics.py` checks WAPE, bias sign
and finiteness. 15 tests, all passing.

---

## 5. Logging model

| Property | Behaviour |
|---|---|
| Location | `logs/<step>.log` — one file per pipeline step |
| Rotation | at midnight; yesterday becomes `logs/<step>.log.2026-09-07` |
| Retention | `logging.backup_count` files (default 14 days) |
| Level | `logging.level` (default `INFO`) |
| Console | mirrored to stdout when `logging.console: true` |
| Format | `timestamp \| LEVEL \| demand_forecasting.<step> \| message` |

Steps that produce a log file: `prepare_dataset`, `tune_bayesian`, `train_ml`, `train_dl`,
`train_darts`, `evaluate_compare`, `select_model`, `drift`, `inference_ml`, `inference_dl`,
`inference_darts`, `strategy_analysis`, `api`.

Every one of them records, at minimum: run context (timestamp, Python/platform, package version,
CWD, CLI arguments), the **entire effective configuration**, the settings actually used
(hyperparameters and their source, split boundaries, device, feature counts), progress milestones,
resulting metrics, and the paths of every artifact written.

### Division of responsibility with MLflow

| Concern | Where it lives |
|---|---|
| Experiment comparison, params, metrics, model artifacts | MLflow (`./mlruns` or DagsHub) |
| Chronological narrative of a run, progress, failures, stack traces | `logs/<step>.log` |
| Data and pipeline versioning | DVC |
| The one model currently serving | `configs/model_registry.yaml` |

The log is what you read when a run fails or behaves oddly; MLflow is what you read when comparing
runs. The overlap (metrics appear in both) is intentional — the log stays readable without a server.

---

## 6. Leakage controls, collected

1. `gross_sales` / `net_sales` permanently blocked — deterministic functions of the target.
2. Every rolling target statistic is `shift(1)`-ed before `rolling()`.
3. Same-day `stock_out_flag` and `stock_on_hand` are never features; only lagged versions are.
4. The split is chronological, never random.
5. Tree validation and test use recursive forecasting from a fixed origin — a horizon day's lag
   features come from earlier *predictions*, never from actual future demand.
6. DL models see only the lookback window plus genuinely-known future covariates.
7. Scaling statistics and categorical maps are fitted on training rows only.
8. Bayesian tuning optimises validation WAPE; the test window is evaluated once.
9. DL early stopping uses validation; the refit then runs a fixed epoch count.
10. Inference and the API blank every column the `*_known_future` contract marks unavailable.
11. `test_leakage.py` asserts (1)-(2) mechanically on every test run.

---

## 7. Reproducing a run

```bash
export PYTHONPATH=$PWD/src
dvc repro                                                    # rebuild features from tracked data
python -m demand_forecasting.tune_bayesian --model lightgbm  # → artifacts/tuning/*.json
python -m demand_forecasting.train_ml --model lightgbm \
  --params-json artifacts/tuning/lightgbm_best_params.json   # → artifacts/lightgbm/*, MLflow run
python -m demand_forecasting.evaluate_compare                # → reports/model_comparison.csv
python -m demand_forecasting.select_model --name lightgbm --family ml \
  --artifact artifacts/lightgbm/model.joblib \
  --metrics artifacts/lightgbm/metrics.json                  # → configs/model_registry.yaml
```

The four things needed to reproduce any result are the DVC data version, the Git commit, the
`Effective configuration` block in the step's log, and the MLflow run id printed in that same log.
