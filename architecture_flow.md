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
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
        ▼                                                       ▼
  tune_bayesian.py                                        tune_dl.py
  (Optuna TPE, trees, n_trials)                (Optuna TPE, LSTM/Transformer/
        │                                       TiDE/TSMixer; searches lookback)
        │  artifacts/tuning/<model>_best_params.json + <model>_trials.csv
        │  one nested MLflow run per trial
        └───────────────────────────┬───────────────────────────┘
                                    │  --params-json
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
   train_ml.py                 train_dl.py               train_darts.py
  (LGBM/XGB/CatBoost)        (LSTM / Transformer)        (TiDE / TSMixer)
        │                           │                           │
        └──────────► artifacts/<model>/{model.*, metrics.json, *_predictions.csv}
                                    │            + MLflow run  + logs/<step>.log
                                    ▼
                          evaluate_compare.py
              (ranks every model by validation selection_metric)
                                    │  reports/v1/model_comparison.csv
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
          report_best_models.py                select_model.py
   (best per family + test vs actuals,   (writes configs/model_registry.yaml)
    per-series / per-horizon metrics)             │
                                                  │
                    ┌─────────────────────────────┴──┐
                    ▼                                ▼
          inference_router.py                    drift.py
       (routes to ml / dl / darts)        (PSI, KS, promo regime)
                    │                                │
        ┌───────────┴───────────┐            reports/drift_report.json
        ▼                       ▼
   batch CSV forecast     api.py POST /forecast
   (inference_*.py)       (Docker, port 8000)
        ▲                       ▲
        └───────────┬───────────┘
                    │  future_covariates.csv / forecast_request.json
          make_future_template.py
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
| `bayes` | trial counts (`n_trials` trees, `n_trials_dl` sequence), timeout, `objective_metric`, `log_trial_models` |
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
Every trial is also a nested MLflow run carrying its params and the full validation metric set;
`bayes.log_trial_models` additionally persists each trial's fitted model.

#### `src/demand_forecasting/tune_dl.py` — **step `tune_dl`**
The same Optuna TPE treatment for the sequence models — `lstm`, `transformer`, `tide`, `tsmixer` —
scored on a real 14-day validation forecast and logged as nested MLflow runs. Crucially it searches
**`lookback` (the input chunk length)**, which was previously a fixed constant. `apply_dl_params()`
routes `lookback` into `data` and everything else into `dl`, on a deep copy so trials stay isolated.
`train_dl.py` and `train_darts.py` consume the result via `--params-json`. See Q10 and Q11.

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

It now mirrors `train_ml.py` / `train_dl.py` end to end: `evaluate_window()` forecasts validation
from the train-end origin, the model is refit on train+validation, the test window is scored once
from the val-end origin, and the run writes `metrics.json`, both prediction CSVs and a full MLflow
run. TiDE/TSMixer therefore appear in `reports/v1/model_comparison.csv` and are promotable.

`forecast_from_origin()` is the key helper: target and past covariates stop at the origin, while
known-future covariates (calendar plus whatever the `*_known_future` flags permit) legitimately
extend across the horizon.

---

### 4.5 Evaluation, comparison and promotion

#### `src/demand_forecasting/metrics.py`
| Function | Meaning |
|---|---|
| `wape` | `Σ|y−ŷ| / Σ|y|` — the primary decision metric; defined even when rows are zero |
| `mape` | mean of `|y−ŷ|/|y|` over **non-zero actuals only**; unstable on low-demand rows |
| `mape_coverage` | fraction of rows MAPE was actually computed on |
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
`reports/v1/model_comparison.csv`. The ranked table and the selected champion are written to
`logs/evaluate_compare.log`. Test columns are carried for reporting but are never the sort key.

#### `src/demand_forecasting/report_best_models.py` — **step `report_best_models`**
Picks the lowest `val_<metric>` model **within each family** (ml / dl / darts), then loads that
model's `test_predictions.csv` and scores it against actuals. Produces the per-series and
per-horizon breakdowns that `full_evaluation` does not compute, plus a pooled-vs-macro WAPE summary.
See Q15 for the full output list.

#### `src/demand_forecasting/make_future_template.py` — **step `make_future_template`**
Generates the known-future covariate rows for the next horizon: finds series active on the final
history date, builds the 14 contiguous dates that follow, fills deterministic calendar fields, and
writes both `future_covariates.csv` (batch CLIs) and `forecast_request.json` (the API body). The
required fields are derived from the `*_known_future` contract, so the request can never drift from
the model. See Q16.

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
and finiteness. `test_hierarchy.py` covers bottom-up coherence, share normalisation and middle-out
allocation across multi-day horizons. 25 tests, all passing.

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

Steps that produce a log file: `prepare_dataset`, `tune_bayesian`, `tune_dl`, `train_ml`,
`train_dl`, `train_darts`, `evaluate_compare`, `report_best_models`, `select_model`, `drift`,
`inference_ml`, `inference_dl`, `inference_darts`, `make_future_template`, `strategy_analysis`,
`api`.

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

### Verified empirically

The unit tests only cover the feature builder. The end-to-end forecast paths were additionally
probed by corrupting information that is unknown at the forecast origin and checking the forecast
does not move (a bit-identical result means the information cannot have been used):

| Probe | Perturbation | Result |
|---|---|---|
| Tree: future labels | `units_sold × 1000 + 12345`, `gross/net_sales × 999` | max Δ = 0.0 |
| Tree: future inventory | flip `stock_out_flag`, `stock_on_hand → 0` | max Δ = 0.0 |
| Tree: unknown covariates | `list_price × 5`, `discount 0.9`, `temp −50`, `rain 999` | max Δ = 0.0 |
| Tree: promo (declared **known**) | flip `promo_flag` | max Δ = 151.3 — correctly *does* move |
| Tree: recursion | inspect D+2 `demand_lag_1` | equals D+1 **prediction**, not D+1 actual |
| Tree: blocked columns | inspect the 52 model inputs | none of the blocked names present |
| DL: future labels + inventory | same corruption | max Δ = 0.0 |
| DL: promo (declared **known**) | flip `promo_flag` | max Δ = 115.8 — correctly *does* move |
| Stockout `rolling_median` | build targets from train-only vs full data | 0 / 21,340 training targets change |

The promo probes are the control: they prove the zero-deltas above are genuine leakage blocks and
not an artefact of the models ignoring their future inputs entirely.

---

## 7. Reproducing a run

```bash
export PYTHONPATH=$PWD/src
dvc repro                                                    # rebuild features from tracked data
python -m demand_forecasting.tune_bayesian --model lightgbm  # → artifacts/tuning/*.json
python -m demand_forecasting.train_ml --model lightgbm \
  --params-json artifacts/tuning/lightgbm_best_params.json   # → artifacts/lightgbm/*, MLflow run
python -m demand_forecasting.evaluate_compare                # → reports/v1/model_comparison.csv
python -m demand_forecasting.select_model --name lightgbm --family ml \
  --artifact artifacts/lightgbm/model.joblib \
  --metrics artifacts/lightgbm/metrics.json                  # → configs/model_registry.yaml
```

The four things needed to reproduce any result are the DVC data version, the Git commit, the
`Effective configuration` block in the step's log, and the MLflow run id printed in that same log.

---

## 8. Q&A

A running record of design questions asked about this project and the answers, with the evidence
behind them. Newest entries are appended at the end.

---

### Q1. Should we use multiple k-fold-style validation blocks instead of a single one?

**Short answer:** not k-fold — that is invalid for this problem. But yes, the single validation
window should become **multiple rolling origins**. One window is defensible for a v1 and is what is
implemented today, but it is the weakest part of the current model-selection story.

#### Why literal k-fold is wrong here

Standard k-fold shuffles rows into random folds. For a forecasting problem that is not merely
suboptimal, it is invalid:

- A fold boundary that cuts through time puts **future rows in training and past rows in
  validation**. The model learns from 2023-12-10 to predict 2023-06-01.
- Our features are lags and rolling windows over a series. Neighbouring rows share almost all their
  information, so a random split leaves near-duplicates of every validation row in training. The
  score becomes an interpolation score, not a forecasting score.
- It does not measure the deployed task. Production runs one 14-day-ahead batch forecast from a
  fixed origin; a random fold measures "fill in a missing day given both its neighbours", which is
  a much easier and irrelevant problem.

The time-series analogue of k-fold is **rolling-origin (walk-forward) backtesting**: several
sequential origins, where training always ends strictly before each origin.

```text
origin 1   train ──────────────┤ val(14d)
origin 2   train ──────────────────────┤ val(14d)
origin 3   train ──────────────────────────────┤ val(14d)
                                                     ... K origins
                                                              held-out test (untouched)
```

#### What we do today

Exactly **one** origin: train ends 2023-12-03, validation is 2023-12-04 → 2023-12-17, and that
single window is both the Optuna objective (`tune_bayesian.py`) and the ranking key in
`evaluate_compare.py`.

#### Why one window is defensible

- It matches the deployment geometry exactly — one 14-day batch forecast from a fixed origin.
- The sample is not small: 1,004 active series × 14 days = **14,056 rows**, so *row-level* sampling
  noise on pooled WAPE is minor.
- It is cheap. `recursive_forecast` is the dominant cost in tuning (it rebuilds causal features once
  per horizon day), so each extra origin multiplies tuning time roughly linearly.

#### Why one window is nevertheless not enough

The uncertainty that matters is not row-level noise — it is **origin-to-origin variance**, and a
single window cannot measure it at all.

1. **Regime concentration.** Both the validation and test windows fall in **late December**. The EDA
   found demand is elevated Jun–Aug and Oct–Dec, so the entire selection *and* reporting exercise
   sits inside one high-demand, holiday, promo-heavy fortnight. A model chosen there is not
   demonstrably the best model for March.
2. **The model differences are small.** In the audit run the top five models spanned validation WAPE
   0.256 → 0.274. Gaps that narrow are well within the swing a different origin can produce, so
   ranking on one window risks promoting the winner of a coin flip.
3. **No stability signal.** One number gives a point estimate with no spread. With K origins you get
   a mean *and* a standard deviation, and "consistently second-best" often beats "won once by a
   nose" for something you have to operate.

#### Recommendation

Keep the final test window fixed and untouched. Replace the single validation origin with K rolling
origins (6–12, spaced biweekly or monthly so several seasons are represented), and select on the
**mean validation WAPE across origins**, reporting the spread alongside it.

Because the cost is linear in K, a practical compromise is a two-stage search: tune on 2–3 origins
(or a row subsample) to narrow the hyperparameter space, then evaluate only the surviving candidates
on the full set of origins.

Note that no purge/embargo gap is needed between train and origin: every feature is strictly
backward-looking (`shift(1)` before every rolling window) and `make_temporal_split` already ends
training the day before the origin, so there is no overlap to purge.

**Status: known gap.** `explanation.md` §3 already flags rolling-origin backtesting as the
recommended maturity step. It is not implemented — `splits.py` produces a single `TemporalSplit`.
Adding it means a generator of origins plus a loop in `tune_bayesian.py` and `train_ml.py`.

---

### Q2. How are the metrics computed across so many store-SKU combinations — per series, or averaged?

**Short answer:** everything is computed **once over all rows pooled together** (micro-averaging).
Metrics are *not* computed per store-SKU and then averaged, and there is currently no per-series
breakdown at all. Also note **MAPE is not implemented** — deliberately.

#### What the code actually does

`full_evaluation(df, y_col, pred_col)` in `metrics.py` receives one flat dataframe — every
`(store_id, sku_id, date)` row in the window, i.e. **14,056 rows** for validation (1,004 active
series × 14 horizon days) — and calls `regression_metrics()` on the whole thing at once:

```python
overall = regression_metrics(df[y_col], df[pred_col])   # one pooled number per metric
```

So `metrics.json → validation.overall.wape` is a single global figure covering all series and all
14 horizon days simultaneously.

| Metric | Formula | How it aggregates |
|---|---|---|
| `wape` | `Σ|y−ŷ| / Σ|y|` over all rows | **volume-weighted** — high-volume series dominate |
| `mae` | mean of `|y−ŷ|` over all rows | equal weight per row, but large-volume rows produce large absolute errors |
| `rmse` | `sqrt(mean((y−ŷ)²))` over all rows | equal weight per row, extra penalty on big misses |
| `smape` | mean of the per-row symmetric ratio | **equal weight per row** — behaves differently from WAPE |
| `bias` | `Σ(ŷ−y) / Σ|y|` over all rows | volume-weighted |

Note that WAPE and SMAPE aggregate differently: WAPE is a ratio of sums (volume-weighted), while
SMAPE is a mean of ratios (row-weighted). They can disagree, and that disagreement is informative.

#### The consequence, made concrete

Pooled WAPE is volume-weighted, which means it can hide systematic failure on small series. Verified
numerically with two rows:

| Series | actual | predicted | error | per-series WAPE |
|---|---|---|---|---|
| HIGH | 500 | 450 | 50 | 10% |
| LOW | 10 | 5 | 5 | **50%** |

- **Pooled (what we report):** `55 / 510` = **10.8%**
- **Per-series, then averaged (macro):** `(10% + 50%) / 2` = **30.0%**

Same predictions, a ~3× difference in the headline number. The pooled figure is the right one for a
business question ("how many units are we off across the estate?"), because a unit of error on a
500/day SKU really does cost more than a unit on a 10/day SKU. But it is the wrong one for asking
"does this model work for *every* SKU?", and low-volume series are exactly where cold-start and
sparse-demand problems live.

#### What breakdowns exist

`full_evaluation` builds exactly three, hardcoded:

```python
for group_col in ["channel", "category", "promo_flag"]:
```

Within each group the same pooling applies. `grouped_metrics()` sorts worst-WAPE-first so problem
segments surface at the top.

#### Why there is no MAPE

MAPE divides by `y` on every row. This dataset contains low- and zero-demand days, so MAPE would be
undefined or explode toward infinity on exactly the rows we care most about, and a single near-zero
actual can dominate the whole average. WAPE is the standard robust replacement: same "percentage
error" intuition, one shared denominator, always defined as long as total demand is non-zero. SMAPE
is reported alongside as a bounded row-level view.

#### Gaps worth closing

1. **No per-series metrics.** Nothing reports WAPE per `(store_id, sku_id)`, so a systematically
   broken series is invisible in the headline number.
2. **No per-horizon-day metrics.** Error should grow from D+1 to D+14 (especially for the recursive
   tree path, where predictions feed later lags). We never measure that decay, so we cannot tell a
   model that is strong at D+1 and collapsing by D+14 from one that is uniformly mediocre.
3. **No macro WAPE.** Reporting pooled *and* per-series-averaged WAPE side by side would immediately
   expose the small-series problem shown above.

`explanation.md` §8 lists horizon day, store, subcategory and volume decile as breakdowns that
*should* be inspected; none of them are implemented. The per-row prediction CSVs
(`artifacts/<model>/validation_predictions.csv`, `test_predictions.csv`) are saved, so all of these
can be computed after the fact without retraining — the data is there, only the reporting is
missing.

> **Update.** MAPE has since been implemented (`metrics.mape`, with `mape_coverage` reporting the
> fraction of rows scored) and is logged for train/validation/test alongside the other metrics.
> Selection is configurable via `training.selection_metric` / `--metric`. See Q3 for why WAPE
> remains the recommended default.

---

### Q3. Which metric should actually drive model selection — and does it matter?

**It matters: the metric changes which model gets promoted.** Measured on a 12-series subset with
the full pipeline:

| model | val_wape | val_mape | test_wape | test_mape |
|---|---|---|---|---|
| lstm | 0.2552 | **0.3151** | 0.3024 | 0.8697 |
| lightgbm | **0.2511** | 0.3336 | 0.2821 | 0.9060 |
| xgboost | 0.2737 | 0.3513 | 0.3121 | 0.8865 |
| tide | 0.3742 | 0.4576 | 0.3781 | 1.0791 |

Ranking on WAPE promotes **lightgbm**; ranking on MAPE promotes **lstm**. Same runs, same
predictions, different champion.

Two things to notice in that table:

1. **MAPE is far less stable across windows.** It nearly triples from validation (~0.32) to test
   (~0.87–1.08), while WAPE moves modestly (~0.25 → ~0.28). A metric that swings that much between
   two adjacent 14-day windows is a poor basis for a promotion decision.
2. **The instability is structural, not noise.** MAPE divides by each actual. On this dataset 0.28%
   of rows have zero demand (excluded from MAPE entirely — `mape_coverage` reports how many) and
   ~1.5% have demand below 5 units, where a 2-unit miss reads as a 40–200% error. A handful of
   small-demand rows can dominate the average.

WAPE (`Σ|y−ŷ| / Σ|y|`) shares the "percentage error" intuition but uses one shared denominator, so
it is always defined and volume-weighted — which also matches the business question, since a unit of
error on a 500/day SKU genuinely costs more than one on a 10/day SKU.

**Recommendation:** keep `selection_metric: wape` and read MAPE alongside it. Both are always
computed and logged; the config only picks the ranking key. If a stakeholder requires MAPE, run
`--metric mape` consistently across `tune_bayesian`, `evaluate_compare` and `select_model` — tuning
on one metric and selecting on another is the one combination to avoid.

---

### Q4. How is `config.yaml` linked into the other code files?

There is no import-time magic and no global singleton. Config is **plain data, passed explicitly**.

Every entry point takes `--config` (default `configs/config.yaml`), calls `load_config()` once, and
threads the resulting dict down as an argument:

```python
# config.py - the entire mechanism
def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

```python
# every CLI, same shape
cfg = load_config(args.config)
logger = setup_logging("train_ml", cfg)          # logging section
raw = read_raw(cfg["data"]["raw_path"], cfg)     # data section
split = make_temporal_split(raw, cfg)            # split section
feat = prepare_features(raw, cfg)                # features section
```

Library functions never read the file themselves — they receive `cfg` and index the section they
own (`build_causal_features` reads `cfg["features"]`, `make_temporal_split` reads `cfg["split"]`,
and so on). Three consequences that matter:

1. **Testability.** `tests/conftest.py` supplies a plain dict fixture; no file or monkeypatching.
2. **Alternate configs are free.** Passing `--config _v3/config.yaml` redirects data paths, artifact
   directory, MLflow experiment and log directory in one move — that is exactly how the smoke
   tests in this document were run without touching the real project directories.
3. **Auditability.** Because the config is just a dict, `log_run_context()` dumps the *entire
   effective configuration* into every step's log, so a run can be reproduced from its log alone.

The one override path is `apply_dl_params(cfg, params)` (`config.py`), which returns a **deep copy**
with tuned sequence-model values applied — `lookback` into `data`, everything else into `dl`. The
copy keeps each Optuna trial isolated from the next.

---

### Q5. How is the feature / training dataset built, and how is the stockout handling configured?

The chain is always the same three calls, so training and inference cannot diverge:

```python
raw  = read_raw(path, cfg)             # parse dates, sort by (store, sku, date)
df   = add_stockout_target(raw, cfg)   # create demand_target + sample_weight
feat = build_causal_features(df, cfg, demand_col="demand_target")
```

`prepare_features()` in `data.py` is just those last two steps bundled, and
`prepare_dataset.py` is the CLI that writes the result to `data/processed/features.parquet` (the
DVC `prepare` stage). Observed on the full data: **1,100,000 rows × 72 columns**, labelled train
1,071,888 / validation 14,056 / test 14,056.

#### The stockout problem

On a stockout day `units_sold` is **censored** — it records what was available to sell, not what
customers wanted. Training directly on it teaches the model that demand was low exactly when it may
have been high. `config.features.stockout_target_mode` picks the response:

| Mode | `demand_target` on a stockout day | `sample_weight` |
|---|---|---|
| `none` **(current default)** | observed `units_sold`, untouched | 1.0 |
| `rolling_median` | `max(observed, prior 56-day non-stockout rolling median)` | 0.5 |
| `percentage` | `observed × (1 + stockout_percentage_uplift)` | 0.5 |

Governing keys: `stockout_imputation_window` (56), `stockout_imputation_min_periods` (7),
`stockout_percentage_uplift` (0.50), `stockout_sample_weight` (0.50).

Worked example under `rolling_median` — a series whose prior non-stockout median is 50 records
`units_sold = 20` with `stock_out_flag = 1`:

```text
demand_target           = max(20, 50) = 50
sample_weight           = 0.5          # trained on, but at half confidence
stockout_imputed_amount = 30           # audit trail: how much was invented
```

Three deliberate design points:

- **The columns are always produced.** Even under `none`, `demand_target`, `sample_weight` and
  `stockout_imputed_amount` exist, so no downstream code branches on the mode. Switching strategy
  changes numbers, never code paths.
- **The weight is honoured everywhere.** Trees pass `model__sample_weight`, the torch models use a
  weighted L1 loss, and Darts receives per-timestep `sample_weight` series.
- **Imputation is causal.** Every baseline is `shift(1)`-ed before rolling, so a row's target can
  never be estimated from itself. Verified: building targets from train-only versus full data
  changes **0 of 21,340** training rows (Q3 probe table).

`none` is the current default because the imputation is a modelling assumption, and the honest v1
is to measure it as an experiment rather than bake it in.

---

### Q6. How do we make sure leakage is not happening?

Four layers: design, mechanism, tests, and empirical probes.

**1. Design** — the split is chronological, never random (Q1), and the feature contract is explicit
in config: `promo_known_future: true`, `price_known_future: false`, `weather_known_future: false`.

**2. Mechanism** — see §6 for the full list of eleven controls. The two loadbearing ones:

- Every target-derived statistic is shifted before it is rolled:
  `s.shift(1).rolling(7).mean()`, never `s.rolling(7).mean()`.
- Validation and test are produced by **recursive forecasting from a fixed origin**
  (`recursive_forecast`), so day D+2's `lag_1` is day D+1's *prediction*, never the actual.

**3. Unit tests** — `tests/test_leakage.py` mutates `y_t` and asserts row `t`'s features are
unchanged, and mutates future targets and asserts past features are unchanged.

**4. Empirical probes** — the unit tests only cover the feature builder, so the end-to-end forecast
paths were attacked directly by corrupting information unknown at the origin. Full results in
§6 "Verified empirically"; the summary is that corrupting future labels, future inventory, and the
covariates declared unknown all produce **bit-identical forecasts** (max Δ = 0.0) for both the tree
and DL paths, while flipping `promo_flag` — which *is* declared known — moves predictions by 151.3
and 115.8 units respectively.

That last row is the control, and it is the reason the zeros are meaningful: it proves the models
are genuinely reading their future inputs rather than ignoring them.

---

### Q7. How are categorical features handled for each model?

Three different mechanisms, because the model families need different things.

**Tree models** (`models/ml.py`) — ordinal encoding inside the sklearn pipeline:

```python
("cat", Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
]), categorical)
```

Categoricals: `store_id`, `sku_id`, `store_sku_id`, `country`, `city`, `channel`, `category`,
`subcategory`, `brand`. Unseen categories at inference map to `-1` rather than raising, which is
what makes cold-start requests survivable. Ordinal (not one-hot) because trees split on thresholds
and 1,005 `store_sku_id` values would explode into 1,005 one-hot columns for no gain.

Note this deliberately does **not** use CatBoost's native categorical handling. That costs some
CatBoost accuracy (ordered target statistics are genuinely good) but buys one identical inference
contract across all three libraries. A native-CatBoost branch is a reasonable experiment.

**Torch models** (`models/dl_data.py`) — learned embeddings:

```python
static_cols   = ["store_id", "sku_id", "channel", "category", "subcategory", "brand"]
static_maps   = {col: {value: idx + 1 for idx, value in enumerate(sorted(train[col].unique()))}}
cardinalities = [len(static_maps[col]) + 1 for col in static_cols]   # +1 reserves 0
```

Index **0 is reserved for unseen**, so an unknown SKU degrades to a learned "unknown" vector rather
than crashing. Maps are fitted on **training rows only** and saved inside the checkpoint, so
inference reproduces the exact encoding. Each ID becomes an `embedding_dim`-wide vector (default 16,
tunable), concatenated into every decoder step (LSTM) or into the future queries (Transformer).

**Darts models** (`train_darts.py`) — integer static covariates via `make_static_maps()`, attached
with `with_static_covariates()` and consumed because both models are built with
`use_static_covariates=True`. Unknown values map to `-1`.

---

### Q8. What additional features were engineered — and are the sine/cosine terms future covariates?

**Yes — the cyclical terms are future covariates, and that is the point of them.**

`add_calendar_features()` derives everything deterministically from the date alone, which means they
are computable for any future date with no data at all:

```python
out["dow_sin"]   = np.sin(2 * np.pi * d.dt.dayofweek / 7)
out["dow_cos"]   = np.cos(2 * np.pi * d.dt.dayofweek / 7)
out["month_sin"] = np.sin(2 * np.pi * (d.dt.month - 1) / 12)
out["month_cos"] = np.cos(2 * np.pi * (d.dt.month - 1) / 12)
out["doy_sin"]   = np.sin(2 * np.pi * (d.dt.dayofyear - 1) / 365.25)
out["doy_cos"]   = np.cos(2 * np.pi * (d.dt.dayofyear - 1) / 365.25)
```

Why encode cyclically at all: raw `weekday` implies Sunday (6) is six units from Monday (0), when
they are adjacent. The sin/cos pair places each weekday on a circle so the distance between Sunday
and Monday is the same as between Monday and Tuesday. Same argument for month and day-of-year across
the December→January boundary. This matters here because the EDA found strong weekly seasonality
(autocorrelation peaks at lags 7/14/21/28) and an annual pattern.

The full engineered set beyond the 33 raw columns:

| Group | Features | Available in future? |
|---|---|---|
| Calendar | year, month, day, weekday, weekofyear, `is_weekend`, `is_holiday` | **Yes** |
| Cyclical | `dow_sin/cos`, `month_sin/cos`, `doy_sin/cos` | **Yes** |
| Demand history | `demand_lag_{1,7,14,28}` | No — history only |
| Demand rolling | shifted mean/std/max over 7/14/28 | No |
| Promotions | `promo_flag` (known), `promo_prev_1`, `promo_rate_28` | flag yes, history no |
| Price | `list_price_lag_1`, `discount_pct_lag_1`, `list_price_mean_28`, `discount_pct_mean_28` | No under current contract |
| Inventory | `stockout_lag_{1,7,14}`, `stockout_rate_28`, `stock_on_hand_lag_1`, `stock_on_hand_mean_7` | No |
| Cross-series | `store_demand_mean_lag_1`, `sku_demand_mean_lag_1` | No |
| Identity | `store_sku_id`, `series_age_days` | Yes (static) |

`price_vs_28d_mean` and `discount_change_1` are created **only** when `price_known_future: true`,
since they mix a future price with historical context.

---

### Q9. What exactly are the past and future covariates?

`models/dl_data.py::get_dl_feature_columns(cfg)` is the single source of truth, and it is
config-driven so the three model families cannot disagree.

**Past covariates** — the 56-day lookback window, `past_x [B, 56, P]`. These may contain anything
observed up to the origin, including the target itself:

```python
BASE_PAST_NUMERIC = [
    "demand_target",                                   # the target's own history
    "promo_flag", "list_price", "discount_pct",        # realized commercial context
    "temperature", "rain_mm",                          # realized weather
    "stock_out_flag", "stock_on_hand",                 # realized inventory
    "is_holiday", "is_weekend",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
]
```

**Future covariates** — the 14-day horizon, `future_x [B, 14, F]`. Only things genuinely known at
the origin. Calendar is unconditional; the rest is gated by the contract flags:

```python
BASE_FUTURE_NUMERIC = ["is_holiday", "is_weekend", "dow_sin", "dow_cos", "month_sin", "month_cos"]

if promo_known_future:   future_cols += ["promo_flag"]              # currently ON
if price_known_future:   future_cols += ["list_price", "discount_pct"]   # currently OFF
if weather_known_future: future_cols += ["temperature", "rain_mm"]       # currently OFF
```

**Static covariates** — `static_ids [B, 6]`: `store_id`, `sku_id`, `channel`, `category`,
`subcategory`, `brand`.

So under the current config: **P = 14 past channels, F = 6 future channels, S = 6 static IDs.**
Turning on `price_known_future` would make F = 8 and simultaneously add those fields to the tree
feature set, the Darts future covariates, and the API's required request fields.

Darts uses the same split via `train_darts.get_future_cols(cfg)`, with
`PAST_COLS = ["stock_out_flag", "stock_on_hand"]` as its past-observed covariates. The critical
asymmetry: in `forecast_from_origin()` the target and past covariates stop at the origin, while
future covariates legitimately extend across the horizon — because they carry no target information.

---

### Q10. What input chunk lengths do we search? Is lookback tuned?

**It is now — this was a real gap and has been closed.** `lookback` (the input chunk length) was a
fixed constant at 56 and never searched.

`tune_dl.py` searches it for every sequence model:

```python
params = {
    "lookback": trial.suggest_int("lookback", max(14, horizon), 112, step=14),   # 14..112
    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    ...
}
```

The lower bound is one full horizon (a model cannot see less history than it must forecast) and the
step of 14 keeps candidates on whole-week boundaries, which matters given the weekly seasonality.

Per-model additions: LSTM gets `hidden_size`, `num_layers`, `embedding_dim`, `weight_decay`;
Transformer gets `transformer_d_model`, `transformer_heads`, `transformer_layers`, `embedding_dim`,
`weight_decay`; TiDE/TSMixer get `hidden_size`.

**It demonstrably matters.** In a smoke run the search moved away from the fixed default in both
directions — LSTM chose **lookback 42**, TiDE chose **lookback 84**, against the hardcoded 56.

Two supporting fixes were needed to make this real:

- `train_darts.build_model()` hardcoded `hidden_size=128`, so tuning it would have been silently
  ignored. It now reads from `cfg["dl"]`.
- `train_dl.py` and `train_darts.py` gained `--params-json`, so tuned values actually reach
  training via `apply_dl_params()`.

**For tree models there is no "input chunk length"** — the equivalent is how far back the lag and
rolling features reach (`features.demand_lags` = 1/7/14/28 and `demand_roll_windows` = 7/14/28).
These are config-driven but **not** currently searched, because changing them means rebuilding the
feature table inside every trial. That is the honest remaining gap here; the hook is
`build_causal_features(df, cfg)` reading those lists from config.

---

### Q11. Are we running Bayesian optimisation for all the models?

**Now yes — for all seven.** Previously only the three tree models were tuned.

| Model | Tuner | Trials (config) | Searches lookback? |
|---|---|---|---|
| lightgbm, xgboost, catboost | `tune_bayesian.py` | `bayes.n_trials` (25) | n/a (lag windows fixed) |
| lstm, transformer | `tune_dl.py` | `bayes.n_trials_dl` (10) | **yes** |
| tide, tsmixer | `tune_dl.py` | `bayes.n_trials_dl` (10) | **yes** |

All use Optuna TPE (`multivariate=True`, seeded from `project.random_seed`), all minimise a
validation metric measured on a **real 14-day forecast from the train-end origin**, and all log one
nested MLflow run per trial.

`n_trials_dl` defaults to 10 rather than 25 because each sequence-model trial trains a whole network,
which is far more expensive than a tree fit.

```bash
python -m demand_forecasting.tune_bayesian --model lightgbm
python -m demand_forecasting.tune_dl       --model lstm
python -m demand_forecasting.tune_dl       --model tide --n-trials 5
```

---

### Q12. How are metrics computed across all Store-SKU pairs, and what is the "final" number?

This extends Q2, which covers the pooling semantics in detail. The short version:

**The headline metric is pooled (micro-averaged) over every row in the window** — all 1,004 active
series × 14 horizon days = **14,056 rows** — not computed per series and averaged.

```python
overall = regression_metrics(df[y_col], df[pred_col])   # one number over all rows
```

So `metrics.json → validation.overall.wape` is a single global figure. WAPE and bias are ratios of
sums (**volume-weighted**); MAE, RMSE and SMAPE are means over rows.

**Per-series and per-horizon numbers now exist**, via `report_best_models.py`, which was added
precisely because the pooled figure hides segment failure:

| Report | Contents |
|---|---|
| `reports/v1/best_models_per_series_metrics.csv` | one row per `(model, store_id, sku_id)`, sorted worst-WAPE-first |
| `reports/v1/best_models_per_horizon_metrics.csv` | one row per `(model, horizon_day)` — D+1 … D+14 |
| `reports/v1/best_models_pooled_vs_macro.csv` | pooled vs macro WAPE, plus best/worst series |

The per-horizon view is the one most worth reading, because it exposes error growth across the
horizon that a single pooled number cannot. From a smoke run (LightGBM):

```text
horizon_day   wape     mape
          1  0.2511   0.2745
          2  0.1638   0.3013
          3  0.2863   4.2483   <- MAPE explodes on a low-demand row; WAPE barely moves
          9  0.6907   4.1418
         14  0.1941   0.2117
```

That is Q3's argument in miniature: the same rows give a stable WAPE and a wildly unstable MAPE.

**"Final metrics" therefore means:** the pooled metric over the whole 14-day test window across all
Store-SKU pairs — not an average of per-series numbers, and not an average of per-day numbers.
Compare pooled against macro in `best_models_pooled_vs_macro.csv` before trusting the headline.

---

### Q13. Are inference results stored for every Store-SKU pair? How does that work?

Yes, at row granularity — one row per `(date, store_id, sku_id)`.

**During training/evaluation**, each model writes its own predictions next to its artifact:

```text
artifacts/<model>/validation_predictions.csv    # 14,056 rows: 1,004 series x 14 days
artifacts/<model>/test_predictions.csv          # 14,056 rows
```

These carry the full truth row plus a `prediction` column, so actuals, promo flags, prices and costs
sit alongside the forecast — which is what makes the after-the-fact breakdowns possible without
retraining. Both are also logged to MLflow under `artifact_path="predictions"`.

**During batch inference**, all three CLIs emit the identical contract:

```text
date,store_id,sku_id,prediction
```

**Consolidated for the champions**, `report_best_models.py` writes
`reports/v1/best_models_test_predictions.csv` with the best model per family stacked together:

```text
date, store_id, sku_id, actual, prediction, model, family, error, abs_error, horizon_day
```

For production the recommendation in `explanation.md` §14 still stands: store forecasts keyed by
`forecast_origin`, `target_date`, series id, model version and data version, so a forecast can
always be traced to the model and data that produced it. The CSVs here are the local equivalent.

---

### Q14. Where do the runs live — DagsHub or local?

**By default, local sqlite** — and note this changed with MLflow 3.x:

```text
MLFLOW_TRACKING_URI unset  ->  sqlite:///mlflow.db      (the current default)
./mlruns file store        ->  now raises unless MLFLOW_ALLOW_FILE_STORE=true
```

The legacy `./mlruns` directory in this repo is a leftover from an older MLflow. View runs with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://localhost:5000
```

**For DagsHub**, set the environment and everything routes there automatically —
`configure_mlflow()` only calls `set_tracking_uri()` when `MLFLOW_TRACKING_URI` is present:

```bash
export DAGSHUB_USER=<user> DAGSHUB_REPO=<repo> DAGSHUB_TOKEN=<token>
source scripts/setup_dagshub.sh     # exports MLFLOW_TRACKING_URI / USERNAME / PASSWORD
```

Then runs appear at `https://dagshub.com/<user>/<repo>.mlflow`. `.env` already holds these values
locally and is gitignored — never commit the token.

What lands in every run: hyperparameters, the feature-contract flags, stockout strategy,
`train_*` / `val_*` / `test_*` metrics, business-proxy metrics, split-boundary tags, the model
artifact, and both prediction CSVs. Runs are tagged for filtering:

```text
tags.stage = 'final'                                 # the comparable, promotable models
tags.stage = 'tuning' and tags.model = 'lightgbm'    # every LightGBM trial
tags.stage = 'tuning_parent'                         # one row per study
tags.family = 'darts'
```

Local logs under `logs/<step>.log` are the parallel record and stay readable with no server at all.

---

### Q15. Is there a script that reports the best model per family with test inference vs actuals?

Yes — `report_best_models.py`, added for exactly this.

```bash
python -m demand_forecasting.report_best_models                  # uses config selection_metric
python -m demand_forecasting.report_best_models --metric mape    # best by lowest val MAPE
```

It scans every `artifacts/*/metrics.json`, groups models by family (`ml` / `dl` / `darts`), picks the
lowest `val_<metric>` **within each family**, then loads that model's `test_predictions.csv` and
compares against actuals. Output:

| File | Contents |
|---|---|
| `reports/v1/all_models_metrics.csv` | every model, train/val/test × wape, mape, mae, rmse, smape, bias |
| `reports/v1/best_models.csv` | the champion of each family, ranked overall |
| `reports/v1/best_models_test_predictions.csv` | per-row test predictions vs actuals, with `error` and `horizon_day` |
| `reports/v1/best_models_per_series_metrics.csv` | metrics per Store-SKU, worst first |
| `reports/v1/best_models_per_horizon_metrics.csv` | metrics per horizon day D+1 … D+14 |
| `reports/v1/best_models_pooled_vs_macro.csv` | pooled vs macro WAPE and best/worst series |

Example output from a smoke run:

```text
 overall_rank family    model  train_wape  val_wape  val_mape  test_wape  test_mape
            1     dl     lstm         NaN    0.2477     0.330     0.2729     0.9028
            2     ml lightgbm      0.1171    0.2486     0.347     0.2777     0.9880
            3  darts     tide         NaN    0.3599     0.396     0.3597     0.8772
```

`train_wape` is `NaN` for the sequence models because `train_dl.py` and `train_darts.py` do not score
the training set (they early-stop on validation instead); only `train_ml.py` reports in-sample fit.

---

### Q16. How do I actually run inference for the next 14 days? What request body does the API need?

**Don't hand-write the body — generate it**, so the contract is never guessed:

```bash
python -m demand_forecasting.make_future_template \
  --output-csv reports/future_covariates.csv \
  --output-json reports/forecast_request.json
# add --series-limit 2 for a small smoke test
```

This reads history, finds every series still active on the final date, builds the 14 contiguous
dates that follow, fills the deterministic calendar fields, defaults `promo_flag` to 0, and writes
both a CSV (for the batch CLIs) and a JSON body (for the API).

**Required fields** are derived from the contract, not fixed. Under the current config
(`promo_known_future: true`, price and weather `false`):

```json
{
  "future_covariates": [
    {"date": "2024-01-01", "store_id": "STORE0001", "sku_id": "SKU0001",
     "is_holiday": 0, "promo_flag": 0},
    {"date": "2024-01-02", "store_id": "STORE0001", "sku_id": "SKU0001",
     "is_holiday": 0, "promo_flag": 0}
  ]
}
```

Turning on `price_known_future` would additionally require `list_price` and `discount_pct` per row;
`weather_known_future` would require `temperature` and `rain_mm`.

**Validation rules** — all enforced before any model loads, each returning 422 with the reason:

1. every required field present;
2. dates parseable;
3. no duplicate `(store_id, sku_id, date)`;
4. exactly `horizon` (14) rows per series;
5. dates exactly contiguous, starting the day **after** the last history date;
6. the series must exist in history (static attributes are joined from its latest row).

Verified end to end — the generated body returns 200:

```text
request rows: 28   fields: [date, is_holiday, promo_flag, sku_id, store_id]
status: 200   horizon: 14   count: 28
  {'date': '2024-01-01', 'store_id': 'STORE0001', 'sku_id': 'SKU0001', 'prediction': 97.13}
```

**To serve a specific model**, point the registry at it — the API is model-agnostic and
`inference_router.py` dispatches on `family`:

```yaml
# configs/model_registry.yaml
selected_model:
  name: lstm
  family: dl            # ml | dl | darts
  model_type: lstm
  artifact_path: artifacts/lstm/model.pt
  # darts additionally needs: metadata_path: artifacts/tide/metadata.joblib
```

**To score every family's champion**, loop the registry over each and call the API, or skip HTTP
entirely and use the batch CLIs, which take the same `future_covariates.csv`:

```bash
python -m demand_forecasting.inference_ml --model artifacts/lightgbm/model.joblib \
  --history data/raw/data.csv --future reports/future_covariates.csv --output artifacts/forecast_lightgbm.csv

python -m demand_forecasting.inference_dl --model artifacts/lstm/model.pt \
  --history data/raw/data.csv --future reports/future_covariates.csv --output artifacts/forecast_lstm.csv

python -m demand_forecasting.inference_darts --model-type tide \
  --model artifacts/tide/model.pt --metadata artifacts/tide/metadata.joblib \
  --history data/raw/data.csv --future reports/future_covariates.csv --output artifacts/forecast_tide.csv
```

One caveat worth knowing: the Darts adapter requires **every trained series** to be present in the
request (`validate_future_coverage`), so a single-series API call fails when a Darts model is
promoted. The ML and DL adapters accept any subset.

---

### Q17. What else is worth knowing that the questions above did not cover?

**Things that will bite you**

1. **Never set `training.n_jobs` to the full CPU count.** The OpenMP backends collapse into
   spin-wait contention: the same LightGBM fit took 0.31s at 8 threads and 235s at 16. `-1` now
   resolves to half the CPUs. It looks like a hang, not a slowdown.
2. **MLflow no longer defaults to `./mlruns`** (Q14). Use `sqlite:///mlflow.db`.
3. **Tuning cost is dominated by `recursive_forecast`**, not by model fitting — it rebuilds causal
   features once per horizon day, 14× per trial. This is why `n_trials_dl` is 10 and why
   rolling-origin validation (Q1) multiplies cost linearly.
4. **`data/raw/data.csv` is 212 MB.** Track it with DVC; never commit it.

**Known gaps, honestly stated**

| Gap | Impact |
|---|---|
| Single validation origin (Q1) | selection rests on one December fortnight |
| Tree lag/rolling windows not tuned (Q10) | their "input chunk length" is fixed |
| No per-series metrics in `full_evaluation` | now available via `report_best_models.py`, but not in `metrics.json` |
| `train_dl` / `train_darts` skip train metrics | `train_wape` is `NaN` for sequence models |
| Darts inference needs all trained series (Q16) | blocks single-series API calls for a Darts champion |
| No rolling retrain / scheduled job | retraining is manual |

**Bugs found by auditing, now fixed** — each was silent, and worth knowing the class of failure:

- `splits.py` labelled validation `"val"` while tests asserted `"validation"`.
- `middle_out_allocate()` used `validate="one_to_many"` on parent keys that repeat per date, so it
  raised `MergeError` for **any** horizon beyond one day. `hierarchy.py` had zero test coverage.
- `inference_darts.py` called `TimeSeries.pd_dataframe()`, removed in Darts 0.39 — the entire Darts
  inference path was dead code, and `requirements.txt` pinned an unmet `>=0.47`.
- `strategy_analysis.py` called `read_raw()` without `cfg` and crashed on every run.
- `log_json_artifact()` and `evaluate_compare --artifacts` both ignored `training.save_dir`.
- `train_darts.build_model()` hardcoded `hidden_size=128`, which would have made tuning it a no-op.

**Reproducing any result** needs exactly four things: the DVC data version, the git commit, the
`Effective configuration` block in that step's log, and the MLflow run id printed in the same log.

---

## 9. The v1 production run

This section records the actual full-dataset run: how it was configured, what it cost, where the
artifacts live, what the results were, and what should change in v2.

### 9.1 How it was run

```bash
PYTHON=./venv/bin/python bash scripts/run_v1.sh
```

`scripts/run_v1.sh` is the single orchestrator. It runs, in order: `prepare_dataset` → tuning for
all seven models → final training for all seven → `evaluate_compare` → `report_best_models` →
`select_model` → `make_future_template`.

Two deliberate properties:

- **A failing model does not abort the run.** Each stage goes through a `run()` wrapper that records
  `PASS`/`FAIL` with a duration into `reports/v1_run_status.txt` and continues. One library-specific
  failure should not cost the other six models.
- **Training falls back gracefully.** `params_arg()` attaches `--params-json` only if that model's
  tuning actually produced a file, so a failed tuning stage degrades to default hyperparameters
  rather than crashing training.

Trial counts are per family because the cost profile differs by an order of magnitude:

| Family | Models | Trials | Why |
|---|---|---|---|
| tree | lightgbm, xgboost, catboost | 20 | ~1.7 min/trial |
| torch | lstm, transformer | 4 | each trial trains a network |
| darts | tide, tsmixer | 3 | slowest to construct and fit |

### 9.2 Measured cost, and the two decisions it forced

Everything below was measured on the full 1.1M-row dataset before the run, not estimated:

```text
read_raw                3.8s   1,100,000 rows
feature build           6.7s   <- recursive_forecast runs this 14x per forecast
lightgbm fit           23.2s   1,043,748 rows x 52 features
recursive_forecast     67.9s
=> one tree trial ~ 1.5 min

GPU: NVIDIA RTX 5060 Laptop (CUDA available)
DL training windows at lookback=56: 1,030,655  (4,025 batches/epoch)
DL dataset build:      99.9s
DL per __getitem__:    0.42ms  -> ~7 min/epoch of pure pandas indexing
DL epoch:              ~6.6 min
```

**Decision 1 — subsample DL training windows.** At ~6.6 min/epoch, one LSTM fit is 1-3 hours and
tuning it is 10-30 hours. The cost is dominated by per-sample pandas indexing on the CPU, not by GPU
compute — the GPU is starved. Consecutive windows also share 55 of their 56 history days, so the
full set is highly redundant. `dl.max_train_windows: 150000` samples windows with a seeded RNG,
keeping every series represented. **Validation, test and inference windows are never subsampled** —
only what the model trains on.

**Decision 2 — bound the epoch budget.** `dl.epochs` 30 → 15 and `patience` 5 → 3. Early stopping
still selects the epoch count; this only caps the worst case.

Both are v1 tractability tradeoffs, not modelling improvements — see §9.6.

### 9.3 Where the runs are tracked

**DagsHub MLflow**, verified reachable and writable before launch:

```text
https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow
experiment: fmcg-demand-forecasting-v1
```

Credentials come from `.env`. A gap was fixed to make this work: **nothing in the codebase ever
loaded `.env`**, so despite the file holding valid DagsHub credentials, every run had been going to
the local store. `configure_mlflow()` now calls `load_dotenv(override=False)` — the file supplies
defaults, and anything already exported in the shell still wins.

Run structure in the experiment:

| `tags.stage` | Run name | What it holds |
|---|---|---|
| `tuning_parent` | `<model>-tuning` | study config, best params, `best_val_*`, trials CSV |
| `tuning` | `<model>-tune-trial-NNN` | that trial's params + full validation metric set |
| `final` | `<model>-final` / `<model>-global` | tuned params, train/val/test metrics, model artifact, prediction CSVs |

Useful filters: `tags.stage = 'final'`, or `tags.stage = 'tuning' and tags.model = 'lightgbm'`.

### 9.4 The OOM, and the optimisation it forced

The first launch was **killed by the OS for low memory** partway through LightGBM tuning. The WSL VM
has 11.9 GB total, and the run was consuming close to all of it.

**Cause.** `recursive_forecast()` rebuilt causal features over the *entire* history once per horizon
day — a ~1.07M-row × 72-column frame, 14 times per forecast, plus pandas intermediates inside each
`groupby().transform()`. That is several GB of churn per trial.

**The insight.** Every feature is a *bounded* lag or rolling window. The longest is 28 days. History
older than that provably cannot influence a forecast, so rebuilding features over three years of
data to predict 14 days was pure waste.

`required_history_days(cfg)` derives the bound from config (`max(window) * 2 + horizon` = **70
days**), and `recursive_forecast` trims history to that tail per series before the loop — reducing
the per-day feature frame from ~1.07M rows to ~70k.

**The one subtlety.** `series_age_days` is a running counter (`groupby().cumcount()`), so trimming
would restart it at zero and silently corrupt the feature. The rows dropped are added back as a
per-series offset after each feature build.

**Verified, not assumed.** This sits on the leakage-critical path, so it is pinned by an equivalence
test rather than trusted — `tests/test_recursive_history_trim.py` runs the same forecast with and
without trimming and asserts the predictions match at `atol=0, rtol=0`, plus a second test that
`series_age_days` is identical either way.

Measured on the full dataset:

| | Before | After |
|---|---|---|
| `recursive_forecast` | 67.9s | **21.9s** (3.1× faster) |
| Feature frame per horizon day | ~1.07M rows | ~70k rows |
| Peak RSS during tuning | ~10 GB (OOM-killed) | **3.47 GB** |
| Estimated tree trial | ~1.5 min | **~45s** |

A second, smaller fix: `tune_bayesian` now `del`s the full feature table once `train` is extracted,
since every trial refits on `train` and `recursive_forecast` rebuilds its own features from history.
That frees several hundred MB for the duration of the study.

This is the rare case where the memory fix was also a correctness-neutral 3× speedup — worth
recording because the same "bounded window" argument applies anywhere else the pipeline
recomputes over full history.

### 9.5 Running under host memory pressure

The trim was not enough on its own. The run was killed twice more, and the second time it died
within seconds of starting — before the process had grown at all. The cause was **ambient host
pressure**, not this pipeline:

```text
Windows host   : 23,960 MB total, ~4,600 MB free with nothing of ours running
WSL2 VM        : 11,656 MB allocated
```

WSL2 claims memory from Windows and is slow to hand it back, so on a box already near its limit the
supervisor kills long-lived background jobs almost immediately regardless of what they are doing.

Four changes made the run survivable:

| Change | Effect |
|---|---|
| `bayes.max_tune_rows: 200000` | tuning ranks configurations; it does not need every row. Final training still uses all of them |
| `del bundle/pred/merged` + `gc.collect()` per trial | a fitted booster is hundreds of MB and the collector was not keeping up across trials |
| `training.n_jobs: 4` | each worker carries its own histogram buffers, so fewer threads is less memory as well as safely clear of the OpenMP cliff (§Q17) |
| **Resumable studies** | the important one, below |

**Resumability.** Each Optuna study now persists to `artifacts/tuning/<model>_study.db` with
`load_if_exists=True`, and `run_v1.sh` skips any training stage whose `metrics.json` already exists.
An interruption stops costing the completed work:

```text
run 1: --n-trials 2  -> 2 completed
run 2: --n-trials 4  -> "Resuming study: 2 trials already complete, 2 remaining"
                     -> total completed trials in study: 4
```

That behaviour was smoke-tested before relying on it, because a bug there would have silently
re-run everything.

**Execution mode.** With resumability in place the practical approach on this box is short
foreground chunks rather than one long background job — each tuning stage is re-entrant, so
progress accumulates across invocations however they are interrupted.

Post-trim timings, tuning on 200k rows:

```text
lightgbm : ~37s per trial
xgboost  : slower per trial (denser trees at these settings)
catboost : comparable to lightgbm
tide     : minutes per trial - Darts builds ~4,000 TimeSeries objects for 1,005 series
```

### 9.6 Experiment-tracking gaps found mid-run, and the fixes

Two reported problems — trial models missing from MLflow, and no train loss — were both confirmed
by querying the live DagsHub experiment rather than by reading code:

```text
experiment: fmcg-demand-forecasting-v1  (62 runs: 54 tuning, 8 tuning_parent)

[tuning] tide-tune-trial-000
    artifacts: NONE
    metrics  : []
[tuning] transformer-tune-trial-002
    artifacts: NONE
    metrics  : [epoch_train_weighted_mae_scaled, epoch_val_weighted_mae_scaled, val_*]
    has train metric: False
[tuning_parent] transformer-tuning
    artifacts: ['tuning']        <- proves artifact upload to DagsHub works
```

**Diagnosis.**

| Symptom | Cause |
|---|---|
| No trial models in MLflow | `bayes.log_trial_models` defaulted to `false`, **and** `tune_dl.py` had no model-persisting code at all — so sequence-model trials could never log one |
| No train loss on tree trials | `tune_bayesian` scored only the validation forecast; the training fit was never evaluated |
| No train-loss summary on torch trials | per-epoch `epoch_train_weighted_mae_scaled` was logged, but no summary metric at the selected epoch, so it did not appear as a run-level `train_*` value |
| Only "best" models seemed saved | correct — only the champion reached `artifacts/<model>/`, and only the parent run held tuning artifacts |

Note the parent run already carried artifacts, which ruled out a DagsHub permissions or transport
problem and pointed at our own code.

**Fixes.**

1. `bayes.log_trial_models: true` — every trial now persists its fitted model and logs it under the
   run's `model/` artifact path, so any configuration in the sweep can be reloaded without retraining.
2. `tune_dl.py` gained `save_trial_model()`, handling both the torch checkpoint format (state dict +
   `DLMetadata` + dl config + lookback/horizon, i.e. everything inference needs to rebuild the
   architecture) and the Darts `model.save()` format. Failure to upload is caught and logged so it
   can never lose a trial's metrics.
3. `tune_bayesian` now scores the training fit each trial and logs `train_wape`, `train_mape`,
   `train_mae`, `train_rmse`, `train_smape`, `train_bias`, making per-trial overfitting visible.
4. `train_with_validation()` now returns the train loss at the selected epoch and logs
   `train_weighted_mae_scaled` / `val_weighted_mae_scaled` as run-level metrics.

**Cost.** Logging every trial's binary is a few GB of uploads across a full sweep. That is the
intended tradeoff — full reproducibility of the search over disk — and it is a one-line config
change to revert.

### 9.7 v1 results

Full dataset, 1,004 forecastable Store-SKU series × 14 days = 14,056 scored rows per split.
Selection on **validation WAPE**; test evaluated once.

| rank | model | family | train_wape | **val_wape** | val_mape | test_wape | test_mape |
|---|---|---|---|---|---|---|---|
| 1 | **transformer** | dl | – | **0.2648** | 0.5470 | 0.2690 | 0.5447 |
| 2 | lstm | dl | – | 0.2650 | 0.5384 | 0.2684 | 0.5180 |
| 3 | xgboost | ml | 0.2644 | 0.2716 | 0.5395 | 0.2706 | 0.5226 |
| 4 | catboost | ml | 0.2664 | 0.2719 | 0.5461 | **0.2683** | 0.5283 |
| 5 | lightgbm | ml | 0.2659 | 0.2721 | 0.5458 | 0.2709 | 0.5329 |

(`train_wape` is blank for the sequence models because they early-stop on validation and never
score the training set — see §9.9.)

**Four things this table says.**

1. **Every model lands within 0.007 WAPE.** The spread from best to worst is 2.7% relative. On this
   dataset the choice of model family is close to irrelevant compared with the feature contract and
   the leakage discipline that all five share.

2. **The champion by validation is not the best on test.** Transformer wins validation (0.2648) but
   **catboost has the lowest test WAPE (0.2683)**, from 4th place on validation. The validation gap
   between 1st and 4th is 0.0071; the test ordering reverses it. This is direct evidence for the
   single-origin concern in Q1 — a 0.0002 gap between transformer and lstm is noise, and selecting
   on one 14-day December window cannot distinguish these models.

3. **Nothing is overfitting.** For the tree models train and validation WAPE are nearly identical
   (xgboost 0.2644 vs 0.2716). The models are at capacity, not memorising. More regularisation
   would not help; more signal might.

4. **MAPE is ~0.54 while WAPE is ~0.27 — exactly double.** With ~1.5% of rows under 5 units, MAPE is
   dominated by small-denominator rows, which is why it stays the diagnostic and WAPE the decision
   metric (Q3).

**Error is flat across the horizon.** Per-horizon WAPE for the champion:

```text
D+1  0.2680   D+6  0.2695   D+11 0.2615
D+2  0.2755   D+7  0.2652   D+12 0.2685
D+3  0.2816   D+8  0.2678   D+13 0.2609
D+4  0.2711   D+9  0.2901   D+14 0.2742
D+5  0.2519   D+10 0.2629
```

There is **no error growth from D+1 to D+14** — the range is 0.252–0.290 with no trend. For the
direct multi-horizon Transformer that is expected (each day is predicted from its own known-future
query rather than from a chain of predictions), and it means the 14-day horizon is not the limiting
factor. Weekly seasonality is already captured.

**Pooled vs macro, and the per-series spread:**

| model | pooled test WAPE | macro test WAPE | worst series | best series |
|---|---|---|---|---|
| transformer | 0.2690 | 0.2764 | 0.6458 | 0.0958 |
| xgboost | 0.2706 | 0.2739 | 0.5450 | 0.1239 |

Pooled and macro are close, so the headline is not being propped up by high-volume series. But the
**per-series range is 0.10 to 0.65** — a 6.7× spread. The aggregate hides that some Store-SKU pairs
are forecast three times worse than others, which is where the next real accuracy gain lives
(§9.9).

**Darts (TiDE/TSMixer) is absent from v1.** Its tuning failed on a genuine bug, found and fixed
during the run: `forecast_from_origin()` did not exclude series whose data ends before the forecast
origin. Exactly one series does — `STORE0013 / SKU0073`, last seen **2022-09-12** — and Darts
rejected the whole batch because that series' known-future covariates could not span the horizon:

```text
ValueError: For the given forecasting horizon `n=14`, the provided `future_covariates`
at series sequence index `992` do not extend far enough into the future.
```

The fix filters to series whose data reaches `origin + horizon`, which yields **1,004 of 1,005** —
matching the active-series count the EDA reported. TiDE/TSMixer are covered in v2.

### 9.8 Where the artifacts live, and what each one is for

```text
artifacts/
├── tuning/
│   ├── <model>_best_params.json     winning hyperparameters, consumed by --params-json
│   ├── <model>_trials.csv           every trial: params, metrics, state
│   ├── <model>_study.db             Optuna storage - makes tuning resumable
│   └── <model>_trial_NNN.joblib|pt  every trial's fitted model (log_trial_models)
├── <model>/
│   ├── model.joblib | model.pt      the deployable artifact, refit on train+validation
│   ├── metadata.joblib              darts only: series keys, static maps, feature contract
│   ├── metrics.json                 train/validation/test metrics + breakdowns
│   ├── validation_predictions.csv   per-row: 14,056 rows
│   └── test_predictions.csv         per-row: 14,056 rows
└── <model>_evaluation.json          the same evaluation, logged to MLflow

reports/
├── model_comparison.csv                     all models ranked by the selection metric
├── all_models_metrics.csv                   train/val/test x 6 metrics for every model
├── best_models.csv                          the champion of each family
├── best_models_test_predictions.csv         champions' test predictions vs actuals
├── best_models_per_series_metrics.csv       one row per Store-SKU (1,004 rows)
├── best_models_per_horizon_metrics.csv      one row per horizon day D+1..D+14
├── best_models_pooled_vs_macro.csv          pooled vs macro WAPE, best/worst series
├── forecast_store_sku.csv                   THE DELIVERABLE: 14-day forecast, long form
├── forecast_store_sku_wide.csv              same, one column per model
└── v1_run_status.txt                        PASS/FAIL/SKIP trail with durations

configs/model_registry.yaml                  the promoted champion the API serves
logs/<step>.log                              per-step narrative, midnight rotation
```

**What to read for what:**

| Question | File |
|---|---|
| Which model do we ship? | `configs/model_registry.yaml` |
| How do the models compare? | `reports/v1/model_comparison.csv` |
| Best in each family? | `reports/v1/best_models.csv` |
| **The 14-day Store-SKU forecast** | `reports/v1/forecast_store_sku.csv` |
| Which series forecast badly? | `reports/v1/best_models_per_series_metrics.csv` (sorted worst-first) |
| Does error grow over the horizon? | `reports/v1/best_models_per_horizon_metrics.csv` |
| What did trial 7 of xgboost do? | `artifacts/tuning/xgboost_trials.csv`, or its MLflow run |
| Why did a run behave oddly? | `logs/<step>.log` |

**The deliverable forecast** — `reports/v1/forecast_store_sku.csv`, 28,112 rows
(2 champions × 1,004 series × 14 days), covering **2024-01-01 → 2024-01-14**, the horizon
immediately after history ends:

```text
date,store_id,sku_id,prediction,model,family
2024-01-01,STORE0001,SKU0001,104.20539093017578,transformer,dl
2024-01-02,STORE0001,SKU0001,103.56144714355469,transformer,dl
```

**The promoted model** (`configs/model_registry.yaml`) records the artifact path, the metric that
won, and both validation and test metrics, so what is being served and why is auditable without
opening MLflow:

```yaml
selected_model:
  name: transformer
  family: dl
  model_type: transformer
  artifact_path: artifacts/transformer/model.pt
  selection_metric: validation_wape
  selection_value: 0.2648181843655995
  validation_metrics: {wape: 0.2648, mape: 0.5470, mae: 16.45, rmse: 25.12, bias: 0.0226}
  test_metrics:       {wape: 0.2690, mape: 0.5447, mae: 16.79, rmse: 25.71, bias: 0.0275}
```

`inference_router.py` reads `family` and dispatches to the right adapter, so the API needs no
knowledge of which model is deployed.

### 9.9 Which model actually won, and what I would ship

**Selected: the global Transformer** (validation WAPE 0.2648). But the honest reading is more
interesting than the ranking.

**The models are statistically indistinguishable.** Best to worst spans 0.2648 → 0.2721 on
validation — 2.7% relative — and the ordering *reverses* on test, where catboost (4th on validation)
posts the best number. A 0.0002 gap between transformer and lstm is not a result; it is noise from a
single 14-day origin. Anyone reporting "the Transformer is the best model" from this evidence is
overclaiming.

> **Confirmed by v2 (§9.13).** With a doubled search the Transformer again won validation and again
> failed to be best on test — this time finishing *worst* of the non-darts models. LSTM took the
> test crown. The recommendation below is not a hedge; it is what the evidence supports.

**What I would actually ship: CatBoost or LightGBM**, despite the Transformer winning validation.
The accuracy difference is inside the noise band, and the tree models are dramatically cheaper to
operate:

| | tree | sequence |
|---|---|---|
| Fit time | ~25s | 10–30 min |
| Tuning trial | ~40s | 7–30 min |
| Inference | CPU, ~22s for all 1,004 series | GPU preferred |
| Explainability | SHAP / feature importance out of the box | attention inspection at best |
| Failure mode | degrades gracefully | needs the exact feature schema and checkpoint |
| Retraining cost | trivial | needs GPU scheduling |

CatBoost is the concrete pick: across both runs it is the only model that is simultaneously
top-three on validation, top-three on test, stable between v1 and v2 (identical to five decimal
places), and cheap to retrain. You do not pay 30× the training cost and add a GPU dependency for
0.007 WAPE that reverses on the test window. The Transformer is a genuine challenger worth keeping in the harness — if v2 with a
larger budget opens a *consistent* gap across multiple origins, that changes the calculus.

**The strongest v1 finding is not which model won — it is that the model barely matters.** Five
architectures with very different inductive biases converged to within 3% of each other. That says
the remaining error is not a modelling-capacity problem: train and validation WAPE are nearly
identical, so nothing is overfitting, and no architecture extracted signal the others missed. The
ceiling is being set by the information in the features, not the function class on top of them.

That reframes where v2 effort should go.

### 9.10 What v2 should change

Ordered by expected value per unit of effort.

**1. Rolling-origin validation (highest value).** Everything above rests on one December fortnight,
and the validation ordering already contradicts the test ordering. Six to twelve origins spread
across seasons, selecting on mean WAPE with the spread reported, would turn "transformer wins by
0.0002" into a defensible statement. This is the single change that would most improve decision
quality, and it is already scoped in Q1.

**2. Attack the per-series spread, not the average.** Per-series WAPE ranges 0.10 → 0.65. The
aggregate is healthy while some series are forecast 6.7× worse than others. Concretely: segment the
worst decile, check whether it is low-volume, promo-heavy, or stockout-prone, and consider a
specialist model or per-segment weighting. This is where real accuracy lives, and v1 has the data to
start — `reports/v1/best_models_per_series_metrics.csv` is sorted worst-first.

**3. More signal, not more capacity.** Since nothing overfits, add information rather than
parameters:
- promo *depth* interactions and lead/lag effects around campaigns (promo lift is 63–131% by
  category per the EDA, and is currently a single binary flag in the future covariates);
- price relative to category and to competitor SKUs in the same subcategory;
- calendar richness: paydays, school terms, local events, holiday proximity rather than a same-day
  binary;
- turning on `price_known_future` *if* the business genuinely commits prices 14 days ahead — that is
  a contract question, not a modelling one, and the code already supports the switch everywhere.

**4. Revisit the stockout target.** `stockout_target_mode` is still `none`, so the model trains on
censored sales as if they were demand on 3% of rows. `rolling_median` and `percentage` are
implemented and leakage-tested; v1 simply never ran the comparison. That is a cheap, well-scoped
experiment.

**5. Quantile forecasts instead of point forecasts.** Inventory decisions need a service level, not
a mean. Pinball loss at P50/P90 would make the output directly usable for safety-stock sizing, and
would make the asymmetric business proxy (`underforecast_margin_risk` vs
`overforecast_purchase_cost`) actionable rather than descriptive.

**6. Native categorical handling for CatBoost.** v1 ordinal-encodes for all three libraries to keep
one inference contract (§Q7). A native-CatBoost branch with ordered target statistics is a fair test
of whether that uniformity costs accuracy.

**7. Engineering follow-ups.**
- Score the training set in `train_dl`/`train_darts` so `train_wape` is populated for every family.
- Tune the tree lag/rolling windows — currently the only untuned "input chunk length" (§Q10).
- Vectorise `MultiSeriesWindowDataset.__getitem__`; at 0.42 ms per sample the GPU is starved, and
  removing `max_train_windows` entirely would then be affordable.
- Add rolling retraining on a schedule with the drift report as the trigger.

**Explicitly not worth doing in v2:** a bigger Transformer. v1 shows the architecture is not the
constraint.

### 9.11 The v2 run

v2 keeps the data, features and leakage contract identical to v1 and changes **only the search
budget and training schedule**, so the two are directly comparable.

`configs/config_v2.yaml`:

| Setting | v1 | v2 | Why |
|---|---|---|---|
| `mlflow_experiment` | `fmcg-demand-forecasting-v1` | `fmcg-demand-forecasting-v2` | separate experiment, v1 preserved |
| `training.save_dir` | `artifacts/` | `artifacts_v2/` | v1 artifacts never overwritten |
| `bayes.n_trials` (tree) | 12 | **25** | wider search |
| `bayes.n_trials_dl` | 2–3 | **4** | wider search |
| `dl.epochs` | 15 | **50** | longer schedule |
| `dl.patience` | 3 | **8** | let early stopping do the choosing |
| `bayes.log_trial_models` | false | **true** | every trial's model logged |

Everything else — the split, the `*_known_future` contract, `max_tune_rows`, `max_train_windows`,
`n_jobs` — is unchanged, so any difference in results is attributable to the budget.

**Commands.** Because the config carries the experiment name and artifact directory, the same
orchestrator runs v2:

```bash
PYTHON=./venv/bin/python CONFIG=configs/config_v2.yaml bash scripts/run_v1.sh
```

Or stage by stage, which is what was actually used here (short, resumable chunks are the practical
mode on a memory-constrained box — §9.5):

```bash
export PYTHONPATH=src
C=configs/config_v2.yaml

# tuning - every trial logged to MLflow with params, train + validation metrics, and its model
for m in lightgbm xgboost catboost; do
  python -m demand_forecasting.tune_bayesian --config $C --model $m --n-trials 25
done
for m in lstm transformer tide tsmixer; do
  python -m demand_forecasting.tune_dl --config $C --model $m --n-trials 4
done

# final training on the tuned hyperparameters
for m in lightgbm xgboost catboost; do
  python -m demand_forecasting.train_ml --config $C --model $m \
    --params-json artifacts_v2/tuning/${m}_best_params.json
done
for m in lstm transformer; do
  python -m demand_forecasting.train_dl --config $C --model $m \
    --params-json artifacts_v2/tuning/${m}_best_params.json
done
for m in tide tsmixer; do
  python -m demand_forecasting.train_darts --config $C --model $m \
    --params-json artifacts_v2/tuning/${m}_best_params.json
done

# comparison, per-family champions, promotion, deliverable forecast
python -m demand_forecasting.evaluate_compare    --config $C
python -m demand_forecasting.report_best_models  --config $C
python -m demand_forecasting.select_model        --config $C --name <champion> --family <ml|dl|darts> \
  --model-type <champion> --artifact artifacts_v2/<champion>/model.<ext> \
  --metrics artifacts_v2/<champion>/metrics.json
python -m demand_forecasting.run_final_inference --config $C
```

**Files involved in a run**, in execution order:

| Stage | Module | Reads | Writes |
|---|---|---|---|
| prepare | `prepare_dataset.py` | `data/raw/data.csv`, config | `data/processed/features.parquet` |
| tune (tree) | `tune_bayesian.py` | raw, config | `tuning/<m>_best_params.json`, `_trials.csv`, `_study.db`, `_trial_NNN.joblib` |
| tune (seq) | `tune_dl.py` | raw, config | same, `.pt` for trial models |
| train | `train_ml.py` / `train_dl.py` / `train_darts.py` | raw, best params | `<m>/model.*`, `metrics.json`, `*_predictions.csv` |
| compare | `evaluate_compare.py` | `*/metrics.json` | `reports/v1/model_comparison.csv` |
| report | `report_best_models.py` | `*/metrics.json`, `test_predictions.csv` | `reports/v1/best_models*.csv` |
| promote | `select_model.py` | comparison, metrics | `configs/model_registry.yaml` |
| forecast | `run_final_inference.py` | registry/best models, raw | `reports/v1/forecast_store_sku*.csv`, `v1_store_sku_forecast_and_metrics.csv` |

**The consolidated Store-SKU deliverable.**
`reports/v1/store_sku_forecast_and_metrics.csv` joins the next-14-day forecast with how accurately
that same series was predicted on the held-out test window — one row per (model, store, SKU), sorted
worst-accuracy first:

```text
model,family,store_id,sku_id,forecast_total_units,forecast_mean_units,
  test_rows,test_actual_units,test_wape,test_mape,test_mae,test_rmse,test_bias
transformer,dl,STORE0002,SKU0015,160.71,11.48,14,102.0,0.6458,1.6020,4.7048,5.5968,0.6147
```

This is the file to hand a planner: it says both *what we forecast* and *how much to trust it for
this specific Store-SKU*. The first row is instructive — the worst-forecast series is being
over-predicted by 61% (`test_bias` +0.61), which is an actionable, series-level finding that the
pooled 0.269 WAPE completely hides.

**v2 tuning results (tree models, 25 trials each, 75 trials total).**

Best validation WAPE from the search, measured on the 200k-row tuning cap:

| model | v1 (12 trials) | v2 (25 trials) | change |
|---|---|---|---|
| lightgbm | 0.27668 | **0.27595** | −0.0007 |
| xgboost | — | **0.27197** | — |
| catboost | 0.27457 | **0.27457** | 0.0000 |

Doubling the search budget moved LightGBM by 0.0007 WAPE and moved CatBoost **not at all** — TPE
had already found the same optimum in 12 trials. This is worth stating plainly: **the search budget
was not the binding constraint in v1.** It corroborates §9.9 — five architectures landing within
0.007 of each other, no overfitting gap, and now a doubled search that changes nothing, all point
at the feature set rather than the model or its hyperparameters as the ceiling.

Every one of those 75 trials is in MLflow with its params, train and validation metrics, and its
fitted model artifact (2.7–7.7 MB each), so the whole search is reproducible rather than just its
winner.

### 9.12 v1 vs v2: the budget was not the constraint

Both versions trained end to end on the full data. Comparing the final models (all rows, not the
tuning cap):

| model | v1 val_wape | v2 val_wape | delta | v1 train_wape | v2 train_wape |
|---|---|---|---|---|---|
| catboost | 0.27193 | 0.27193 | **0.00000** | 0.2664 | 0.2664 |
| lightgbm | 0.27208 | 0.27217 | +0.00009 | 0.2659 | 0.2637 |
| xgboost | 0.27158 | 0.27213 | +0.00056 | 0.2644 | **0.2584** |
| transformer | 0.26482 | 0.26542 | +0.00060 | – | – |
| lstm | 0.26504 | 0.26617 | +0.00113 | – | – |

**Not one model improved.** Every delta is zero or positive (worse), across both families and both
kinds of extra budget. CatBoost converged to the *identical* configuration, so TPE had already found
that optimum in 12 trials.

Test-set numbers for the same models:

| model | v1 test_wape | v2 test_wape | v1 test_mape | v2 test_mape |
|---|---|---|---|---|
| catboost | 0.2683 | 0.2683 | 0.5283 | 0.5283 |
| lightgbm | 0.2709 | 0.2704 | 0.5329 | 0.5304 |
| xgboost | 0.2706 | 0.2714 | 0.5226 | 0.5199 |
| lstm | 0.2684 | **0.2673** | 0.5180 | **0.5134** |
| transformer | 0.2690 | 0.2747 | 0.5447 | 0.5728 |

The v2 LSTM posts the best test WAPE (0.2673) and best test MAPE (0.5134) of any model in either
run, while ranking mid-table on validation — the validation/test ordering disagrees yet again.

The XGBoost row is the instructive one. With 25 trials the search found a configuration that fits
the training set noticeably better (train WAPE 0.2644 → 0.2584) while validating slightly *worse*
(0.27158 → 0.27213). That is the hyperparameter search overfitting to the single validation window:
given more attempts, TPE finds configurations that exploit the particular fortnight it is scored on.
More search against one origin buys precision on that origin, not generalisation.

The same held for the sequence models: raising `epochs` 15 → 50 with `patience` 8 did not improve
LSTM either (0.26504 → 0.26617). Early stopping was already selecting its epoch well inside the v1
budget, so the extra ceiling went unused.

**What this settles.** Three independent lines of evidence now point the same way:

1. five architectures land within 0.007 WAPE of each other (§9.7);
2. train and validation WAPE are nearly identical, so nothing is overfitting the *data* (§9.7);
3. doubling the search budget and tripling the epoch budget changes nothing (here).

The ceiling is the information in the feature set, not the model family, its hyperparameters, or
the optimisation budget. **v3 effort should go to features, the stockout target, and rolling-origin
validation — not to bigger searches or bigger networks.** That also makes §9.10's ordering concrete:
item 1 (rolling origins) and item 3 (more signal) are the ones that matter; item 7's "tune more"
suggestions are now demonstrably low value.

### 9.13 Final v2 standings, all six models

All six trained end to end on the full data. Selection on validation WAPE, test evaluated once:

| rank | model | family | train_wape | **val_wape** | val_mape | test_wape | test_mape | test_mae |
|---|---|---|---|---|---|---|---|---|
| 1 | **transformer** | dl | – | **0.2654** | 0.5336 | 0.2747 | 0.5728 | 17.14 |
| 2 | lstm | dl | – | 0.2662 | 0.5340 | **0.2673** | **0.5134** | **16.68** |
| 3 | catboost | ml | 0.2664 | 0.2719 | 0.5461 | 0.2683 | 0.5283 | 16.75 |
| 4 | xgboost | ml | 0.2584 | 0.2721 | 0.5358 | 0.2714 | 0.5199 | 16.94 |
| 5 | lightgbm | ml | 0.2637 | 0.2722 | 0.5447 | 0.2704 | 0.5304 | 16.88 |
| 6 | tide | darts | – | 0.2762 | 0.5535 | 0.2694 | 0.5263 | 16.81 |

Best per family: **transformer** (dl), **catboost** (ml), **tide** (darts) — all three promoted into
`reports/v2/best_models.csv` and used for the final forecast.

**The validation winner is the test loser.** Transformer ranks 1st on validation (0.2654) and
**last but one on test (0.2747)** — the single worst test WAPE of the five non-darts models. LSTM
ranks 2nd on validation and is best on test by every measure (WAPE 0.2673, MAPE 0.5134, MAE 16.68).
The validation gap between them is 0.0008; the test gap is 0.0074 the other way.

Across v1 and v2 the validation ranking has now failed to predict the test ranking **three times**:

| run | validation winner | actual best on test |
|---|---|---|
| v1 | transformer (0.2648) | catboost (0.2683) |
| v2 | transformer (0.2654) | lstm (0.2673) |
| v1→v2 budget | 25 trials "should" beat 12 | 12 trials won or tied every time |

That is the empirical case for Q1 stated three different ways. A single 14-day origin cannot rank
models whose true differences are ~0.005 WAPE, and every conclusion drawn from it — including
"the Transformer is our champion" — carries that caveat.

**Spread across all six models is 0.2654 → 0.2762 on validation and 0.2673 → 0.2747 on test**, i.e.
about 4% relative in both cases, across tree ensembles, recurrent, attention and MLP-mixer
architectures. TiDE, the only model that never got a tuned search (its study failed on the Darts bug
and it was trained with sensible defaults at a capped 12 epochs), still lands within 0.4% of the
tuned Transformer on test. That is the strongest single statement in this whole report about where
the remaining error lives.

**A second Darts bug, found by running inference.** `train_darts` was fixed to skip series that end
before the forecast origin (§9.7), but `inference_darts` iterates the *trained* key list and hit the
same dead series:

```text
ValueError: Expected 14 future rows for series ('STORE0013', 'SKU0073'), found 0
```

A model legitimately trains on series that later go inactive; inference must drop them rather than
fail the batch. `forecastable_keys()` now filters the trained keys to those the request actually
supplies a full horizon for. With that, all three families forecast cleanly:

```text
transformer (dl)    -> 14,056 rows
catboost    (ml)    -> 14,056 rows
tide        (darts) -> 14,056 rows
                       42,168 rows total, 1,004 series, 2024-01-01..2024-01-14
```

### 9.14 v2 outputs

```text
artifacts_v2/
├── tuning/                      75 tree trial models + 3 study DBs + best params + trials CSVs
├── lightgbm|xgboost|catboost/   model.joblib, metrics.json, {validation,test}_predictions.csv
├── lstm|transformer/            model.pt, metrics.json, {validation,test}_predictions.csv
└── tide/                        model.pt, metadata.joblib, metrics.json, predictions

configs/model_registry_v2.yaml   promoted v2 champion (transformer)

reports/
├── v1/  … the v1 equivalents of everything below
└── v2/
    ├── best_models.csv                           champion per family
    ├── all_models_metrics.csv
    ├── best_models_test_predictions.csv
    ├── best_models_per_series_metrics.csv
    ├── best_models_per_horizon_metrics.csv
    ├── best_models_pooled_vs_macro.csv
    ├── forecast_store_sku.csv                    42,168-row Store-SKU forecast (3 champions)
    ├── forecast_store_sku_wide.csv               one column per model
    └── store_sku_forecast_and_metrics.csv        forecast + that series' test accuracy
```

v1 remains untouched in `artifacts/`, `reports/` and `configs/model_registry.yaml`, so the two runs
can be compared at any time.

MLflow experiments on DagsHub:

```text
fmcg-demand-forecasting-v1   62+ runs   12 tree trials/model, 2-3 sequence trials
fmcg-demand-forecasting-v2   ~90 runs   25 tree trials/model, 2 sequence trials, every trial model logged
```

**A design fix this surfaced.** Final inference initially failed for the Transformer:

```text
ValueError: Config lookback does not match the trained checkpoint
```

The guard was right — the Transformer was trained with a **tuned lookback of 112** while
`config.yaml` still said 56. But the conclusion was wrong: now that lookback is searched per model
(§Q10), it is a property of the *artifact*, not of global config. `inference_dl` and
`inference_darts` now take lookback from the checkpoint/metadata and still validate the feature
contract and horizon strictly. Without this, any tuned sequence model would have been undeployable.

---

# Part II — Technical deep dive

Everything above describes the system as designed. This part is the walkthrough: the data as it
actually is, the questions the EDA answered, exactly how each model was fed and trained, what the
numbers mean, and how a forecast is produced for dates that do not exist yet. Written to be read
end to end by someone who needs to rebuild or defend this work.

---

## 10. The dataset, column by column

1,100,000 rows · 2021-01-01 to 2023-12-31 · 1,095 distinct dates · 13 stores · 102 SKUs ·
**1,005 store-SKU series** · 33 columns · **zero missing values anywhere**.

Every series is a daily time series of `units_sold` for one `(store_id, sku_id)` pair. One row per
pair per date, verified rather than assumed.

### 10.1 Numeric columns

| Column | Mean | Median | Std | Min | Max | P05 | P95 | Distinct |
|---|---|---|---|---|---|---|---|---|
| **units_sold** (target) | 59.20 | 49 | 45.01 | 0 | 704 | 10 | 142 | 516 |
| list_price | 7.71 | 7.38 | 4.25 | 1.08 | 14.80 | 1.44 | 14.15 | 99 |
| discount_pct | 0.015 | 0.00 | 0.055 | 0.00 | 0.30 | 0.00 | 0.15 | 5 |
| promo_flag | 0.080 | 0 | 0.272 | 0 | 1 | 0 | 1 | 2 |
| gross_sales | 440.68 | 282.88 | 441.80 | 0 | 6,593.90 | 44.91 | 1,344.25 | 15,354 |
| net_sales | 429.95 | 277.86 | 422.50 | 0 | 5,144.94 | 44.88 | 1,307.25 | 29,938 |
| stock_on_hand | 299.48 | 300 | 80.07 | 0 | 698 | 168 | 431 | 640 |
| stock_out_flag | 0.030 | 0 | 0.171 | 0 | 1 | 0 | 0 | 2 |
| lead_time_days | 6.50 | 6 | 2.01 | 1 | 17 | 3 | 10 | 17 |
| purchase_cost | 4.63 | 4.35 | 2.66 | 0.49 | 11.10 | 0.84 | 9.18 | 1,062 |
| margin_pct | 0.385 | 0.389 | 0.102 | **−0.05** | 0.55 | 0.25 | 0.53 | 601 |
| temperature | 12.82 | 12.84 | 3.37 | 1.80 | 22.83 | 7.25 | 18.53 | 710 |
| rain_mm | 2.90 | 2.57 | 2.10 | 0.00 | 11.58 | 0.23 | 6.85 | 559 |
| is_weekend | 0.287 | 0 | 0.452 | 0 | 1 | 0 | 1 | 2 |
| is_holiday | 0.014 | 0 | 0.116 | 0 | 1 | 0 | 0 | 2 |
| latitude | 46.31 | 45.46 | 4.60 | 40.42 | 52.53 | — | — | 13 |
| longitude | 9.03 | 9.20 | 6.73 | −3.68 | 21.00 | — | — | 13 |

Calendar integers (`year`, `month`, `day`, `weekofyear`, `weekday`) are also present and span their
natural ranges.

### 10.2 Categorical columns and cardinality

| Column | Distinct | Most common | Share |
|---|---|---|---|
| store_id | 13 | STORE0001 (87,600 rows) | 8.0% |
| sku_id | 102 | SKU0001 (14,235 rows) | 1.3% |
| sku_name | 102 | BrandA Soda | 1.3% |
| country | 7 | Italy (350,400 rows) | 31.9% |
| city | 9 | Berlin (175,200 rows) | 15.9% |
| channel | 4 | Hypermarket (525,600 rows) | 47.8% |
| category | 5 | Beverages (266,085 rows) | 24.2% |
| subcategory | 17 | Soda (71,175 rows) | 6.5% |
| brand | 6 | BrandF (187,245 rows) | 17.0% |
| supplier_id | 60 | S037 (18,579 rows) | 1.7% |

Cardinality is low to moderate throughout. That is what makes both ordinal encoding for the trees
and modest embedding tables for the neural models practical — a 102-way SKU embedding is cheap,
whereas 100,000 SKUs would have forced a different design.

### 10.3 The three facts that shaped everything

**Demand is right-skewed.** Mean 59.2 against median 49, max 704, standard deviation 45.0. The mean
sits above the median and the top of the range is 14× it. Squared-error objectives would let a
handful of spikes dominate the gradient, so every model optimises absolute error instead and WAPE
is the decision metric.

**Promotions are rare and large.** 88,257 rows carry a promotion, 8.02% of the data. Mean demand on
those rows is **104.27** against **55.26** on non-promotional rows — an 89% lift overall, and 63% to
131% depending on category. Rare plus large is exactly the combination an aggregate metric hides.

**Stockouts are not zero-inventory events.** 33,114 rows are flagged, 3.01% of the data. Of those,
**33,110 still show positive `stock_on_hand`** — only 4 rows have zero inventory. The flag and the
inventory column measure different things, so a stockout cannot be re-derived from stock levels and
the flag has to be trusted as given.

---

## 11. What the EDA asked, and what it settled

The notebook (`notebooks/01_eda.ipynb`) is organised as questions, because each answer became a
configuration value. This is the full chain from finding to setting.

| # | Question | What the data said | What it fixed |
|---|---|---|---|
| 1 | Are there missing values or duplicate keys? | None; exactly one row per (store, SKU, date) | Lags computed directly, no imputation layer beneath them |
| 2 | Is a stockout the same as zero inventory? | No — 33,110 of 33,114 stockouts have positive stock | The flag is authoritative and irreducible |
| 3 | Is the product/store hierarchy stable over time? | Each SKU maps to exactly one category/subcategory/brand; each store to one country/city/channel | Static attributes encoded once; no effective-dated master data needed |
| 4 | Is history dense enough for bottom-level forecasting? | Median series has all 1,095 days; ≥75% complete | A single **global** model over all series, not ~1,000 local ones |
| 5 | Are any series dead? | One: STORE0013/SKU0073 ends 2022-09-12, explaining a 475-row shortfall | Active forecast set fixed at **1,004** series |
| 6 | What shape is the target? | Right-skewed, median 49 / mean 59 / max 704 | L1 objectives; WAPE over RMSE for decisions |
| 7 | Is there weekly seasonality? | Autocorrelation peaks cleanly at **7, 14, 21, 28** | Demand lags 1/7/14/28; rolling windows 7/14/28 |
| 8 | Is there annual seasonality? | Peaks Jun–Aug and Oct–Dec, repeating all three years | Cyclical day-of-week, month and day-of-year encodings |
| 9 | Do weekends differ? | Yes, Saturdays and Sundays are consistently higher | `is_weekend` as an explicit feature |
| 10 | Do promotions matter, and how much? | 63–131% lift by category; demand rises monotonically with discount depth | Promotion flag as a **known-future** covariate, plus a dedicated promo error slice |
| 11 | Is promo lift causal? | No — promotions are scheduled, plausibly onto already-strong periods | Used as a covariate; no causal uplift claimed |
| 12 | How common are stockouts? | 3.01% of rows | Configurable censored-demand target with down-weighting |
| 13 | Are price and weather known 14 days ahead? | Present historically, but not committed | Both excluded from the future covariate set |
| 14 | How heterogeneous are series? | Scale varies widely; shared seasonal timing, series-specific magnitude | A *conditional* global model with identity features, not one unconditioned curve |

**Why 7/14/28 and not something else.** The autocorrelation peaks are the entire justification. Lag
1 captures short-run level, lag 7 is the same weekday last week (the dominant peak), and 14 and 28
confirm the weekly signal is stable while spanning a monthly cycle. Rolling windows use the same
periods so every trailing statistic covers a whole number of weeks.

**Why cyclical encodings.** As plain integers, Sunday (6) sits six units from Monday (0) and December
(12) twelve units from January (1). Sine/cosine pairs place them on a circle, so adjacent periods are
adjacent in feature space. The same argument applies to day-of-year across the year boundary.

---

## 12. Building the training set

### 12.1 The three steps

```text
read_raw()             parse dates, sort by (store_id, sku_id, date)
add_stockout_target()  produce demand_target, sample_weight, stockout_imputed_amount
build_causal_features()  lags, rollings, calendar, promo, price, cross-series, identity
```

The identical three run at inference. That is the mechanism preventing train/serve skew: the two
paths are the same code, so they cannot drift apart.

Output: **1,100,000 rows × 71 columns**, of which **43 numeric + 9 categorical = 52 features** reach
the tree models.

### 12.2 How many series, and how the panel is shaped

The panel stays **long**, not split per series. One row per (store, SKU, date), with identity carried
as features. A single global model therefore sees **all 1,005 series at once** (1,004 forecastable),
sharing weekly and promotional structure across them while conditioning on identity for scale.

Every group-wise calculation is scoped by `groupby(["store_id", "sku_id"])`, so no series can leak
into another's lags. This is the single most important implementation detail in feature construction.

### 12.3 Split sizes

| Split | Dates | Rows | Series | Days |
|---|---|---|---|---|
| Train | 2021-01-01 → 2023-12-03 | **1,071,888** | 1,005 | 1,067 |
| Validation | 2023-12-04 → 2023-12-17 | **14,056** | 1,004 | 14 |
| Test | 2023-12-18 → 2023-12-31 | **14,056** | 1,004 | 14 |

14,056 = 1,004 active series × 14 days. Training rows drop to **1,043,748** after removing rows with
no 28-day warm-up lag (the first 28 days of each series).

### 12.4 The 52 model features and their ranges

Measured on the full feature table. Nulls are warm-up rows at the start of each series, dropped
before training.

| Feature | Mean | Median | Std | Min | Max | Nulls |
|---|---|---|---|---|---|---|
| demand_lag_1 | 59.19 | 49 | 45.00 | 0 | 704 | 1,005 |
| demand_lag_7 | 59.17 | 49 | 44.98 | 0 | 704 | 7,035 |
| demand_lag_14 | 59.16 | 49 | 44.95 | 0 | 704 | 14,070 |
| demand_lag_28 | 59.12 | 49 | 44.91 | 0 | 704 | 28,140 |
| demand_roll_mean_7 | 59.18 | 54.14 | 36.52 | 2.86 | 344.00 | 3,015 |
| demand_roll_std_7 | 22.46 | 18.23 | 17.35 | 0.38 | 234.52 | 3,015 |
| demand_roll_max_7 | 93.51 | 83 | 62.79 | 4 | 704 | 3,015 |
| demand_roll_mean_14 | 59.17 | 54.86 | 35.70 | 3.43 | 273.29 | 3,015 |
| demand_roll_std_14 | 23.16 | 19.85 | 16.43 | 0.45 | 189.60 | 3,015 |
| demand_roll_max_14 | 104.76 | 94 | 69.89 | 6 | 704 | 3,015 |
| demand_roll_mean_28 | 59.15 | 55.18 | 35.21 | 4.21 | 272.33 | 3,015 |
| demand_roll_std_28 | 23.57 | 20.77 | 15.98 | 0.45 | 182.83 | 3,015 |
| demand_roll_max_28 | 115.46 | 103 | 76.95 | 7 | 704 | 3,015 |
| stockout_lag_1 / _7 / _14 | 0.030 | 0 | 0.171 | 0 | 1 | 1,005 / 7,035 / 14,070 |
| stockout_rate_28 | 0.030 | 0.036 | 0.033 | 0 | 0.375 | 7,035 |
| stock_on_hand_lag_1 | 299.48 | 300 | 80.07 | 0 | 698 | 1,005 |
| stock_on_hand_mean_7 | 299.47 | 299.57 | 30.35 | 145.43 | 470.86 | 3,015 |
| list_price_lag_1 | 7.71 | 7.38 | 4.25 | 1.08 | 14.80 | 1,005 |
| discount_pct_lag_1 | 0.015 | 0 | 0.055 | 0 | 0.30 | 1,005 |
| list_price_mean_28 | 7.71 | 7.38 | 4.25 | 1.08 | 14.80 | 7,035 |
| discount_pct_mean_28 | 0.015 | 0 | 0.021 | 0 | 0.129 | 7,035 |
| promo_prev_1 | 0.080 | 0 | 0.272 | 0 | 1 | 1,005 |
| promo_rate_28 | 0.080 | 0 | 0.110 | 0 | 0.600 | 7,035 |
| store_demand_mean_lag_1 | 59.19 | 59.86 | 11.71 | 25.86 | 95.36 | 1,005 |
| sku_demand_mean_lag_1 | 59.19 | 54.75 | 37.40 | 4.27 | 402.11 | 1,005 |
| series_age_days | 546.87 | 547 | 316.09 | 0 | 1,094 | 0 |
| promo_flag (known future) | 0.080 | 0 | 0.272 | 0 | 1 | 0 |
| is_weekend / is_holiday | 0.287 / 0.014 | 0 | — | 0 | 1 | 0 |
| dow_sin / dow_cos | ~0 | — | 0.707 | −1 | 1 | 0 |
| month_sin / month_cos | ~0 | — | 0.707 | −1 | 1 | 0 |
| doy_sin / doy_cos | ~0 | — | 0.707 | −1 | 1 | 0 |
| year, month, day, weekofyear, weekday | calendar integers | | | | | 0 |
| lead_time_days | 6.50 | 6 | 2.01 | 1 | 17 | 0 |

**Categoricals (9):** store_id, sku_id, country, city, channel, category, subcategory, brand,
store_sku_id.

**Deliberately excluded:** `gross_sales` and `net_sales` (both are units × price, so they hand the
model its own target); same-day `stock_out_flag` and `stock_on_hand` (outcomes, not inputs);
`purchase_cost` and `margin_pct` (kept for the business-cost proxy only); `sku_name`, `supplier_id`,
`latitude`, `longitude`; and — under the current contract — same-day `list_price`, `discount_pct`,
`temperature`, `rain_mm`.

---

## 13. Stockouts and sample weighting — what actually ran

### 13.1 The problem

On a stockout day, `units_sold` records what was *available to sell*, not what customers wanted.
Training on it directly teaches the model demand collapsed on exactly the days it may have spiked.

### 13.2 The three strategies

`add_stockout_target()` always emits the same three columns, so nothing downstream branches:

| Mode | demand_target on a stockout day | sample_weight |
|---|---|---|
| `none` | observed units_sold, unchanged | 1.0 |
| `rolling_median` | max(observed, prior 56-day non-stockout rolling median) | 0.5 |
| `percentage` | observed × 1.50 | 0.5 |

Every baseline used for imputation is `shift(1)`-ed before rolling, so a row's target can never
estimate itself.

### 13.3 What was actually configured — stated plainly

**Both production runs used `none`.** Measured on the real data:

```text
stockout_target_mode = none
sample_weight value counts : {1.0: 1,100,000}
demand_target == units_sold : True for every row
stockout_imputed_amount ≠ 0 : 0 rows
```

So **the down-weighting never activated**. Every row trained at weight 1.0, including the 33,114
censored ones. The machinery is built, wired through all three model families and leakage-tested —
it simply was not switched on.

Had `rolling_median` been enabled, the effect would have been:

```text
sample_weight counts      : {1.0: 1,066,886, 0.5: 33,114}
rows whose target changed : 33,084
mean uplift on those rows : +46.38 units
```

That is a substantial intervention — a mean uplift of 46 units on 3% of rows — which is precisely
why it should be measured as an experiment rather than assumed. It remains the cheapest open item.

### 13.4 How the weight reaches each model family

| Family | Mechanism |
|---|---|
| Tree models | `pipeline.fit(..., model__sample_weight=train["sample_weight"])` — passed straight to LightGBM/XGBoost/CatBoost |
| Torch models | Weighted L1: `(abs(pred − y) * weight).sum() / weight.sum()` — per-timestep weights inside the loss |
| Darts | `model.fit(..., sample_weight=weight_series)` — one weight TimeSeries per target series |

---

## 14. Hyperparameter search spaces

Optuna TPE (`multivariate=True`), seeded from `project.random_seed = 42`. Every trial is scored on a
**genuine recursive 14-day validation forecast**, not a row-wise score over pre-computed lags — so a
trial is evaluated exactly the way the model will be deployed.

### 14.1 Tree models — 25 trials each in run 2

| Model | Parameter | Range | Scale | Chosen |
|---|---|---|---|---|
| **LightGBM** | n_estimators | 300 – 1,200 | int | 834 |
| | learning_rate | 0.01 – 0.15 | log | 0.0120 |
| | num_leaves | 31 – 255 | int | 158 |
| | max_depth | 5 – 14 | int | 9 |
| | min_child_samples | 10 – 100 | int | 56 |
| | subsample | 0.7 – 1.0 | float | 0.723 |
| | colsample_bytree | 0.7 – 1.0 | float | 0.737 |
| | reg_lambda | 1e-3 – 10 | log | 1.264 |
| **XGBoost** | n_estimators | 300 – 1,200 | int | 1,094 |
| | learning_rate | 0.01 – 0.15 | log | 0.0140 |
| | max_depth | 4 – 12 | int | 8 |
| | min_child_weight | 1 – 20 | log | 5.92 |
| | subsample | 0.7 – 1.0 | float | 0.990 |
| | colsample_bytree | 0.7 – 1.0 | float | 0.707 |
| | reg_lambda | 1e-3 – 20 | log | 1.756 |
| **CatBoost** | iterations | 300 – 1,200 | int | 1,007 |
| | learning_rate | 0.01 – 0.15 | log | 0.0172 |
| | depth | 5 – 10 | int | 8 |
| | l2_leaf_reg | 1 – 20 | log | 5.90 |
| | random_strength | 0 – 2 | float | 0.093 |

All three converged on **low learning rates with many trees** — the signature of a noisy target with
modest signal, where the optimiser buys variance reduction rather than fitting sharp structure.

### 14.2 Sequence models

Shared across all four: **lookback 14 – 112 in steps of 14** (whole weeks, at least one horizon),
learning rate 1e-4 – 5e-3 (log), dropout 0 – 0.3, batch size {128, 256, 512}.

| Model | Additional parameters | Chosen |
|---|---|---|
| **LSTM** | hidden_size {64,128,256}; num_layers 1–3; embedding_dim {8,16,32}; weight_decay 1e-6 – 1e-3 log | lookback **42**, hidden 128, layers 3, emb 16, lr 0.00412, dropout 0.220 |
| **Transformer** | d_model {64,128,256}; heads {2,4,8}; layers 2–4; embedding_dim {8,16,32}; weight_decay 1e-6 – 1e-3 log | lookback **70**, d_model 256, heads 8, layers 2, emb 8, lr 0.00054, dropout 0.087 |
| **TiDE** | hidden_size {64,128,256} | lookback 56, hidden 128 (defaults; its tuning study failed on the Darts bug and it was trained with sensible values at a capped 12 epochs) |

**Every winning lookback is a multiple of seven** — 42, 70, 56 — which was not forced beyond the
step size and is independent confirmation that the weekly structure the EDA found is real.

---

## 15. Model architectures

### 15.1 Global LSTM encoder-decoder (written from scratch)

```text
past_x  [B, 42, 14]  ─► encoder LSTM (3 layers, hidden 128) ─► (h, c)
static  [B, 6]       ─► 6 embedding tables            ─► static_vec [B, 6×16 = 96]

for each horizon step h in 0..13:
    decoder input = concat( future_x[:, h, :]  (7) ,  static_vec  (96) ,  prev_y  (1) )  = 104
    (h, c), out   = decoder LSTM(decoder input, (h, c))
    ŷ_h           = head(out)          # Linear(128→64) → ReLU → Linear(64→1)
outputs [B, 14]
```

The first autoregressive token is the last normalised demand value, `past_x[:, -1, 0:1]`. During
training, scheduled **teacher forcing at probability 0.2** replaces the previous prediction with the
true value; at inference it is **always 0.0**, so the decoder never sees a label it would not have.

The same weights serve all 1,004 series — identity enters only through the embeddings.

### 15.2 Global temporal Transformer (written from scratch)

```text
past_x   [B, 70, 14] ─► Linear(14→256) ─► +positional ─► TransformerEncoder(2 layers, 8 heads) ─► memory [B, 70, 256]
future_x [B, 14, 7]  ─► Linear(7→256)  ─┐
static   [B, 6] ─► emb ─► Linear(48→256)─┴─► queries [B, 14, 256]
                     ─► TransformerDecoder(1 layer, cross-attends memory) ─► [B, 14, 256]
                     ─► Linear(256→1) ─► [B, 14]
```

Feed-forward width is 4 × d_model, `norm_first=True` (pre-norm, more stable), and the decoder uses
`max(1, num_layers − 1)` layers.

The key difference from the LSTM: **all 14 days are emitted at once**. Each horizon day forms its own
query from that day's known-future covariates plus static context, so day 14 is conditioned on day
14's promotion rather than on a chain of 13 previous predictions. No recursive error accumulation.

### 15.3 Tree models

Global models over the long panel. One sklearn `Pipeline`:

```text
ColumnTransformer
├── numeric      SimpleImputer(median)                          → 43 columns
└── categorical  SimpleImputer(most_frequent) → OrdinalEncoder  →  9 columns
                 (handle_unknown="use_encoded_value", unknown_value=-1)
└── estimator    LightGBM (regression_l1) | XGBoost (reg:squarederror) | CatBoost (MAE)
```

### 15.4 TiDE (Darts)

Encoder-decoder MLP with static covariates enabled, `input_chunk_length=56`,
`output_chunk_length=14`, hidden 128, decoder output dim 32, 2 encoder and 2 decoder layers.

---

## 16. How future covariates reach each model

This is where the leakage contract is enforced, and each family does it differently.

| Family | Mechanism |
|---|---|
| **Trees** | The future row is appended to history and `build_causal_features` runs over the combined frame. Known-future columns keep their supplied values; everything unknown is set to NaN and then blocked from the feature list entirely |
| **Torch** | A separate `future_x [B, 14, F]` tensor, distinct from `past_x`. The decoder receives it per step (LSTM) or turns it into per-day queries (Transformer). Future targets are never in this tensor |
| **Darts** | `future_covariates` TimeSeries extend past the training range; `past_covariates` stop at the origin. Darts enforces the distinction internally |

Under the current contract the future channels are:

```text
is_holiday, is_weekend, dow_sin, dow_cos, month_sin, month_cos, promo_flag     (7 channels)
```

Past channels carry everything observed, including the target itself:

```text
demand_target, promo_flag, list_price, discount_pct, temperature, rain_mm,
stock_out_flag, stock_on_hand, is_holiday, is_weekend, dow_sin, dow_cos,
month_sin, month_cos                                                          (14 channels)
```

Flipping `price_known_future` to `true` would move `list_price` and `discount_pct` into the future
set for all three families *and* into the API's required request fields, from one config line.

---

## 17. Categorical encoding by family

| Family | Method | Unseen values | Why |
|---|---|---|---|
| **Trees** | Ordinal encoding inside the pipeline | Mapped to **−1** | Trees split on thresholds; one-hot on 1,005 store-SKU keys would explode the width for no gain. Ordinal also keeps one identical inference contract across all three libraries |
| **Torch** | Learned embeddings, one table per field | Index **0** reserved for unknown | Lets the model learn similarity between SKUs rather than treating them as arbitrary integers |
| **Darts** | Integer static covariates via `make_static_maps()` | Mapped to **−1** | Consumed because both models are built with `use_static_covariates=True` |

Embedding tables actually built (cardinality = distinct values + 1 for unknown):

```text
store_id 14 · sku_id 103 · channel 5 · category 6 · subcategory 18 · brand 7
```

Each is `embedding_dim` wide (16 for the LSTM, 8 for the Transformer), concatenated into a static
vector of 96 and 48 respectively.

Maps are fitted on **training rows only** and frozen into the checkpoint, so inference reproduces the
exact encoding. A genuinely new SKU therefore degrades to a learned "unknown" vector rather than
crashing — the cold-start path.

Note the deliberate trade-off: CatBoost's native ordered target statistics are *not* used. That
probably costs some CatBoost accuracy, but it buys one identical inference contract across the three
tree libraries. Testing native handling is on the future-work list.

---

## 18. Metrics — computed on a real example

All metrics are pooled over every row in the window: 1,004 series × 14 days = **14,056 rows**. Not
computed per series and averaged.

### 18.1 A worked example

Take three rows from the test window:

| Row | Actual | Predicted | Error | Abs error |
|---|---|---|---|---|
| A | 100 | 90 | −10 | 10 |
| B | 50 | 60 | +10 | 10 |
| C | 10 | 5 | −5 | 5 |
| **Sum** | **160** | **155** | **−5** | **25** |

| Metric | Formula | This example | Meaning |
|---|---|---|---|
| **WAPE** | Σ\|error\| / Σ actual | 25 / 160 = **15.6%** | "Off by 15.6% of the units actually sold." Volume-weighted: row A's miss counts as much as row C's despite being 10× the demand |
| **MAPE** | mean(\|error\| / actual) | (0.10 + 0.20 + 0.50)/3 = **26.7%** | Row C — 5 units on a base of 10 — contributes 50% and drags the average up. This is why MAPE is unstable here |
| **MAE** | mean(\|error\|) | 25 / 3 = **8.33 units** | "We miss by 8.3 units per store-SKU-day." Translates directly into cases |
| **RMSE** | √mean(error²) | √(225/3) = **8.66** | Penalises large misses; the promo-spike detector |
| **Bias** | Σ error / Σ actual | −5 / 160 = **−3.1%** | Negative means systematic under-forecast → understock and lost sales. Positive → overstock and markdown |

Note WAPE (15.6%) and MAPE (26.7%) differ by 11 points on the same three rows, entirely because of
row C's small denominator. On the real data the gap is similar: WAPE ~0.27 against MAPE ~0.53.

### 18.2 Why these, commercially

- **WAPE** drives selection. A unit of error on a 500/day SKU genuinely costs more than one on a
  10/day SKU, and WAPE weights it that way. It is also always defined.
- **MAPE** is reported because stakeholders ask for it, never used to select. 0.28% of rows are zero
  (excluded, with coverage reported) and 1.46% are under five units.
- **MAE** is the units-per-day figure a planner can act on.
- **RMSE** surfaces promo-spike failures that WAPE smooths over.
- **Bias** is the inventory-critical one, and the metric that changed the production recommendation.

### 18.3 Business-cost proxy

`business_proxy()` computes under-forecast units × unit margin against over-forecast units ×
purchase cost. It is a directional comparison of the asymmetry, **not** an inventory simulation —
lead time, safety stock and replenishment policy are not modelled.

---

## 19. Results and the model chosen

### 19.1 Every model, plus the baselines

| Rank | Model | Train WAPE | Val WAPE | Test WAPE | Test MAPE | Test MAE | Test bias |
|---|---|---|---|---|---|---|---|
| 1 | Transformer | — | **0.2654** | 0.2747 | 0.5728 | 17.14 | **+7.32%** |
| 2 | LSTM | — | 0.2662 | **0.2673** | **0.5134** | **16.68** | −1.10% |
| 3 | CatBoost | 0.2664 | 0.2719 | 0.2683 | 0.5283 | 16.75 | +0.99% |
| 4 | XGBoost | 0.2584 | 0.2721 | 0.2714 | 0.5199 | 16.94 | −1.27% |
| 5 | LightGBM | 0.2637 | 0.2722 | 0.2704 | 0.5304 | 16.88 | **+0.86%** |
| 6 | TiDE | — | 0.2762 | 0.2694 | 0.5263 | 16.81 | −0.00% |
| — | *SARIMA baseline* | — | *0.3051* | *0.3029* | *0.5530* | *18.91* | *−1.70%* |
| — | *28-day moving average* | — | *0.3172* | *0.3118* | *0.5639* | *19.46* | *−1.55%* |
| — | *Seasonal naive* | — | *0.4307* | *0.4240* | *0.6911* | *26.46* | *−0.25%* |
| — | *Last value flat* | — | *0.4817* | *0.4938* | *0.8436* | *30.82* | *+16.89%* |

Train WAPE is blank for the sequence models: they early-stop on validation and never score the
training set.

### 19.2 The comparison, and the decision

**Promoted by the pipeline: Transformer** (lowest validation WAPE). **The model I would deploy:
CatBoost.** Four reasons:

1. **The accuracy gap is inside the noise.** All six span 0.2654–0.2762 on validation, ~4% relative.
   Transformer to CatBoost is 0.0065 WAPE.
2. **Validation has mispredicted test three times.** Run 1: Transformer won validation, CatBoost best
   on test. Run 2: Transformer won validation, LSTM best on test — with the Transformer *worst* of
   the five non-Darts models. One 14-day December origin cannot separate models this close.
3. **Bias settles it.** The Transformer over-forecasts test by **+7.32%** — roughly 64,000 phantom
   units. CatBoost is +0.99%, LightGBM +0.86%. For an order decision that dwarfs 0.006 of WAPE.
4. **Operational cost.** CatBoost trains in ~25 s against 10–30 min, tunes at ~40 s/trial against
   7–30 min, scores all 1,004 series on CPU in ~22 s, and gives SHAP explanations free.

CatBoost is also the only model top-three on **both** validation and test *and* byte-identical
between the two runs. LightGBM is the runner-up and the least biased model in the study.

Against SARIMA the models improve by **9.3% (Transformer) to 11.8% (LSTM)**. Real, but a global model
beating a per-series statistical model by about a tenth — not by half.

### 19.3 Feature importance, and what it confirms

| Model | Method | Top feature | Share | Promotion flag |
|---|---|---|---|---|
| LightGBM | native gain | demand_roll_mean_28 | 70.3% | 3.1% |
| XGBoost | native gain | demand_roll_mean_28 | 29.7% | **25.0%** |
| CatBoost | native gain | demand_roll_mean_28 | 35.0% | **34.2%** |
| LSTM | permutation on val WAPE | sku_id (static) | 46.1% | **17.2%** |
| Transformer | permutation on val WAPE | demand_target (past window) | 64.6% | **13.9%** |

Recent demand level dominates everywhere; the promotion flag is second almost universally, which
vindicates treating the promo calendar as known-future. Weekday, weekend and the cyclical
day-of-week term all appear in the tree top-tens — the weekly seasonality from the EDA reappearing
after training. Individual demand lags appear in **no** top-ten: the rolling means absorb them.

---

## 20. Inference: how a forecast is actually produced

### 20.1 Where models are loaded from — stated precisely

**Models are loaded from the local filesystem, not downloaded from the MLflow registry.**

```text
configs/model_registry_v2.yaml  →  artifact_path: artifacts_v2/transformer/model.pt
                                        ↓
                          inference_router.run_selected()
                                        ↓
                   dispatch on family → ml | dl | darts adapter
```

MLflow is the **experiment record**; the YAML registry is the **serving contract**. Every model is
also logged to MLflow as a run artifact, so a run could be pulled down, but the serving path does not
do that today. That is a deliberate simplification for a local/Docker deployment and a genuine gap
for a real one — MLflow Model Registry aliases (`champion`, `candidate`) with approval gates would be
the production answer.

### 20.2 How promotion is tracked

Two registries, deliberately separate:

| File | Role |
|---|---|
| `model_registry_v2.yaml` | **Serving contract** — one model, what the API loads |
| `model_registry_all_v2.yaml` | **Audit record** — all six, ranked, with metrics, tuned params, artifact paths and predicted-vs-actual totals |

`select_model.py` re-reads the comparison table and **refuses to promote anything that is not the
current validation champion**, which prevents promoting a model because its *test* number looked
good. The registry records the metric that won and its value:

```yaml
selected_model:
  name: transformer
  family: dl
  artifact_path: artifacts_v2/transformer/model.pt
  selection_metric: validation_wape
  selection_value: 0.2654202684461504
  validation_metrics: {wape: 0.2654, mape: 0.5336, mae: 16.48, ...}
  test_metrics:       {wape: 0.2747, mape: 0.5728, mae: 17.14, ...}
```

In MLflow, runs are tagged `stage` = `tuning_parent` | `tuning` | `final` | `baseline`, so
`tags.stage = 'final'` isolates the comparable models and `tags.stage = 'baseline'` the references.

### 20.3 What we feed the model for genuinely future dates

This is the question that matters most, because for 2024-01-01 onwards **we have no historical
feature values at all** — no lags, no rolling means, nothing. Here is exactly how it works.

**Step 1 — the caller supplies only what is genuinely known.** `make_future_template.py` generates
it. For one series, the first three rows look like this:

```text
 store_id  sku_id       date  is_holiday  is_weekend  promo_flag
STORE0001 SKU0001 2024-01-01           0           0           0
STORE0001 SKU0001 2024-01-02           0           0           0
STORE0001 SKU0001 2024-01-03           0           0           0
```

**Six columns.** No demand, no lags, no rolling means, no price, no inventory. That is the entire
input for the future. 14,056 such rows cover every active series for 14 days.

**Step 2 — the history is what supplies the lags.** The last observed days for this series:

```text
      date  units_sold  promo_flag  list_price  stock_out_flag
2023-12-24         107           0        6.24               0
2023-12-25         110           0        6.24               0
2023-12-26          38           0        6.24               1   <- stockout day
2023-12-27          88           0        6.24               0
2023-12-28         162           1        6.24               0   <- promotion day
2023-12-29         126           0        6.24               0
2023-12-30         120           0        6.24               0
2023-12-31         106           0        6.24               0
```

**Step 3 — day 1 features are computed by appending the future row to history and rebuilding.** The
future row arrives with its target blanked; the feature builder then derives every historical
feature from the real history behind it:

```text
forecast date 2024-01-01
   demand_lag_1        = 106.0000   ← actual demand on 2023-12-31
   demand_lag_7        = 110.0000   ← actual demand on 2023-12-25
   demand_lag_14       = 147.0000
   demand_lag_28       = 124.0000
   demand_roll_mean_7  = 107.1429
   demand_roll_mean_28 = 111.3571
   promo_prev_1        = 0.0000
   promo_rate_28       = 0.0357
   stockout_lag_1      = 0.0000
   series_age_days     = 1095.0000
```

So the caller never supplies lags — **they are derived**, which is exactly why the same feature code
must run in training and inference.

**Step 4 — predict day 1, then feed the prediction back as history.**

```text
   day 1 prediction = 101.0195
```

**Step 5 — day 2's lag comes from the prediction, not the actual.**

```text
   day 2 (2024-01-02) demand_lag_1 = 101.0195
   day 1 prediction was            101.0195   ← identical, so the chain is genuine
   day 2 demand_lag_7             = 38.0000   ← still a real observation (2023-12-26, the stockout day)
```

This is the whole recursion in one trace. Day 2's one-day lag is day 1's **prediction**; its
seven-day lag is still a real observation because that date is inside history. As the horizon
advances, progressively more of the lag window is model output rather than data — by day 14, the
one-day lag is thirteen predictions deep.

Rows are processed **one forecast date at a time**, in order, with history growing by one synthetic
row per series per step. They are not passed as a single batch, because day 2 cannot be built until
day 1 exists.

### 20.4 How leakage is prevented in this loop

Three mechanisms, and the trace above demonstrates all three:

1. **The target is blanked on arrival.** Any of `units_sold`, `demand_target`, `gross_sales`,
   `net_sales`, `stock_out_flag`, `stock_on_hand` present on a future row is overwritten with NaN
   before feature building.
2. **Unknown covariates are blanked by contract.** With price and weather set to unknown, those
   columns are nulled and excluded from the feature list, so a caller cannot smuggle them in.
3. **Only the prediction is appended.** The synthetic history row carries the *predicted* value as
   both `units_sold` and `demand_target`. The actual is never consulted — it does not exist yet for a
   real future forecast, and for validation and test it is deliberately withheld.

Verified empirically rather than asserted: corrupting future actuals by ×1000 leaves every forecast
**bit-identical**, while flipping the promotion flag (declared known) moves predictions by 151 units.
See §6.

### 20.5 Sequence models: same contract, different mechanics

No recursion at all. One window per series is assembled and all 14 days emitted at once:

```text
lookback from checkpoint = 42, horizon = 14
past_x   [B, 42, 14]  ← 42 days of real history, 14 channels
future_x [B, 14,  7]  ← 14 days of known-future covariates, 7 channels
static   [B, 6]       ← store, SKU, channel, category, subcategory, brand
target scaling: mean = 59.1547, std = 44.9522   (fitted on TRAINING rows only)
```

The lookback comes from the **checkpoint**, not global config — it is tuned per model, so the artifact
is authoritative. Predictions are de-scaled with the stored mean and std and clipped at zero.

### 20.6 The output

```text
date,store_id,sku_id,prediction,model,family
2024-01-01,STORE0001,SKU0001,100.10,catboost,ml
```

14,056 rows per model (1,004 series × 14 days); 84,336 rows across all six.
`store_sku_forecast_and_metrics.csv` joins each series' forecast to how accurately that same series
was predicted on the held-out window, sorted worst-first — the file to hand a planner.

---

## 21. File flow — what runs when, and what it holds

Cross-reference to §4, which describes each module in detail. This is the execution order with
inputs and outputs.

| # | Stage | File | Reads | Writes |
|---|---|---|---|---|
| 1 | Features | `prepare_dataset.py` | raw CSV, config | `data/processed/features.parquet` |
| 2 | Tune trees | `tune_bayesian.py` | raw, config | `tuning/<m>_best_params.json`, `_trials.csv`, `_study.db`, per-trial models; MLflow nested runs |
| 3 | Tune sequences | `tune_dl.py` | raw, config | same, plus tuned **lookback** |
| 4 | Train trees | `train_ml.py` | raw, best params | `<m>/model.joblib`, `metrics.json`, prediction CSVs; MLflow run |
| 5 | Train torch | `train_dl.py` | raw, best params | `<m>/model.pt` (state dict + metadata + config), metrics, predictions |
| 6 | Train Darts | `train_darts.py` | raw, best params | `<m>/model.pt` + `.ckpt` + `metadata.joblib`, metrics, predictions |
| 7 | Baselines | `baseline.py` | raw, config | `baseline_metrics.csv`, predictions; 4 MLflow runs |
| 8 | Compare | `evaluate_compare.py` | all `metrics.json` | `model_comparison.csv` |
| 9 | Report | `report_best_models.py` | metrics + test predictions | best-per-family, per-series and per-horizon breakdowns |
| 10 | Importance | `feature_importance.py` | fitted models | `feature_importance.csv` |
| 11 | Promote | `select_model.py` | comparison, metrics | `model_registry.yaml` |
| 12 | Register all | `register_all_models.py` | all metrics + artifacts | `model_registry_all.yaml` |
| 13 | Future inputs | `make_future_template.py` | raw history, config | `future_covariates.csv`, `forecast_request.json` |
| 14 | Forecast | `run_final_inference.py` | registry/best models, raw | `forecast_store_sku.csv`, `store_sku_forecast_and_metrics.csv` |
| 15 | Serve | `api.py` → `inference_router.py` | registry, history, request | JSON forecast response |

**Supporting modules** (not pipeline stages): `config.py` loads YAML and applies tuned overrides;
`data.py` is the canonical load path; `features.py` owns every feature definition and the blocked-column
policy; `stockout.py` builds the target and weights; `splits.py` computes boundaries; `metrics.py`
holds the metric maths; `tracking.py` wires MLflow and loads `.env`; `logging_utils.py` gives each
step its own rotating log; `hierarchy.py` does bottom-up and middle-out; `drift.py` monitors PSI/KS.

**Report writing:** `scripts/md_to_docx.py` renders the Markdown report to Word;
`scripts/append_docx_section.py` and `scripts/replace_docx_section.py` add or swap a section in an
existing document so manual Word edits survive.

**Orchestration:** `scripts/run_v1.sh` runs the whole chain with per-family trial counts, a
pass/fail trail, and skip-if-exists resume. `scripts/run_all.sh` is the generic equivalent.
