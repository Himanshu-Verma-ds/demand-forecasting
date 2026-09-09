# FMCG Multi-Store Demand Forecasting — Final Report

**Objective.** Forecast daily `units_sold` for every active `(store_id, sku_id)` pair, 14 days ahead,
as a production-shaped system: leakage-safe, tracked, reproducible and servable.

**Data.** 1,100,000 rows · 2021-01-01 → 2023-12-31 (1,095 days) · 13 stores · 102 SKUs ·
**1,005 Store-SKU series** · 5 categories · 17 subcategories · 4 channels.

**Outcome.** Six model families trained across two full experiment runs (163 MLflow runs on DagsHub).
All six land within **4% relative** of each other. The promoted champion is the global Transformer
(validation WAPE **0.2654**); the model I recommend shipping is **CatBoost**, for reasons in §11.

> This report is also available as **`final_report.docx`** (same content, regenerate with
> `python scripts/md_to_docx.py final_conclusion/final_report.md final_conclusion/final_report.docx`).

| Where | What |
|---|---|
| `final_conclusion/final_report.docx` | this report in Word format |
| `final_conclusion/inference_bundle/` | models + data to regenerate the forecast |
| `reports/v1/`, `reports/v2/` | metrics, per-series/per-horizon breakdowns, forecasts |
| `logs/` | per-step rotating logs, plus `logs/orchestration/` |
| [MLflow v1](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/2) · [MLflow v2](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3) | every tuning trial and final run |
| `architecture_flow.md` | file-by-file design reference and Q&A |

---

## 1. EDA and the decisions it drove

Full analysis in `notebooks/01_eda.ipynb`. Every section there ends in a *Decision impact* note — the
EDA was run to make architecture choices, not to produce charts.

### 1.1 Data quality and shape

- **No missing values**, one row per `(store, SKU, date)` — verified, not assumed. A clean daily grain
  means no imputation strategy is needed before lag construction.
- `units_sold` is **right-skewed**: median ≈ 49, mean ≈ 59, max 704.
- `stock_on_hand` has zeros; `lead_time_days` reaches 17 vs a mean of 6; `margin_pct` goes negative.
- **`stock_out_flag` is not the same as zero inventory** — almost all stockout rows still show
  positive `stock_on_hand`, so the two fields describe different concepts or timestamps. This matters:
  it means stockouts cannot be re-derived from inventory and the flag must be trusted as given.

→ *Decision:* right skew makes **WAPE/MAE** better decision metrics than RMSE alone, and argues for
an L1 objective (LightGBM `regression_l1`, CatBoost `MAE`).

### 1.2 Hierarchy integrity

- Every `sku_id` maps to exactly one name/category/subcategory/brand; every `store_id` to one
  country/city/channel. **No effective-dated master-data problem.**
- The median series has the **full 1,095 days**; ≥75% of series are complete.
- **1,004 of 1,005 series reach the final date.** The exception is `STORE0013 / SKU0073`, which runs
  continuously to **2022-09-12** and then stops — and exactly accounts for STORE0013's 475-row shortfall.

→ *Decision:* dense bottom-level history makes **global bottom-level forecasting** viable, and
**bottom-up** the natural reconciliation baseline. The one dead series is excluded from the active
forecast set. That single row later caused two real production bugs (§5.4) — the EDA finding was the
thing that made them diagnosable in minutes rather than hours.

### 1.3 Seasonality — and why the lags are what they are

This is the single most load-bearing EDA result for feature design.

- Aggregate demand is **higher on Saturdays and Sundays**.
- Monthly peaks in **Jun–Aug** and **Oct–Dec**, and the notebook confirms *the pattern repeats in all
  three years* — it is seasonality, not a one-off.
- **Autocorrelation of aggregate daily demand peaks at lags 7, 14, 21 and 28** — clean weekly recurrence.
- The 12 highest-volume series show the same annual shape with series-specific magnitude: shared
  seasonal structure, individual scale.

→ *Decision — the lag set follows directly from the autocorrelation peaks:*

| Feature | Why this value |
|---|---|
| `demand_lag_1` | yesterday — short-run level and momentum |
| `demand_lag_7` | **same weekday last week** — the dominant ACF peak |
| `demand_lag_14`, `demand_lag_28` | same weekday 2 and 4 weeks back — confirms the weekly signal is stable, and 28 spans a full monthly cycle |
| rolling mean/std/max over **7 / 14 / 28** | level, volatility and recent peak over exactly one, two and four weekly cycles |
| `dow_sin/cos`, `month_sin/cos`, `doy_sin/cos` | cyclical encodings so Sunday→Monday and Dec→Jan are adjacent rather than maximally distant |

Nothing here is a default choice: 7/14/28 are the measured ACF peaks, and the cyclical terms exist
because raw `weekday` would tell the model Sunday (6) is six units away from Monday (0).

### 1.4 Promotions

- Promo rows show **63%–131% higher average demand** depending on category (Home Care and Beverages
  strongest), and demand rises **monotonically with discount depth**.
- The notebook is explicit that this is **descriptive, not causal** — promotions are scheduled, likely
  onto already-strong periods.

→ *Decision:* `promo_flag` is a **known-future covariate** (the promo calendar is committed ahead of
the origin), but no causal uplift is claimed. Promo days get their own error slice, because ~7% of
rows carrying double demand can hide inside a good aggregate WAPE. They do: the champion's validation
WAPE is 0.2755 on promo days vs 0.2639 off them, and MAE is **2× worse** (31.0 vs 15.4).

### 1.5 Stockouts — censored demand

Stockouts on ~3% of rows. On those days `units_sold` is *what was available to sell*, not demand.
Training on it teaches the model that demand was low exactly when it may have been high.

→ *Decision:* `stockout_target_mode` supports `none` / `rolling_median` / `percentage`, always
emitting `demand_target`, `sample_weight` and `stockout_imputed_amount` so no downstream code branches.
**v1 and v2 both ran `none`** — the imputation is a modelling assumption, and the honest v1 is to ship
the un-imputed baseline and treat correction as a measured experiment. It remains untested (§12).

### 1.6 Price and weather

→ *Decision:* same-day price/weather are only legal if genuinely known for the whole horizon.
We set `price_known_future: false` and `weather_known_future: false`, so the training contract matches
what a planner can actually supply at 08:00 on forecast day.

---

## 2. How the training data is built for Store × SKU forecasting

Three deterministic steps, identical in training and inference — which is what makes train/serve skew
impossible:

```python
raw  = read_raw(path, cfg)                                   # parse dates, sort by (store, sku, date)
df   = add_stockout_target(raw, cfg)                         # demand_target + sample_weight
feat = build_causal_features(df, cfg, "demand_target")       # lags, rollings, calendar, promo, price
```

Output: **1,100,000 rows × 72 columns**, of which ~52 numeric + 9 categorical reach the models.

**The panel is kept long, not split per series.** One row per `(store, SKU, date)`, with `store_id`,
`sku_id`, `store_sku_id`, `channel`, `category`, `subcategory`, `brand` as features. One **global**
model therefore learns from all 1,005 series at once — sharing weekly/seasonal/promo structure while
conditioning on identity for scale. Per the EDA (§1.2), that beats ~1,000 local models operationally
and handles cold-start, which a local model cannot do at all.

Every group-wise operation is computed **within** `(store_id, sku_id)` via `groupby(series_cols)`, so
no SKU's history ever leaks into another's lag.

Feature groups reaching the models:

| Group | Features |
|---|---|
| Demand history | `demand_lag_{1,7,14,28}`; shifted rolling mean/std/max over 7/14/28 |
| Calendar (known future) | year, month, day, weekday, weekofyear, `is_weekend`, `is_holiday`, 6 cyclical terms |
| Promotions | `promo_flag` (known future), `promo_prev_1`, `promo_rate_28` |
| Price history | `list_price_lag_1`, `discount_pct_lag_1`, and their 28-day means |
| Inventory history | `stockout_lag_{1,7,14}`, `stockout_rate_28`, `stock_on_hand_lag_1`, `stock_on_hand_mean_7` |
| Cross-series | `store_demand_mean_lag_1`, `sku_demand_mean_lag_1` (day-lagged group means) |
| Identity | 9 categoricals + `series_age_days` |

For the sequence models the same information becomes tensors: `past_x [B, lookback, 14]`,
`future_x [B, 14, 6]`, `static_ids [B, 6]`, `y [B, 14]`.

---

## 3. Preventing data leakage

The central risk in demand forecasting is trivially easy to hit and produces beautiful, meaningless
metrics. Eleven controls, and — more importantly — **empirical proof they hold**.

### 3.1 The controls

1. `gross_sales` and `net_sales` **permanently blocked** — they are `units_sold × price`, i.e. the
   target in disguise.
2. **Every rolling target statistic is shifted before rolling**: `s.shift(1).rolling(28).mean()`,
   never `s.rolling(28).mean()`.
3. Same-day `stock_out_flag` and `stock_on_hand` are **never** features — they are outcomes. Only
   lagged versions are used.
4. The split is **chronological**, never random.
5. Tree validation/test use **recursive forecasting from a fixed origin**: day D+2's `lag_1` is day
   D+1's *prediction*, never its actual.
6. Sequence models see only the lookback window plus genuinely-known future covariates.
7. Scaling statistics and categorical maps are fitted on **training rows only**, then frozen into the
   checkpoint.
8. Bayesian tuning optimises **validation** WAPE; test is scored once.
9. DL early stopping uses validation; the refit then runs a **fixed** epoch count so test never
   influences stopping.
10. Inference and the API blank every column the `*_known_future` contract marks unavailable.
11. `tests/test_leakage.py` asserts (1)–(2) mechanically on every test run.

### 3.2 The recursion, concretely

```text
origin = 2023-12-03 (end of training)
D+1  2023-12-04  features from real history through 12-03      → ŷ = 47.2
D+2  2023-12-05  lag_1 = 47.2  (the prediction, not the actual) → ŷ = 51.8
...
D+14 2023-12-17
```

A common notebook shortcut computes `lag_1` on the full dataframe and *then* filters validation rows.
The Dec-5 row would then contain Dec-4's **actual** demand — unknowable at the Dec-3 origin. That
single mistake can halve reported error.

### 3.3 Proof, not assertion

Unit tests only cover the feature builder, so the end-to-end forecast paths were attacked directly:
corrupt information that is unknown at the origin and check the forecast does not move. A
bit-identical result means the information cannot have been used.

| Probe | Perturbation | Result |
|---|---|---|
| Tree — future labels | `units_sold × 1000 + 12345`, gross/net × 999 | **Δ = 0.0** |
| Tree — future inventory | flip `stock_out_flag`, `stock_on_hand → 0` | **Δ = 0.0** |
| Tree — unknown covariates | price × 5, discount 0.9, temp −50, rain 999 | **Δ = 0.0** |
| Tree — recursion | inspect D+2 `demand_lag_1` | = D+1 **prediction**, not actual |
| Tree — blocked columns | inspect the 52 model inputs | none present |
| DL — future labels + inventory | same corruption | **Δ = 0.0** |
| **Control:** promo (declared *known*) | flip `promo_flag` | **Δ = 151.3 / 115.8 units** — correctly *does* move |
| Stockout `rolling_median` | build targets from train-only vs full data | **0 / 21,340** training targets change |

The control row is why the zeros mean anything: it proves the models genuinely read their future
inputs rather than ignoring all of them.

---

## 4. Train / validation / test split

Computed backwards from the last date, so it updates automatically as data grows:

```text
train       2021-01-01 .. 2023-12-03     1,071,888 rows
validation  2023-12-04 .. 2023-12-17        14,056 rows   ← model selection + tuning objective
test        2023-12-18 .. 2023-12-31        14,056 rows   ← scored once
```

14,056 = **1,004 active series × 14 days**.

**Why 14 days each:** the deployment task is one 14-day batch forecast from a fixed origin. Selecting
on the same geometry means validation measures the deployed task, not a proxy.

**Why not k-fold:** random folds put future rows in training, and because features are lags and
rolling windows, neighbouring rows are near-duplicates — a random split leaves copies of every
validation row in training. It also measures the wrong task ("fill in a missing day given both
neighbours"). The correct analogue is rolling-origin backtesting (§12).

**Refit discipline:** after validation selects the configuration, a *fresh* model is refit on
train+validation and the test window is forecast from the val-end origin. Test is never used to
choose anything.

---

## 5. Experiments — every run

### 5.1 What was run

| | v1 | v2 |
|---|---|---|
| MLflow experiment | [`fmcg-demand-forecasting-v1`](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/2) (67 runs) | [`fmcg-demand-forecasting-v2`](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3) (96 runs) |
| Tree trials / model | 12 | **25** |
| Sequence trials / model | 2–3 | 2–3 |
| DL epochs / patience | 15 / 3 | **50 / 8** |
| Trial models logged | no | **yes (80 artifacts)** |
| Artifacts | `artifacts/` | `artifacts_v2/` |

Identical data, features, split and leakage contract — **only the search budget and schedule differ**,
so the comparison is clean.

### 5.2 Final standings (v2, all six models)

| rank | model | family | train_wape | **val_wape** | val_mape | test_wape | test_mape | test_mae |
|---|---|---|---|---|---|---|---|---|
| 1 | **transformer** | dl | – | **0.2654** | 0.5336 | 0.2747 | 0.5728 | 17.14 |
| 2 | lstm | dl | – | 0.2662 | 0.5340 | **0.2673** | **0.5134** | **16.68** |
| 3 | catboost | ml | 0.2664 | 0.2719 | 0.5461 | 0.2683 | 0.5283 | 16.75 |
| 4 | xgboost | ml | 0.2584 | 0.2721 | 0.5358 | 0.2714 | 0.5199 | 16.94 |
| 5 | lightgbm | ml | 0.2637 | 0.2722 | 0.5447 | 0.2704 | 0.5304 | 16.88 |
| 6 | tide | darts | – | 0.2762 | 0.5535 | 0.2694 | 0.5263 | 16.81 |

### 5.3 Winning hyperparameters, and why they look like this

```jsonc
// catboost   val 0.2719 — the recommendation
{"iterations": 1007, "learning_rate": 0.0172, "depth": 8, "l2_leaf_reg": 5.90, "random_strength": 0.093}

// lightgbm   val 0.2722
{"n_estimators": 834, "learning_rate": 0.0120, "num_leaves": 158, "max_depth": 9,
 "min_child_samples": 56, "subsample": 0.723, "colsample_bytree": 0.737, "reg_lambda": 1.264}

// xgboost    val 0.2721
{"n_estimators": 1094, "learning_rate": 0.0140, "max_depth": 8, "min_child_weight": 5.92,
 "subsample": 0.990, "colsample_bytree": 0.707, "reg_lambda": 1.756}

// transformer val 0.2654 — the promoted champion
{"lookback": 70, "learning_rate": 0.00054, "dropout": 0.087, "batch_size": 128,
 "transformer_d_model": 256, "transformer_heads": 8, "transformer_layers": 2, "embedding_dim": 8}

// lstm        val 0.2662 — best on test
{"lookback": 42, "learning_rate": 0.00412, "dropout": 0.220, "batch_size": 128,
 "hidden_size": 128, "num_layers": 3, "embedding_dim": 16}
```

Three patterns, all consistent with each other:

1. **Low learning rate + many estimators.** All three tree models converged to ~0.012–0.017 with
   800–1,100 trees, i.e. slow, heavily-averaged learning. That is the signature of a **noisy target
   with modest signal**: the optimiser is buying variance reduction, not fitting sharp structure.
2. **Real but moderate regularisation** — `min_child_samples` 56, `l2_leaf_reg` 5.9, column sampling
   ~0.71–0.74. Enough to stop memorising individual series, not enough to suggest the models were
   drowning in noise.
3. **Tuned `lookback` beat the hardcoded 56 in both directions** — LSTM chose **42**, Transformer
   **70**, TiDE **56**. Adding lookback to the search space (it was previously a fixed constant) was
   worth it: the LSTM sweep spanned 0.2650 (lookback 28) → 0.2708 (lookback 70), a spread comparable
   to the gap between entire model families. Interestingly the Transformer prefers *more* history
   (70) and the LSTM *less* (28–42) — consistent with attention being able to reach back selectively
   while a recurrent encoder has to carry everything through its state.

Notably the Transformer won with only **2 layers** and 8 heads at `d_model` 256 — wide and shallow,
not deep. On 14-day horizons with strong weekly structure, depth was not what helped.

### 5.4 Bugs found by actually running it

Six silent defects surfaced only under real execution; each is now fixed and, where sensible, pinned
by a test:

| Bug | Consequence |
|---|---|
| `splits.py` labelled validation `"val"` while tests asserted `"validation"` | test suite red |
| `middle_out_allocate()` used `validate="one_to_many"` on keys that repeat per date | crashed for **any** horizon > 1 day; `hierarchy.py` had zero test coverage |
| `inference_darts` called `TimeSeries.pd_dataframe()`, removed in Darts 0.39 | the entire Darts inference path was dead code |
| `strategy_analysis.py` called `read_raw()` without `cfg` | crashed on every run |
| `train_darts` / `inference_darts` did not exclude the dead series | Darts rejected the whole batch because `STORE0013/SKU0073`'s covariates could not span the horizon |
| tuned `lookback` compared against global config at inference | any tuned sequence model was **undeployable** |

Plus two performance/infra findings:

- **`n_jobs = -1` was ~750× slower.** The same LightGBM fit took 0.31 s at 8 threads and **235 s at
  16** — OpenMP spin-wait contention when the thread count saturates the box. It looks like a hang.
- **`recursive_forecast` rebuilt features over the full 1.07M-row history 14× per forecast.** Since
  every feature is a bounded window (max 28 days), history older than that cannot matter. Trimming to
  70 days cut it **67.9 s → 21.9 s** and peak RSS **~10 GB → 3.47 GB**. Pinned by an equivalence test
  asserting **bit-identical** predictions (`atol=0, rtol=0`).

---

## 5.5 Overfitting control: early stopping, checkpoint selection and regularisation

Overfitting was controlled at four levels — the training loop, the checkpoint choice, the
hyperparameters, and the evaluation protocol. The evidence that it worked is in §5.6.

### 5.5.1 Early stopping (sequence models)

`train_with_validation()` in `train_dl.py`. Each epoch runs a full training pass then a **validation
pass with the optimiser disabled**, so the validation loss is a clean out-of-sample reading:

```python
for epoch in range(dl_cfg["epochs"]):
    train_loss = run_epoch(model, train_loader, device, optimizer, kind)   # updates weights
    val_loss   = run_epoch(model, val_loader,   device, optimizer=None)    # no updates

    if val_loss < best_loss - 1e-4:          # min-delta: ignore noise-level "improvements"
        best_loss, best_epoch = val_loss, epoch
        patience_counter = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= dl_cfg["patience"]:
            break                            # stop before the model starts memorising
```

| Control | Value (v2) | Why |
|---|---|---|
| Monitored quantity | validation weighted L1 (scaled) | the same loss being optimised, measured out-of-sample |
| `patience` | 8 epochs | tolerate temporary plateaus without paying for a full run |
| **min delta** | `1e-4` | an "improvement" smaller than this is noise; without it patience never triggers |
| `epochs` (ceiling) | 50 | a cap, not a target — early stopping picks the actual number |

### 5.5.2 Checkpoint selection — best, not last

The single most important detail, and an easy one to get wrong:

```python
best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}   # deep copy
...
model.load_state_dict(best_state)     # restore the BEST epoch, never the final one
```

Two things this gets right:

1. **The returned model is the best-validation epoch, not the last epoch trained.** With patience 8,
   the last epoch is by definition up to 8 epochs *worse* than the best. Returning it would discard
   the whole point of early stopping.
2. **The state is deep-copied and moved to CPU.** `state_dict()` returns references to live tensors;
   without `.clone()` the "saved" checkpoint would mutate as training continued, silently saving the
   final weights instead of the best ones.

The run also fails loudly rather than silently shipping an untrained model:

```python
if best_state is None:
    raise RuntimeError("Training finished without producing a valid checkpoint")
```

### 5.5.3 The refit problem, and how it is handled

After validation selects the configuration, the model is refit on **train + validation** so it can
use the most recent two weeks before forecasting test. But that leaves no held-out data to early-stop
against — and using test would be leakage.

`refit_fixed_epochs()` resolves this by training a **fresh** model for a *fixed* `best_epoch + 1`
epochs, with **no early stopping at all**. The epoch count is a hyperparameter inherited from the
validation stage; test never influences when training stops. This is why `train_dl.py` tags its runs
`evaluation: direct_multi_horizon_no_test_early_stopping`.

### 5.5.4 Tree models — no early stopping, by choice

The tree models are fit **without** `eval_set` / `early_stopping_rounds`. Instead, `n_estimators`
(or `iterations`) is itself a tuned hyperparameter, searched jointly with learning rate and
regularisation over 25 Optuna trials against the recursive 14-day validation forecast.

This is deliberate. Early stopping on an `eval_set` would optimise row-wise error on a static
validation slice, whereas the tuner optimises **the actual deployed task** — a recursive 14-day
forecast from a fixed origin. Tuning the tree count against the real objective is the stronger
signal; adding `eval_set` early stopping on top would optimise a proxy.

The tuner landed on 834–1,094 trees at learning rates of 0.012–0.017 (§5.3), i.e. it chose slow
learning with many trees rather than stopping early — the expected shape for a noisy target.

### 5.5.5 Every other regularisation lever

| Lever | Where | Setting |
|---|---|---|
| Dropout | LSTM (between layers), Transformer | tuned 0.087–0.220 |
| Weight decay (AdamW) | both torch models | tuned 3.5e-6 – 7.0e-4 |
| **Gradient clipping** | `run_epoch` | `clip_grad_norm_(..., 1.0)` — caps exploding gradients on demand spikes |
| Scheduled teacher forcing | LSTM only | 0.2 during training, **0.0 at inference** — prevents the decoder relying on ground truth it will not have |
| Feature subsampling | LightGBM / XGBoost | `colsample_bytree` 0.707–0.737 |
| Row subsampling | LightGBM / XGBoost | `subsample` 0.723–0.990 |
| L2 | all three trees | `reg_lambda` 1.26–1.76, `l2_leaf_reg` 5.90 |
| Minimum leaf population | LightGBM / XGBoost | `min_child_samples` 56 / `min_child_weight` 5.92 |
| Sample weighting | all families | stockout-imputed rows down-weighted to 0.5 when imputation is enabled |
| Target/feature scaling | torch models | fitted on **training rows only**, frozen into the checkpoint |

### 5.5.6 Did it work? Yes — and the evidence says we could afford *less* of it

| model | train WAPE | validation WAPE | gap |
|---|---|---|---|
| xgboost | 0.2584 | 0.2721 | 0.0137 |
| lightgbm | 0.2637 | 0.2722 | 0.0085 |
| catboost | 0.2664 | 0.2719 | **0.0055** |

A train–validation gap of 0.5–1.4 percentage points on a noisy retail target is essentially no
overfitting. The models are **at capacity, not memorising**.

That has a direct consequence for v3: adding regularisation or shrinking models would not help, and
the v1→v2 result (doubling the search budget and tripling the epoch ceiling changed nothing, §9)
confirms it from the other direction. The binding constraint is the information in the features
(§11), not model variance.

One caveat worth stating: these gaps are measured against the *same* validation window used for
selection, so they understate true generalisation error slightly. The test numbers — which the
selection never saw — are the honest read, and they sit within 0.008 WAPE of validation for every
model, so the conclusion holds.

---

## 6. Metrics — how they are computed and what they mean commercially

All metrics are **pooled (micro-averaged) over every row** in the window — all 1,004 series × 14 days
= 14,056 rows — not computed per series and averaged.

| Metric | Formula | Aggregation | Business reading |
|---|---|---|---|
| **WAPE** | `Σ|y−ŷ| / Σ|y|` | volume-weighted | *"Across the estate we are off by 27% of the units we actually sell."* The decision metric: always defined, and weights a unit of error on a 500/day SKU above one on a 10/day SKU — which is how cost actually behaves. |
| **MAPE** | mean of `|y−ŷ|/|y|`, non-zero actuals only | row-weighted | Familiar to stakeholders but **unstable here**: 0.28% of rows are zero (excluded — see `mape_coverage`) and ~1.5% are under 5 units, where a 2-unit miss reads as 40–200%. Reported as a diagnostic, never as the selection key. |
| **MAE** | mean `|y−ŷ|` | row-weighted | *"We miss by ~16.8 units per store-SKU-day."* Directly interpretable in cases/pallets. |
| **RMSE** | `√mean((y−ŷ)²)` | row-weighted | Penalises large misses — the promo-spike detector. |
| **Bias** | `Σ(ŷ−y) / Σ|y|` | volume-weighted | **The inventory-critical one.** Positive = systematic over-forecast = overstock and markdown; negative = understock and lost sales. |
| **Business proxy** | under-units × unit margin; over-units × purchase cost | — | Directional cost of the asymmetry. Not an inventory simulator: no lead time, safety stock or replenishment policy. |

**Why WAPE over MAPE for decisions:** MAPE divides by each actual, so low-demand rows dominate.
Measured on the champion's own test window, MAPE more than doubles vs WAPE (0.57 vs 0.27) purely
because of small denominators. WAPE has one shared denominator and is always defined.

**The pooled figure hides segment failure**, which is why the per-series and per-horizon breakdowns
exist:

| model | pooled test WAPE | macro (per-series) | worst series | best series |
|---|---|---|---|---|
| catboost | 0.2683 | 0.2723 | 0.5574 | 0.1252 |
| tide | 0.2694 | 0.2755 | 0.6717 | 0.1261 |
| transformer | 0.2747 | 0.2835 | 0.6867 | 0.1144 |

Pooled and macro are close, so the headline is not propped up by high-volume series — but the
**per-series range is 0.11 → 0.69**, a 6× spread. Some Store-SKU pairs are forecast six times worse
than others, and that is where the next real gain lives.

**Segment breakdowns (champion, validation):**

| Slice | WAPE | MAE | Reading |
|---|---|---|---|
| Promo days (n=1,011) | 0.2755 | **31.0** | 2× the MAE of normal days — promo spikes are the weak point |
| Non-promo (n=13,045) | 0.2639 | 15.4 | |
| Convenience (n=616) | 0.2581 | 9.9 | best channel |
| Supermarket (n=3,360) | 0.2689 | 13.3 | worst channel, still within 1pp |
| Snacks (n=3,346) | 0.2646 | **22.6** | highest absolute error — high volume |
| Home Care (n=2,254) | 0.2631 | 13.9 | |

Error is **flat across the horizon** — D+1 to D+14 ranges 0.25–0.29 with no trend. For the direct
multi-horizon models that is expected, and it means 14 days is not the limiting factor.

---

## 7. Inference results

### 7.1 Validation and test, against actuals

**All six models**, validation and test, against actual units sold
(`reports/v2/all_models_full_comparison.csv`):

| model | family | val bias | **test bias** | test actual | test predicted |
|---|---|---|---|---|---|
| **lightgbm** | ml | **+0.06%** | +0.86% | 877,244 | 884,801 |
| catboost | ml | +0.14% | +0.99% | 877,244 | 885,955 |
| lstm | dl | +0.43% | −1.10% | 877,244 | 867,556 |
| transformer | dl | +0.93% | **+7.32%** | 877,244 | 941,432 |
| xgboost | ml | −1.45% | −1.27% | 877,244 | 866,078 |
| tide | darts | +3.29% | **−0.00%** | 877,244 | 877,224 |

**This table is the most commercially important result in the report.** The promoted champion
over-forecasts the held-out window by **7.3% — about 64,000 units of phantom demand**. On a 14-day
national order that is real overstock and markdown exposure.

Every other model is within ±1.3%. **LightGBM is the least biased on validation (+0.06%)** and
second-least on test; TiDE is essentially exact on test (−0.00%) despite the weakest WAPE, having
been the *most* biased on validation (+3.29%) — a reminder that bias is not stable across windows
either. WAPE alone would never have surfaced any of this, because all six WAPEs sit within 0.011.

This is precisely why all six models are registered and reported rather than only the per-family
winners: the ranking metric and the inventory-relevant metric disagree, and dropping four models
would have hidden the disagreement.

Per-series extremes for CatBoost on test:

```text
best   STORE0010 / SKU0045   WAPE 0.125   actual 1,099 units
       STORE0010 / SKU0090   WAPE 0.131   actual   267 units
worst  STORE0005 / SKU0045   WAPE 0.534   actual   985 units
       STORE0011 / SKU0023   WAPE 0.557   actual   964 units
```

Note SKU0045 appears at both extremes in different stores — the difficulty is a *store × SKU*
property, not a SKU property. That is an argument for store-level segmentation experiments, not
SKU-level ones.

### 7.2 Out-of-sample forecast (the deliverable)

Genuine out-of-sample: the 14 days **after all data ends**.

```text
horizon   2024-01-01 .. 2024-01-14
coverage  1,004 active Store-SKU series × 14 days = 14,056 rows per model
```

**All six models** forecast the same horizon — 84,336 rows total:

| model | family | total units forecast | mean units / series / day |
|---|---|---|---|
| transformer | dl | 808,023 | 57.49 |
| catboost | ml | 804,418 | 57.23 |
| lightgbm | ml | 790,566 | 56.24 |
| xgboost | ml | 779,889 | 55.48 |
| tide | darts | 762,232 | 54.23 |
| lstm | dl | 761,973 | 54.21 |

Every model sits below the 877,244 units actually sold in the final 14 days of 2023 — consistent
with the EDA's seasonal profile, where **Oct–Dec is a peak and early January is not**. All six
independently predict a post-holiday decline, which is the behaviour we want to see.

The spread is worth noting: **808,023 vs 761,973 units is a 6% disagreement** between the highest
and lowest forecast for the same fortnight, from models whose WAPEs differ by ~1%. Since these are
genuinely out-of-sample there is no ground truth to adjudicate, which is itself an argument for
running an ensemble or at minimum reporting the range to planners rather than a single number.

Files (`reports/v2/`, mirrored in the bundle):

| File | Contents |
|---|---|
| `forecast_store_sku.csv` | 42,168 rows — the three per-family champions |
| `inference_bundle/data/forecast_all_models_*.csv` | **84,336 rows — all six models** |
| `forecast_store_sku_wide.csv` | one column per model for side-by-side comparison |
| `store_sku_forecast_and_metrics.csv` | forecast **+ that series' test accuracy**, worst-first — the file to hand a planner |
| `all_models_full_comparison.csv` | every model: val/test metrics **and** bias against actuals |

### 7.3 Model registration — all six, not just the champion

Two registries, because they answer different questions:

| File | Purpose |
|---|---|
| `configs/model_registry_v2.yaml` | **the serving contract** — one model, the one the API loads |
| `configs/model_registry_all_v2.yaml` | **the audit record** — all six, ranked, with metrics and artifact paths |

`select_model.py` writes the first and refuses to promote anything that is not the validation
champion. `register_all_models.py` writes the second, so no trained model is lost just because it
did not win. Each entry carries family, artifact path, tuned hyperparameters, train/validation/test
metrics, and predicted-vs-actual totals:

```yaml
registered_at: '2026-09-09T06:52:00Z'
selection_metric: validation_wape
champion: transformer
model_count: 6
models:
  - name: transformer
    family: dl
    rank_by_validation: 1
    is_champion: true
    artifact_path: artifacts_v2/transformer/model.pt
    tuned_params: {lookback: 70, transformer_d_model: 256, ...}
    validation_metrics: {wape: 0.265420, mape: 0.533587, ...}
    test_metrics: {wape: 0.274722, ...}
    test_totals: {actual_units: 877244.0, predicted_units: 941432.49, bias_pct: 7.3169}
  - name: lstm
    rank_by_validation: 2
    ...
```

Regenerate for either run:

```bash
python -m demand_forecasting.register_all_models --config configs/config_v2.yaml \
    --output configs/model_registry_all_v2.yaml     # 6 models
python -m demand_forecasting.register_all_models --config configs/config.yaml \
    --output configs/model_registry_all_v1.yaml     # 5 models (v1 had no Darts)
```

All six model binaries are also bundled in `final_conclusion/inference_bundle/models/`, so any of
them can be reloaded and compared without retraining.

### 7.4 Regenerating it

`final_conclusion/inference_bundle/` holds the models, the known-future covariates, the config and
the registry. Verified end to end:

```bash
export PYTHONPATH=src
python final_conclusion/inference_bundle/generate_forecast.py --history data/raw/data.csv --model all
```

See that folder's `README.md`. Raw history is deliberately not copied (212 MB, DVC-tracked).

---

## 8. MLflow tracking on DagsHub

**Tracking server:** <https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow>

| Experiment | Runs | Link |
|---|---|---|
| `fmcg-demand-forecasting-v1` | 67 (5 final · 54 tuning · 8 parents) | [experiment 2](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/2) |
| `fmcg-demand-forecasting-v2` | 96 (8 final · 81 tuning · 7 parents) | [experiment 3](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3) |

**Final runs (v2)** — every hyperparameter, metric and model artifact:

| Model | val_wape | Run |
|---|---|---|
| transformer | 0.26542 | [81f426ed](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3/runs/81f426ed1bd947788c3ef0d41299fd40) |
| lstm | 0.26617 | [12c11670](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3/runs/12c116700648433abd34144ee74b4e6f) |
| catboost | 0.27193 | [ecf48825](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3/runs/ecf488259fdb4bb8a534b1c591d9e1a7) |
| xgboost | 0.27213 | [8efb2268](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3/runs/8efb226821904b339aefec9160ee057d) |
| lightgbm | 0.27217 | [e37112b1](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3/runs/e37112b147c54425812c2722f2cda594) |
| tide | 0.27622 | [cc14be10](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3/runs/cc14be10653d45fa98f0f6ec11a90465) |

**Run structure.** Each tuning study opens a parent run (`stage=tuning_parent`) with one **nested run
per Optuna trial** (`stage=tuning`) carrying that trial's hyperparameters, train *and* validation
metrics, and — in v2 — **its fitted model artifact**. Filters:

```text
tags.stage = 'final'
tags.stage = 'tuning' and tags.model = 'lightgbm'
tags.family = 'darts'
```

Two gaps were found and fixed mid-project by querying the live experiment rather than reading code:
trial models were never logged (`log_trial_models` defaulted false, and `tune_dl.py` had no
persisting code at all), and no `train_*` metrics existed on tuning runs. Both now log. Separately,
**nothing in the codebase ever called `load_dotenv()`** — valid DagsHub credentials sat unused while
everything logged locally.

---

## 9. Bayesian hyperparameter tuning

**Optuna TPE** (`multivariate=True`, seeded from `project.random_seed`), minimising a validation
metric measured on a **real recursive 14-day forecast** — a trial is scored exactly the way the model
will be deployed, not on a row-wise score over pre-computed lags.

| Family | Tuner | Trials | Searches lookback |
|---|---|---|---|
| lightgbm / xgboost / catboost | `tune_bayesian.py` | 25 (v2) | n/a |
| lstm / transformer | `tune_dl.py` | 2–3 | **yes**, 14→112 in weekly steps |
| tide / tsmixer | `tune_dl.py` | 2 | **yes** |

Studies persist to `artifacts_v2/tuning/<model>_study.db` with `load_if_exists=True`, so an
interrupted sweep resumes rather than restarting — which mattered, because the run was OOM-killed
three times on a box with 24 GB total and ~4.6 GB free.

### The most important tuning result: budget was not the constraint

| model | v1 (12 trials, 15 epochs) | v2 (25 trials, 50 epochs) | delta |
|---|---|---|---|
| catboost | 0.27193 | 0.27193 | **0.00000** |
| lightgbm | 0.27208 | 0.27217 | +0.00009 |
| xgboost | 0.27158 | 0.27213 | +0.00056 |
| transformer | 0.26482 | 0.26542 | +0.00060 |
| lstm | 0.26504 | 0.26617 | +0.00113 |

**Not one model improved.** CatBoost converged to the *identical* configuration — TPE had already
found that optimum in 12 trials. XGBoost is the instructive row: with 25 trials it found a
configuration that fits training noticeably better (train WAPE 0.2644 → **0.2584**) while validating
*worse*. That is the **search overfitting to the single validation window** — more attempts buy
precision on that specific fortnight, not generalisation.

---

## 10. Pipeline and engineering

| Concern | Implementation |
|---|---|
| **Config** | `configs/config.yaml` (+ `config_v2.yaml`) — every setting; nothing hard-coded. Passed explicitly as a dict, so alternate configs redirect data, artifacts, experiment and logs in one move |
| **Data versioning** | **DVC** tracks `data/raw/data.csv` (212 MB, never in Git) with a DagsHub remote; `dvc.yaml` defines the `prepare` stage so features rebuild reproducibly |
| **Experiment tracking** | **MLflow → DagsHub**, credentials from `.env` via `load_dotenv(override=False)` so an exported shell variable still wins |
| **Logging** | One midnight-rotating file per step under `logs/`, 14-day retention. Each run records context, the **entire effective config**, settings used, progress, metrics and artifact paths — a run is reproducible from its log alone |
| **Source control** | Git; `.gitignore` excludes raw data, artifacts, logs, `mlruns/`, `mlflow.db`, `catboost_info/` |
| **Model registry** | `configs/model_registry_v2.yaml` — artifact path, family, selection metric *and* value, plus validation and test metrics. `select_model.py` **refuses to promote anything that is not the current validation champion** |
| **Serving** | FastAPI `POST /forecast`; `inference_router.py` dispatches on `family` so the API is model-agnostic. Validation rejects wrong horizons, duplicates, gaps and unknown series with a 422 before any model loads |
| **Containerisation** | `Dockerfile` (python:3.11-slim, non-root); data/artifacts/configs/logs mounted, not baked in |
| **Orchestration** | `scripts/run_v1.sh` — per-family trial counts, `PASS`/`FAIL` status trail, skip-if-exists resume, graceful fallback when a tuning stage produced no params |
| **Testing** | 28 tests: leakage, splits, metrics, hierarchy, and the history-trim equivalence test |
| **Monitoring** | `drift.py` — PSI/KS numeric, PSI categorical, promo-regime check, thresholds in config |

**Pipeline order:**

```text
prepare_dataset → tune_bayesian / tune_dl → train_ml / train_dl / train_darts
    → evaluate_compare → report_best_models → select_model → run_final_inference
                                                    ↓
                                        model_registry.yaml → api.py / Docker
```

---

## 10.1 Production readiness — what was built to make this deployable

This section deliberately restates points from §3, §7 and §10 in one place: the question "is this
production-grade?" should be answerable without reading the whole report.

### A. Reproducibility

| Element | Implementation |
|---|---|
| **Data versioning** | DVC tracks `data/raw/data.csv` (212 MB) with a DagsHub remote. Git holds only the `.dvc` pointer; `dvc repro` rebuilds features from the tracked version |
| **Code versioning** | Git, with `.gitignore` excluding raw data, artifacts, logs, `mlruns/`, `mlflow.db`, `catboost_info/` |
| **Config as data** | One YAML per run (`config.yaml`, `config_v2.yaml`) read into a plain dict and passed explicitly. No behaviour is hard-coded in a training script |
| **Determinism** | `project.random_seed: 42` seeds Python, NumPy, PyTorch, CUDA, the Optuna TPE sampler, and every row subsample |
| **Full config in every log** | `log_run_context()` dumps the *entire effective configuration* into each step's log, so a run is reproducible from its log alone |
| **Resumable search** | Optuna studies persist to `<model>_study.db` with `load_if_exists=True`; an interrupted sweep continues instead of restarting |

Four things reproduce any result: the DVC data version, the Git commit, the `Effective configuration`
block in the step log, and the MLflow run id printed in the same log.

### B. Experiment tracking and model governance

- **MLflow on DagsHub**, 163 runs across two experiments. Every tuning trial is a nested run under a
  study parent, carrying its hyperparameters, **train and validation** metrics, and (v2) its fitted
  model artifact — so any configuration in the sweep can be reloaded, not just the winner.
- **Two registries with different jobs**: `model_registry_v2.yaml` is the serving contract (one
  model, what the API loads); `model_registry_all_v2.yaml` is the audit record (all six, ranked,
  with metrics and artifact paths).
- **A promotion gate**: `select_model.py` re-reads the comparison table and **refuses to promote
  anything that is not the current validation champion**, which prevents someone promoting a model
  because its *test* number looked good.
- **Run tagging** (`stage`, `model`, `family`) so `tags.stage = 'final'` isolates comparable runs.

### C. Serving

- **FastAPI** `POST /forecast`, with `GET /health`.
- **`inference_router.py` dispatches on `family`**, so the API is model-agnostic — swapping the
  registry from a tree model to a Transformer to a Darts model requires no API change.
- **Request validation before any model loads**, each failure returning a 422 that says why:
  required fields present (driven by the `*_known_future` contract), dates parseable, no duplicate
  `(store, SKU, date)`, exactly `horizon` rows per series, dates contiguous from the day after
  history ends, and the series known.
- **Docker**: `python:3.11-slim`, non-root user, data/artifacts/configs/logs **mounted rather than
  baked in**, so an image is not invalidated by a retrain.
- **`make_future_template.py`** generates the request body from the contract, so the caller never has
  to guess which fields are required.

### D. Observability

- **One midnight-rotating log per step** under `logs/`, 14-day retention, `logs/orchestration/` for
  run-level status. Each records run context, the full config, settings actually used, progress
  milestones, resulting metrics and every artifact path written.
- **Drift monitoring** (`drift.py`): PSI and KS for numeric features, PSI for categoricals, an
  explicit promo-regime check, thresholds in config (PSI ≥ 0.10 warn, ≥ 0.25 alert; promo-rate
  relative change ≥ 0.30 alert). Three layers are specified: input drift, prediction drift before
  labels arrive, and performance drift after.
- **Per-series and per-horizon breakdowns** so a degradation can be localised to a Store-SKU or a
  horizon day rather than showing up only as a worse aggregate.

### E. Correctness guarantees

- **28 automated tests**: leakage (mutating `y_t` must not change row `t`'s features), split
  boundaries, metric maths, hierarchy coherence, and an **equivalence test** pinning the
  history-trim optimisation to bit-identical predictions.
- **Perturbation probes** (§3.3) proving the end-to-end forecast paths ignore future labels, future
  inventory, and covariates declared unknown — with a control proving the models *do* read the
  covariates declared known.
- **Schema validation at load**: DL and Darts inference verify the stored feature contract against
  the current config and refuse to run on a mismatch; lookback is taken from the artifact because it
  is tuned per model.
- **Fail-loud defaults**: missing predictions after a merge, empty splits, invalid stockout config,
  a checkpoint that never improved — all raise rather than silently continuing.

### F. Operational resilience

- **`run_v1.sh`** records a `PASS`/`FAIL`/`SKIP` trail with durations, **continues past a failing
  model** rather than aborting the other five, skips stages whose outputs exist, and falls back to
  default hyperparameters if a tuning stage produced no params file.
- **Resource guards learned the hard way**: `n_jobs` never saturates the box (the OpenMP cliff cost
  750×), `recursive_forecast` trims history to the longest feature window (10 GB → 3.5 GB peak), and
  DL training windows are capped so an epoch is minutes rather than hours.
- **Graceful degradation for inactive series**: a model trains on series that later go inactive;
  inference filters them out rather than failing the batch.

### G. Honest gaps

Not claiming production-complete. Missing: rolling-origin validation (§12.1), automated retraining
on a schedule, a seasonal-naive baseline, prediction intervals, CI/CD, authentication and rate
limiting on the API, and a feature store — the API currently reads history from CSV, which will not
scale to a real serving path.

---

## 11. Conclusion — which model, and why

**Promoted:** global Transformer, validation WAPE 0.2654 (`lookback` 70, `d_model` 256, 8 heads,
2 layers).

**Recommended to ship: CatBoost.** Four reasons, all evidenced:

1. **The accuracy difference is inside the noise band.** All six models span 0.2654 → 0.2762 on
   validation — 4% relative. The gap between the Transformer and CatBoost is 0.0065 WAPE.
2. **The validation ranking has failed to predict the test ranking three times.** In v1 the
   Transformer won validation and CatBoost was best on test; in v2 the Transformer won validation and
   the LSTM was best on test — with the Transformer finishing *worst* of the five non-Darts models.
   Selecting on one 14-day December origin cannot resolve differences of ~0.005 WAPE.
3. **The bias result is decisive.** The Transformer over-forecast test by **+7.3%**; CatBoost by
   **+1.0%** (+0.14% on validation) and LightGBM by **+0.86%** (+0.06% on validation — the least
   biased of all six). For inventory that difference dominates a 0.006 WAPE edge.
4. **Operational cost.** CatBoost trains in ~25 s vs 10–30 min, tunes at ~40 s/trial vs 7–30 min,
   runs inference on CPU in ~22 s for all 1,004 series, and gives SHAP explanations out of the box.

CatBoost is also the only model that is simultaneously top-three on validation, top-three on test,
and **bit-identical between v1 and v2** — the most stable thing in the study.

**Runner-up: LightGBM.** It is last of the six on validation WAPE (0.2722) and third on test
(0.2704), but it is the **least biased model in the study** (+0.06% validation, +0.86% test) and the
fastest to train. If the operating priority is "do not systematically over- or under-order", the
honest ranking puts it first. That the WAPE ranking and the bias ranking disagree this sharply is the
strongest argument in this report for reporting several models rather than a single champion — and
the reason all six are registered.

**The other three are not discardable.** LSTM has the best test WAPE, MAPE and MAE of any model.
TiDE has near-perfect test bias. XGBoost is the most conservative (negative bias on both splits),
which is the safer failure mode for perishable stock. Each is a defensible pick under a different
objective, so all six are registered with their metrics rather than only the per-family winners.

### The strongest finding is that the model barely matters

Three independent lines of evidence converge:

1. Six architectures with very different inductive biases land within 4% of each other.
2. Train and validation WAPE are nearly identical (0.2664 vs 0.2719 for CatBoost) — **nothing is
   overfitting the data**.
3. Doubling the search budget and tripling the epoch budget changed nothing — and TiDE, which never
   got a tuned search at all, still lands within 0.4% of the tuned Transformer on test.

The ceiling is the **information in the feature set**, not the function class on top of it, its
hyperparameters, or the optimisation budget. A ~27% WAPE on daily store-SKU demand with ~8% promo
rate and 3% stockouts is a defensible v1 baseline — but the way to improve it is more signal, not
more model.

---

## 12. Future improvements

Ordered by expected value per unit of effort.

**1. Rolling-origin validation — the single highest-value change.** Everything above rests on one
December fortnight, and the validation ordering has now contradicted the test ordering three times.
Six to twelve origins spread across seasons, selecting on **mean WAPE with the spread reported**,
would turn "the Transformer wins by 0.0008" into a defensible statement. Cost is linear in the number
of origins; a two-stage search (narrow on 2–3 origins, confirm finalists on all) keeps it affordable.

**2. Output chunk shift — remove recursion from the tree path.** The tree models forecast recursively:
D+2's `lag_1` is D+1's prediction, so errors compound across the horizon. A **direct multi-horizon**
formulation instead trains 14 models (or one multi-output model) each predicting day *D+k* from
origin-time features only — no feedback loop, no error accumulation. The sequence models already do
this, which may be exactly why they edge the trees on validation. Darts exposes this natively as
`output_chunk_shift`. Concretely: train `h ∈ {1..14}` heads on `demand_lag_{h, h+6, h+13, h+27}` so
every lag is genuinely observable at the origin. Worth testing directly against the recursive
baseline — our per-horizon curve is currently *flat*, which suggests recursion is not yet hurting
much, so this should be measured rather than assumed.

**3. Attack the per-series spread, not the average.** Per-series WAPE ranges 0.11 → 0.69. Segment the
worst decile and test whether it is low-volume, promo-heavy or stockout-prone; `store_sku_forecast_and_metrics.csv`
is sorted worst-first and is the place to start. That SKU0045 is both a best and a worst series in
different stores points at store-level effects.

**4. More signal, not more capacity.** Since nothing overfits:
   - promo **depth** interactions and lead/lag effects around campaigns — promo lift is 63–131% by
     category yet enters as a single binary flag, and promo-day MAE is 2× normal;
   - price relative to category and to competing SKUs in the same subcategory;
   - richer calendar: paydays, school terms, holiday *proximity* rather than a same-day binary;
   - turn on `price_known_future` **if** the business genuinely commits prices 14 days ahead — a
     contract question, not a modelling one; the code already supports the switch everywhere.

**5. Test the stockout correction.** `stockout_target_mode` is still `none`, so 3% of rows train on
censored sales as if they were demand. `rolling_median` and `percentage` are implemented and
leakage-tested but were never compared. Cheap, well-scoped experiment.

**6. Quantile forecasts instead of point forecasts.** Inventory needs a service level, not a mean.
Pinball loss at P50/P90 would make the output directly usable for safety-stock sizing and would turn
the asymmetric business proxy from descriptive into actionable — especially given the bias findings
in §7.1.

**7. Different architectures worth a fair test.** N-BEATS/N-HiTS (strong on pure seasonality), TFT
(native known-future handling plus interpretable attention), PatchTST, and DeepAR for native
probabilistic output. Also a **seasonal-naive baseline** (`y[t−7]`) — v1 never established one, so we
cannot currently state how much the models beat "same weekday last week", which is the first question
any reviewer will ask.

**8. Engineering follow-ups.**
   - Score the training set in `train_dl`/`train_darts` so `train_wape` is populated for every family.
   - Tune the tree lag/rolling windows — currently the only untuned "input chunk length".
   - Vectorise `MultiSeriesWindowDataset.__getitem__`; at 0.42 ms/sample the GPU is starved and
     `max_train_windows` could then be removed entirely.
   - CatBoost native categorical handling (ordered target statistics) vs the current shared ordinal
     encoding.
   - Scheduled retraining with the drift report as trigger.

**Explicitly not worth doing:** a bigger Transformer, or a wider hyperparameter sweep. v1→v2 already
demonstrated neither moves the number.
