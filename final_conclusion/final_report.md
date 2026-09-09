# FMCG Multi-Store Demand Forecasting

## Modelling report and production design

**Author:** Himanshu Verma
**Repository:** github.com/Himanshu-Verma-ds/demand-forecasting · dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment
**Experiment tracking:** DagsHub MLflow, experiments 2 (v1) and 3 (v2)

---

## 1. What we set out to build

The brief was to forecast daily `units_sold` for every active store-SKU pair, fourteen days ahead.
Three years of daily history were supplied: 1,100,000 rows from 2021-01-01 to 2023-12-31, covering
13 stores, 102 SKUs and 1,005 distinct store-SKU series across 5 categories, 17 subcategories and
4 channels.

We treated this as a production problem rather than a modelling exercise. That framing drove most of
the decisions that follow: the evaluation had to reproduce the deployed task exactly, every feature
had to be available at the moment a planner would actually run the forecast, and the whole thing had
to be reproducible from a commit hash and a config file.

Six model families were trained end to end across two complete experiment runs, producing 163 tracked
MLflow runs. The short version of the result is that all six landed within four percent of each
other, and the most useful finding of the project is *why* that happened.

---

## 2. Exploratory analysis, and what it settled

The EDA was not a chart gallery. We ran it to answer specific questions whose answers would become
configuration values, and the notebook (`notebooks/01_eda.ipynb`) is organised that way: every
section ends with a note on what the finding implies for the model. This section walks through what
we found and what each finding fixed.

### 2.1 Is the data clean enough to build lag features directly?

Yes, and this was worth confirming before anything else. There are no missing values anywhere, and
exactly one row per (store, SKU, date), and we checked for duplicate keys rather than assuming it. A
clean daily grain means lag and rolling features can be computed directly, with no imputation layer
underneath them that would need its own leakage argument.

Two quirks did turn up. `lead_time_days` reaches 17 against a mean of 6, and `margin_pct` goes
negative on some rows, which matters for the business-cost proxy but not for demand itself. More
interestingly, `stock_out_flag` is *not* equivalent to zero inventory: almost every stockout row
still shows positive `stock_on_hand`. The two fields evidently describe different concepts or
timestamps, so stockouts cannot be re-derived from inventory levels and the flag has to be taken at
face value.

### 2.2 Is the hierarchy stable enough for a global model?

Every `sku_id` maps to exactly one name, category, subcategory and brand, and every `store_id` maps
to one country, city and channel. There is no effective-dated master-data problem to solve, which
means static attributes can be treated as genuinely static: safe to encode once and attach to every
row of a series.

Coverage is unusually strong. The median series has the complete 1,095 days, at least 75% of series
are complete, and 1,004 of 1,005 reach the final date. The single exception is `STORE0013 / SKU0073`,
which runs continuously from 2021-01-01 to **2022-09-12** and then stops. That one series accounts
for the entire 475-row shortfall we had already noticed on STORE0013.

This settled the modelling strategy. Dense, near-complete bottom-level history makes a **single global
model across all series** viable, and makes bottom-up aggregation the natural reconciliation
baseline. Roughly a thousand local models would have been statistically defensible here but
operationally indefensible, and they would have been useless for any new SKU. We also fixed the
active forecast set at **1,004 series**, excluding the dead one.

That last point turned out to matter more than expected. Two production bugs later traced back to
that single series, and because the EDA had already identified it by name and date, both took minutes
to diagnose instead of hours.

### 2.3 What does the demand distribution imply about the loss function?

`units_sold` is clearly right-skewed: median around 49, mean around 59, and a maximum of 704. Scale
varies substantially between series, which is exactly what you would expect across hypermarkets and
convenience stores.

Skew of this shape argues against optimising squared error, which would let a handful of high-volume
spikes dominate the gradient. We therefore used **L1 objectives** throughout: `regression_l1` for
LightGBM, `MAE` for CatBoost, and a weighted L1 loss for both PyTorch models, and made **WAPE** the
decision metric rather than RMSE. RMSE is still reported, because it is the right lens for promo-spike
failures, but it does not drive selection.

Heterogeneous scale also argued for a *conditional* global model rather than one unconditioned curve.
Store, SKU, category and channel identifiers let a single model learn different levels while sharing
seasonal and promotional structure across series.

### 2.4 What is the seasonal structure? (This set the lag configuration.)

This was the most consequential part of the analysis, because the lag set falls directly out of it.

Aggregate demand is higher on Saturdays and Sundays. Monthly totals peak in June–August and again in
October–December, and the notebook confirms the same shape repeats in all three years, so it is
seasonality rather than a one-off. Autocorrelation of aggregate daily demand shows clean peaks at
**lags 7, 14, 21 and 28**. Looking at the twelve highest-volume series individually, the same annual
pattern appears with series-specific magnitude: shared timing, individual scale.

Every temporal feature in the model traces back to those peaks:

| Configuration | Value | Justification from the EDA |
|---|---|---|
| `demand_lags` | 1, 7, 14, 28 | 1 for short-run level; 7 is the dominant ACF peak (same weekday last week); 14 and 28 confirm the weekly signal is stable and span a monthly cycle |
| `demand_roll_windows` | 7, 14, 28 | level, volatility and recent peak measured over exactly one, two and four weekly cycles |
| Cyclical calendar terms | `dow_sin/cos`, `month_sin/cos`, `doy_sin/cos` | raw integers would place Sunday six units from Monday, and December twelve from January; the sine/cosine pair makes them adjacent |
| `is_weekend` | binary | the weekend lift is visible in aggregate and worth giving the model directly |
| `stock_roll_window` | 28 | matches the longest demand window so all trailing statistics share a horizon |
| `lookback` search range | 14 to 112, in steps of 14 | weekly structure means candidate history windows should be whole numbers of weeks |

None of these are library defaults. The choice of 7/14/28 is the measured autocorrelation, and the
cyclical encodings exist because of a specific defect in the alternative.

The lookback range deserves a note. Rather than fixing the sequence models' input window at an
arbitrary value, we made it a tunable and let the search choose, constrained to whole weeks and to at
least one full horizon. The models picked **42 days for the LSTM, 70 for the Transformer and 56 for
TiDE**, which is six, ten and eight weeks respectively. Every winning value is a multiple of seven, which is
a quiet confirmation that the weekly structure the EDA found is real and that the models are using it.

### 2.5 How should promotions be treated?

Promotional rows show substantially higher average demand across every category, with descriptive
uplift ranging from roughly 63% to 131%; Home Care and Beverages show the largest gaps. Average demand
also rises monotonically with discount depth.

We were careful about what this does and does not establish. Promotions are scheduled, not randomised,
and are plausibly placed onto periods that were already strong. The relationship is therefore
descriptive, and we make no causal claim about uplift.

What it does justify is treating `promo_flag` as a **known-future covariate** (`promo_known_future:
true`), on the stated assumption that the promotional calendar is committed before the forecast
origin. Because roughly 7% of rows carry close to double the usual demand, we also insisted on a
separate promo error slice, because a good aggregate number can easily conceal a bad promotional one.

It does. On the champion's validation window, promo days score 0.2755 WAPE against 0.2639 on normal
days, but the mean absolute error is **twice as large**: 31.0 units versus 15.4. Promotions are the
weakest part of the model, and the aggregate metric barely registers it.

### 2.6 What about stockouts?

Stockouts affect roughly 3% of rows. The conceptual problem is that on those days `units_sold` records
what was *available to sell*, not what customers wanted. Training on it directly teaches the model
that demand was low precisely when it may have been high.

We built three strategies behind `stockout_target_mode`: leave the target alone, replace it with the
maximum of observed sales and a trailing non-stockout rolling median, or apply a fixed percentage
uplift, with imputed rows down-weighted to 0.5 so the model trusts them less. All three always emit
`demand_target`, `sample_weight` and `stockout_imputed_amount`, so nothing downstream branches on the
choice.

Both production runs used `none`. Correcting censored demand is a modelling assumption, and the honest
first version ships the uncorrected baseline and treats the correction as an experiment to be
measured. We did not get to that measurement, and it is the cheapest open item on the list in
section 12.

### 2.7 Which covariates are legitimately known in advance?

Price and weather are available in the historical data, but that is not the question. The question is
whether a planner running the forecast on the morning of day zero would have them for all fourteen
days ahead. For price we judged not, absent a committed pricing calendar, and for weather a 14-day
forecast is not something we can assume is wired in.

Both are therefore set to `false`, and the code enforces it everywhere, in feature construction, all
three inference paths, and the API's request validation. The training contract and the serving
contract are the same object, which is the only way this stays honest.

### 2.8 Summary: EDA findings to configuration

| What the EDA found | What it fixed in the configuration |
|---|---|
| No missing values, one row per (store, SKU, date) | Lag features computed directly; no imputation layer |
| Consistent SKU and store metadata | Static attributes safe to encode once per series |
| Median series complete, 1,004 of 1,005 active | Global model over all series; active set of 1,004 |
| `STORE0013/SKU0073` ends 2022-09-12 | Explicitly excluded from forecasting |
| Right-skewed target, heterogeneous scale | L1 objectives; WAPE as decision metric; identity features |
| ACF peaks at 7, 14, 21, 28 | `demand_lags` 1/7/14/28; rolling windows 7/14/28 |
| Weekend and annual seasonality, stable across years | Cyclical encodings; `is_weekend`; whole-week lookback search |
| Promo lift 63–131%, monotonic in discount | `promo_known_future: true`; dedicated promo error slice |
| Stockouts 3%; flag ≠ zero inventory | Configurable censored-demand target; sample weight 0.5 |
| Price and weather not committed 14 days out | `price_known_future: false`, `weather_known_future: false` |
| 13 stores, 102 SKUs, low-moderate cardinality | Ordinal encoding for trees; embeddings of 8–32 for sequence models |

---

## 3. Building the training data

Three deterministic steps produce the modelling table, and the same three run at inference. That
identity is what makes train/serve skew structurally impossible rather than merely unlikely.

```python
raw  = read_raw(path, cfg)                                 # parse dates, sort by (store, sku, date)
df   = add_stockout_target(raw, cfg)                       # demand_target + sample_weight
feat = build_causal_features(df, cfg, "demand_target")     # lags, rollings, calendar, promo, price
```

The result is 1,100,000 rows and 72 columns, of which roughly 52 numeric and 9 categorical reach the
models.

The panel stays long. We do not split into one frame per series; instead every row carries its
`store_id`, `sku_id`, `store_sku_id`, `channel`, `category`, `subcategory` and `brand`, and one global
model reads all 1,005 series at once. Every group-wise operation is scoped by
`groupby(["store_id", "sku_id"])`, so no series can contaminate another's lags.

The features fall into seven groups:

| Group | Contents |
|---|---|
| Demand history | `demand_lag_{1,7,14,28}`; shifted rolling mean, standard deviation and max over 7, 14 and 28 days |
| Calendar (known future) | year, month, day, weekday, week-of-year, `is_weekend`, `is_holiday`, and six cyclical terms |
| Promotions | `promo_flag` (known future), `promo_prev_1`, `promo_rate_28` |
| Price history | `list_price_lag_1`, `discount_pct_lag_1`, and their 28-day trailing means |
| Inventory history | `stockout_lag_{1,7,14}`, `stockout_rate_28`, `stock_on_hand_lag_1`, `stock_on_hand_mean_7` |
| Cross-series | `store_demand_mean_lag_1`, `sku_demand_mean_lag_1` — day-lagged group averages |
| Identity | nine categoricals plus `series_age_days` |

For the sequence models the same information becomes tensors: a past window of shape
`[batch, lookback, 14]`, a known-future window of `[batch, 14, 6]`, and six static embedding indices.

---

## 4. Keeping the evaluation honest

Demand forecasting makes leakage easy to introduce and hard to notice, because the symptom is a
metric that looks good. We treated this as the central engineering risk.

### 4.1 The controls

Revenue columns (`gross_sales`, `net_sales`) are permanently blocked, since both are `units_sold`
multiplied by price and would hand the model its own target. Every rolling statistic derived from the
target is shifted before it is rolled, so `s.shift(1).rolling(28).mean()`, never
`s.rolling(28).mean()`. Same-day `stock_out_flag` and `stock_on_hand` never appear as features
because they are outcomes rather than inputs; only their lagged forms are used.

The split is chronological. Tree validation and test forecasts are produced **recursively from a fixed
origin**, so day D+2's `lag_1` is day D+1's *prediction*, never its actual value. The sequence models
see only their lookback window plus genuinely-known future covariates. Scaling statistics and
categorical index maps are fitted on training rows alone and frozen into the checkpoint.

On the tuning side, Bayesian optimisation minimises validation error only, and the test window is
scored exactly once. Deep-learning early stopping uses validation, and the subsequent refit runs a
fixed epoch count so the test window cannot influence when training halts.

### 4.2 Why recursion matters

A common shortcut computes `lag_1` across the whole dataframe and then filters out the validation
rows. The row for December 5th then contains December 4th's actual demand, which is unknowable at a
December 3rd origin. That single mistake can roughly halve the reported error.

```text
origin = 2023-12-03 (end of training)
D+1  2023-12-04   features from real history through 12-03        -> 47.2
D+2  2023-12-05   lag_1 = 47.2, the prediction, not the actual    -> 51.8
...
D+14 2023-12-17
```

### 4.3 Proving it rather than asserting it

Unit tests only cover the feature builder, so we attacked the end-to-end forecast paths directly:
corrupt something that is unknowable at the origin, and check the forecast does not move. A
bit-identical result means the information could not have been used.

| Probe | What we corrupted | Result |
|---|---|---|
| Trees, future labels | `units_sold × 1000 + 12345`, revenue columns × 999 | No change |
| Trees, future inventory | Flipped `stock_out_flag`, zeroed `stock_on_hand` | No change |
| Trees, unknown covariates | Price × 5, discount 0.9, temperature −50, rain 999 | No change |
| Trees, recursion | Inspected D+2's `demand_lag_1` | Equals D+1's prediction, not its actual |
| Trees, blocked columns | Inspected the 52 model inputs | None of the blocked names present |
| Sequence models | Same corruption of labels and inventory | No change |
| **Control:** promo, declared known | Flipped `promo_flag` | Moved by 151.3 and 115.8 units |
| Stockout imputation | Built targets from train-only versus full data | 0 of 21,340 training targets changed |

The control row is what makes the others meaningful. Without it, six zero-deltas would be equally
consistent with a model that simply ignores all of its future inputs.

---

## 5. Splitting the data

The boundaries are computed backwards from the last available date, so they move automatically as
data arrives.

```text
train        2021-01-01 to 2023-12-03      1,071,888 rows
validation   2023-12-04 to 2023-12-17         14,056 rows    selection and tuning
test         2023-12-18 to 2023-12-31         14,056 rows    scored once
```

Each evaluation window is 14,056 rows: 1,004 active series across 14 days.

Both windows are fourteen days because the deployed task is a fourteen-day batch forecast from a fixed
origin. Selecting on the same geometry means validation measures the real task rather than a proxy
for it.

We did not use k-fold cross-validation, and it is worth being explicit about why. Random folds place
future rows into training, and because our features are lags and rolling windows, neighbouring rows
are near-duplicates, so a random split leaves copies of nearly every validation row in the training
set. It also measures the wrong problem: filling in a missing day given both of its neighbours is far
easier than forecasting fourteen days forward, and is not what we deploy. The correct analogue for
time series is rolling-origin backtesting, which is the top item in section 12.

After validation freezes the configuration, a fresh model is refit on train plus validation, and the
test window is forecast from the new origin. The test set never influences any choice.

---

## 6. Training setup and overfitting control

Overfitting was managed at four levels: the training loop, the checkpoint decision, the
hyperparameters, and the evaluation protocol.

### 6.1 Early stopping

Each epoch runs a training pass and then a validation pass with the optimiser disabled, so the
validation loss is a clean out-of-sample reading rather than a partially-updated one.

```python
for epoch in range(dl_cfg["epochs"]):
    train_loss = run_epoch(model, train_loader, device, optimizer, kind)   # updates weights
    val_loss   = run_epoch(model, val_loader,   device, optimizer=None)    # no updates

    if val_loss < best_loss - 1e-4:          # min-delta: ignore noise-level improvement
        best_loss, best_epoch = val_loss, epoch
        patience_counter = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= dl_cfg["patience"]:
            break
```

Patience was 3 epochs in the first run and 8 in the second, against ceilings of 15 and 50. The
`1e-4` minimum delta is not decoration: without it, arbitrarily small fluctuations count as
improvements and the patience counter never advances.

### 6.2 Checkpoint selection

The model returned is the **best-validation epoch, not the last one trained**. With patience of 8,
the final epoch is by definition up to eight epochs past the best, and returning it would discard the
entire benefit of early stopping.

Equally important, and easier to get wrong: `state_dict()` returns references to live tensors. We
deep-copy and move to CPU, otherwise the "saved" checkpoint mutates as training continues and quietly
becomes the final weights rather than the best ones. The run raises rather than proceeding if no
checkpoint was ever recorded.

### 6.3 The refit problem

Refitting on train plus validation lets the model use the two most recent weeks before it forecasts
test — but it leaves nothing to early-stop against, and using test for that purpose would be leakage.

We resolve it by training a fresh model for a **fixed** `best_epoch + 1` epochs with no early stopping
at all. The epoch count is inherited from the validation stage as a hyperparameter, so the test window
plays no part in deciding when training ends.

### 6.4 Why the tree models have no early stopping

The gradient-boosted models are fit without `eval_set` or `early_stopping_rounds`. Instead the number
of trees is itself a tuned hyperparameter, searched jointly with learning rate and regularisation
against the recursive fourteen-day validation forecast.

This is deliberate. Early stopping on an `eval_set` optimises row-wise error against a static slice,
whereas the tuner optimises the actual deployed task. Tuning tree count against the real objective is
the stronger signal, and layering `eval_set` stopping on top would only optimise a proxy.

The search settled on 834 to 1,094 trees at learning rates between 0.012 and 0.017: slow, heavily
averaged learning rather than early termination, which is the expected shape for a noisy target.

### 6.5 Remaining regularisation

| Mechanism | Where | Setting |
|---|---|---|
| Dropout | Both sequence models | Tuned, 0.087 to 0.220 |
| Weight decay (AdamW) | Both sequence models | Tuned, 3.5e-6 to 7.0e-4 |
| Gradient clipping | Training loop | `clip_grad_norm_` at 1.0, guarding against demand spikes |
| Scheduled teacher forcing | LSTM only | 0.2 while training, 0.0 at inference |
| Row and feature subsampling | LightGBM, XGBoost | `subsample` 0.72–0.99, `colsample_bytree` 0.71–0.74 |
| L2 penalty | All three tree models | `reg_lambda` 1.26–1.76, `l2_leaf_reg` 5.90 |
| Minimum leaf population | LightGBM, XGBoost | `min_child_samples` 56, `min_child_weight` 5.92 |
| Sample weighting | All families | Imputed stockout rows down-weighted to 0.5 |
| Scaling statistics | Sequence models | Fitted on training rows only, frozen into the checkpoint |

### 6.6 Did any of it prove necessary?

| Model | Train WAPE | Validation WAPE | Gap |
|---|---|---|---|
| XGBoost | 0.2584 | 0.2721 | 0.0137 |
| LightGBM | 0.2637 | 0.2722 | 0.0085 |
| CatBoost | 0.2664 | 0.2719 | 0.0055 |

A gap of half a point to one and a half points on a noisy retail target is, in practical terms, no
overfitting at all. The models are at capacity rather than memorising.

That has a direct consequence: adding regularisation or shrinking the models would not help. The
experiment in section 7 confirms it from the opposite direction.

One honest caveat. These gaps are measured against the same validation window used for selection, so
they slightly understate true generalisation error. The test numbers, which selection never saw, are
the reliable read, and they sit within 0.008 WAPE of validation for every model, so the conclusion
holds.

---

## 7. The two experiment runs

### 7.1 Design

| | Run 1 | Run 2 |
|---|---|---|
| MLflow experiment | `fmcg-demand-forecasting-v1` (67 runs) | `fmcg-demand-forecasting-v2` (96 runs) |
| Tree trials per model | 12 | 25 |
| Sequence trials per model | 2–3 | 2–3 |
| Epoch ceiling / patience | 15 / 3 | 50 / 8 |
| Every trial's model saved | No | Yes, 80 artifacts |
| Artifact directory | `artifacts/` | `artifacts_v2/` |

Data, features, split and leakage contract are identical between the two. Only the search budget and
training schedule differ, which makes the comparison clean.

### 7.2 Bayesian optimisation

We used Optuna's TPE sampler with multivariate sampling, seeded from `project.random_seed`. The
objective is validation WAPE measured on a **genuine recursive fourteen-day forecast**, so a trial is
scored exactly the way the model will be used, not as a row-wise score over pre-computed lags.

Sequence models additionally search `lookback`, which had previously been a fixed constant. Studies
persist to SQLite with `load_if_exists=True`, so an interrupted sweep resumes rather than restarting.
That mattered: the run was killed three times by the operating system on a machine with 24 GB of
memory and roughly 4.6 GB free.

### 7.3 The result that shaped our conclusions

| Model | Run 1 validation | Run 2 validation | Change |
|---|---|---|---|
| CatBoost | 0.27193 | 0.27193 | 0.00000 |
| LightGBM | 0.27208 | 0.27217 | +0.00009 |
| XGBoost | 0.27158 | 0.27213 | +0.00056 |
| Transformer | 0.26482 | 0.26542 | +0.00060 |
| LSTM | 0.26504 | 0.26617 | +0.00113 |

Doubling the tree search and more than tripling the epoch ceiling improved nothing. Not one model got
better; CatBoost converged to a byte-identical configuration, meaning TPE had already found that
optimum within twelve trials.

The XGBoost row is the instructive one. With twenty-five trials the search found a configuration that
fits the training data noticeably better — train WAPE improving from 0.2644 to 0.2584 — while
validating slightly worse. That is the hyperparameter search overfitting to the single validation
window. More attempts buy precision on that specific fortnight, not generalisation.

### 7.4 Winning hyperparameters

```
CatBoost      iterations 1007, learning_rate 0.0172, depth 8,
              l2_leaf_reg 5.90, random_strength 0.093

LightGBM      n_estimators 834, learning_rate 0.0120, num_leaves 158, max_depth 9,
              min_child_samples 56, subsample 0.723, colsample_bytree 0.737, reg_lambda 1.264

XGBoost       n_estimators 1094, learning_rate 0.0140, max_depth 8, min_child_weight 5.92,
              subsample 0.990, colsample_bytree 0.707, reg_lambda 1.756

Transformer   lookback 70, learning_rate 0.00054, dropout 0.087, batch_size 128,
              d_model 256, heads 8, layers 2, embedding_dim 8

LSTM          lookback 42, learning_rate 0.00412, dropout 0.220, batch_size 128,
              hidden_size 128, num_layers 3, embedding_dim 16
```

Three patterns stand out, and they are consistent with one another. All three tree models converged
on low learning rates with many trees, which is the signature of a noisy target with modest signal:
the optimiser is buying variance reduction rather than fitting sharp structure. Regularisation
settled at moderate levels: enough to stop the models memorising individual series, not so much as to
suggest they were drowning in noise. And the tuned lookback values beat the previously hardcoded 56
in both directions, with the Transformer preferring more history than the LSTM. That is consistent
with attention being able to reach back selectively while a recurrent encoder has to carry everything
forward through its hidden state.

The Transformer also won with only two layers at a width of 256, wide and shallow. On a fourteen-day
horizon with strong weekly structure, depth was not what helped.

---

## 8. Metrics, and what they mean commercially

All metrics are pooled across every row in the window — 1,004 series times 14 days — rather than
computed per series and averaged.

**WAPE**, the sum of absolute errors divided by the sum of actuals, is our decision metric. It reads
as "across the estate we are off by 27% of the units we actually sell", it is always defined, and it
weights a unit of error on a 500-per-day SKU above one on a 10-per-day SKU, which is how cost
behaves.

**MAPE** is reported but never used for selection. It divides by each individual actual, and this
dataset has 0.28% of rows at zero (excluded, with the coverage reported alongside) and roughly 1.5%
below five units, where a two-unit miss reads as a 40% to 200% error. On the same test window it more
than doubles relative to WAPE — 0.57 against 0.27 — entirely because of small denominators.

**MAE** is the most directly interpretable: we miss by roughly 16.8 units per store-SKU-day, which
translates into cases or pallets. **RMSE** penalises large misses and is the promo-spike detector.

**Bias**, the signed sum of errors over the sum of actuals, is the inventory-critical one. Positive
means systematic over-forecasting and therefore overstock and markdown; negative means understock and
lost sales. It is the metric that ultimately changed our recommendation.

A business proxy is also computed: under-forecast units multiplied by unit margin against
over-forecast units multiplied by purchase cost. It is a directional cost comparison, not an inventory
simulator: lead time, safety stock and replenishment policy are not modelled.

### 8.1 What the pooled number conceals

| Model | Pooled test WAPE | Per-series average | Worst series | Best series |
|---|---|---|---|---|
| CatBoost | 0.2683 | 0.2723 | 0.5574 | 0.1252 |
| TiDE | 0.2694 | 0.2755 | 0.6717 | 0.1261 |
| Transformer | 0.2747 | 0.2835 | 0.6867 | 0.1144 |

Pooled and per-series averages are close, so the headline is not being propped up by high-volume
series. But the per-series range runs from 0.11 to 0.69 — some store-SKU pairs are forecast six times
worse than others, and that spread is where the next real accuracy gain lives.

Segment breakdowns on the champion's validation window tell a similar story. Promotional days score
0.2755 WAPE against 0.2639 for normal days, but with double the absolute error. By channel, Convenience
is best at 0.2581 and Supermarket worst at 0.2689, all within a point. Snacks carries the highest
absolute error at 22.6 units, which is a volume effect rather than a quality one.

Error is essentially flat across the horizon, ranging 0.25 to 0.29 from D+1 to D+14 with no trend.
For direct multi-horizon models that is expected, and it tells us the fourteen-day horizon is not the
limiting factor.

---

## 9. Results

### 9.1 Final standings

| Rank | Model | Family | Train | Validation | Test WAPE | Test MAPE | Test MAE |
|---|---|---|---|---|---|---|---|
| 1 | Transformer | dl | — | 0.2654 | 0.2747 | 0.5728 | 17.14 |
| 2 | LSTM | dl | — | 0.2662 | 0.2673 | 0.5134 | 16.68 |
| 3 | CatBoost | ml | 0.2664 | 0.2719 | 0.2683 | 0.5283 | 16.75 |
| 4 | XGBoost | ml | 0.2584 | 0.2721 | 0.2714 | 0.5199 | 16.94 |
| 5 | LightGBM | ml | 0.2637 | 0.2722 | 0.2704 | 0.5304 | 16.88 |
| 6 | TiDE | darts | — | 0.2762 | 0.2694 | 0.5263 | 16.81 |

Training WAPE is blank for the sequence models because they early-stop on validation and never score
the training set — an omission worth fixing, noted in section 12.

### 9.2 Bias against actuals

This table changed our recommendation, and it is the one we would put in front of a planning team.

| Model | Validation bias | Test bias | Test actual | Test predicted |
|---|---|---|---|---|
| LightGBM | +0.06% | +0.86% | 877,244 | 884,801 |
| CatBoost | +0.14% | +0.99% | 877,244 | 885,955 |
| LSTM | +0.43% | −1.10% | 877,244 | 867,556 |
| Transformer | +0.93% | **+7.32%** | 877,244 | 941,432 |
| XGBoost | −1.45% | −1.27% | 877,244 | 866,078 |
| TiDE | +3.29% | −0.00% | 877,244 | 877,224 |

The promoted champion over-forecasts the held-out window by 7.3%, roughly 64,000 units of demand that
did not exist. On a fourteen-day national order that is real overstock and markdown exposure. Every
other model sits within 1.3%. LightGBM is the least biased on validation and second-least on test;
TiDE is almost exactly unbiased on test despite having the weakest WAPE, having been the most biased
on validation — so bias is not stable across windows either.

None of this is visible in WAPE, where all six models sit within 0.011 of one another. It is the main
reason we register and report all six models rather than only the per-family winners.

### 9.3 Out-of-sample forecast

The deliverable is the fourteen days that follow all available data: 2024-01-01 to 2024-01-14, for all
1,004 active series.

| Model | Total units forecast | Mean units per series per day |
|---|---|---|
| Transformer | 808,023 | 57.49 |
| CatBoost | 804,418 | 57.23 |
| LightGBM | 790,566 | 56.24 |
| XGBoost | 779,889 | 55.48 |
| TiDE | 762,232 | 54.23 |
| LSTM | 761,973 | 54.21 |

Every model forecasts below the 877,244 units sold in the final fortnight of 2023, which is what we
would expect from the seasonal profile: October to December is a peak, early January is not. All six
independently predict a post-holiday decline.

The spread is worth flagging. A 6% disagreement between the highest and lowest forecast, from models
whose WAPEs differ by about one percent, is a genuine uncertainty signal. There is no ground truth for
this window, which argues for reporting the range to planners — or running an ensemble — rather than a
single point estimate.

### 9.4 Where everything lives

`reports/v1/` and `reports/v2/` hold the model comparisons, per-series and per-horizon breakdowns, and
the forecasts. The file most useful to a planner is
`store_sku_forecast_and_metrics.csv`, which joins each series' fourteen-day forecast with how
accurately that same series was predicted on the held-out window, sorted worst first. It tells you
both what we forecast and how much to trust it for that specific store-SKU.

`final_conclusion/inference_bundle/` contains all six trained models, the known-future covariates, the
configuration, both registries and a standalone script. The forecast reproduces exactly:

```
history ends 2023-12-31
forecasting  2024-01-01 to 2024-01-14 for 1004 store-SKU series
  lightgbm     14,056 rows, 790,566 units
  xgboost      14,056 rows, 779,889 units
  catboost     14,056 rows, 804,418 units
  lstm         14,056 rows, 761,973 units
  transformer  14,056 rows, 808,023 units
  tide         14,056 rows, 762,232 units
```

---

## 10. Tracking and model governance

Everything is tracked in MLflow hosted on DagsHub: 163 runs across the two experiments. Each tuning
study opens a parent run with one nested run per Optuna trial, carrying that trial's hyperparameters,
its train and validation metrics, and — in the second run — its fitted model artifact. Any
configuration in the sweep can be reloaded, not just the winner. Runs are tagged by stage, model and
family, so `tags.stage = 'final'` isolates the comparable models.

Two registries serve different purposes. `model_registry_v2.yaml` is the serving contract: one model,
the one the API loads. `model_registry_all_v2.yaml` is the audit record: all six, ranked, with
metrics, tuned parameters, artifact paths and predicted-versus-actual totals. Promotion is gated —
`select_model.py` re-reads the comparison table and refuses to promote anything that is not the
current validation champion, which stops anyone promoting a model because its test number looked
appealing.

Two tracking defects were found by querying the live experiment rather than reading the code. Trial
models were not being logged at all, because the flag defaulted to off and the sequence-model tuner
had no persisting code in the first place. And no train metrics existed on tuning runs, so per-trial
overfitting was invisible. Both are fixed. Separately, nothing in the codebase ever called
`load_dotenv()`, which meant valid DagsHub credentials sat unused while everything logged locally.

---

## 11. Production engineering

**Reproducibility.** DVC tracks the 212 MB raw dataset against a DagsHub remote, with only the pointer
in Git. Configuration is data — one YAML per run, read into a dict and passed explicitly, with nothing
hard-coded in a training script. A single seed covers Python, NumPy, PyTorch, CUDA, the TPE sampler
and every row subsample. Each step's log contains the entire effective configuration, so a run is
reproducible from its log alone. Four things pin any result: the DVC data version, the Git commit, the
configuration block in the log, and the MLflow run id printed beside it.

**Serving.** A FastAPI service exposes `POST /forecast` and `GET /health`. The router dispatches on
model family, so swapping the registry between a tree model, a Transformer and a Darts model requires
no change to the API. Requests are validated before any model loads — required fields present, dates
parseable, no duplicate keys, exactly fourteen rows per series, dates contiguous from the day after
history ends, series known — each failure returning a 422 that says which rule was broken. The Docker
image runs as a non-root user on `python:3.11-slim`, with data, artifacts, configs and logs mounted
rather than baked in.

**Observability.** One log file per pipeline step, rotating at midnight with fourteen days of
retention. Each records run context, the full configuration, the settings actually used, progress
milestones, resulting metrics and every artifact path written. Drift monitoring computes PSI and KS
for numeric features, PSI for categoricals and an explicit promotional-regime check, with thresholds
in configuration.

**Correctness.** Twenty-eight automated tests cover leakage, split boundaries, metric arithmetic,
hierarchy coherence, and an equivalence test that pins a performance optimisation to bit-identical
predictions. Inference validates the stored feature contract against the current configuration and
refuses to run on a mismatch.

**Resilience.** The orchestration script records a pass/fail trail with durations, continues past a
failing model rather than aborting the remaining five, skips stages whose outputs already exist, and
falls back to default hyperparameters when a tuning stage produced no parameter file.

Three defects found during the runs are worth recording, because each was silent. Setting `n_jobs`
to the full CPU count triggered OpenMP spin-wait contention that made the same LightGBM fit take 235
seconds instead of 0.31 — it presents as a hang, not a slowdown. The recursive forecast was rebuilding
features across the entire 1.07 million-row history fourteen times per forecast; since every feature
is a bounded window of at most 28 days, trimming to 70 days cut it from 67.9 to 21.9 seconds and peak
memory from roughly 10 GB to 3.5 GB, pinned by an equivalence test. And the Darts integration failed
on the one series ending in 2022 — twice, once in training and once in inference, because a model
legitimately trains on series that later go inactive but must exclude them when forecasting.

**What is missing.** We are not claiming production-complete. There is no rolling-origin validation,
no automated retraining schedule, no seasonal-naive baseline, no prediction intervals, no CI/CD, and
no authentication or rate limiting on the API. The service also reads history from CSV, which will not
survive contact with a real serving path.

---

## 12. Conclusions and recommendations

The promoted model is the global Transformer at 0.2654 validation WAPE. **The model we would actually
deploy is CatBoost**, and the reasoning is worth setting out plainly.

The accuracy difference between them is inside the noise band. All six models span 0.2654 to 0.2762,
about four percent relative, and the gap between the Transformer and CatBoost is 0.0065 WAPE.
Meanwhile the validation ranking has failed to predict the test ranking three separate times: in the
first run the Transformer won validation and CatBoost was best on test; in the second the Transformer
won validation and the LSTM was best on test, with the Transformer finishing worst of the five
non-Darts models. A single fourteen-day December origin cannot resolve differences of half a
percentage point.

The bias result settles it. The Transformer over-forecasts test by 7.3% against CatBoost's 1.0% and
LightGBM's 0.86%. For an inventory decision that difference dominates a 0.006 WAPE edge entirely.
Operationally the tree models are also far cheaper: roughly 25 seconds to train against 10 to 30
minutes, 40 seconds per tuning trial against 7 to 30, CPU inference across all 1,004 series in about
22 seconds, and SHAP explanations available without extra work.

CatBoost is additionally the only model that is simultaneously top-three on validation, top-three on
test, and byte-identical between the two runs. LightGBM is the natural runner-up: last of six on
validation WAPE, third on test, and the least biased model in the study. If the operating priority is
to avoid systematically over- or under-ordering, it arguably ranks first. That the WAPE ranking and
the bias ranking disagree this sharply is the single strongest argument in this report for reporting
several models rather than one champion.

The other three are not disposable either. The LSTM has the best test WAPE, MAPE and MAE of any model.
TiDE is almost perfectly unbiased on test. XGBoost is the most conservative, under-forecasting on both
splits, which is the safer failure mode for perishable stock. Each is defensible under a different
objective, which is why all six are registered with their metrics.

**The most useful finding is that the model barely matters.** Three independent lines of evidence
converge on it. Six architectures with very different inductive biases land within four percent of one
another. Train and validation error are nearly identical, so nothing is overfitting the data. And
doubling the search budget while tripling the epoch ceiling changed nothing — while TiDE, which never
received a tuned search at all, still lands within 0.4% of the tuned Transformer on test.

The ceiling is the information in the feature set, not the function class sitting on top of it. A WAPE
of roughly 27% on daily store-SKU demand, with 8% promotional days and 3% stockouts, is a defensible
first version. The way to improve it is more signal, not more model.

---

## 13. Where we would go next

**Rolling-origin validation, first and by some distance.** Everything above rests on a single December
fortnight, and the validation ordering has now contradicted the test ordering three times. Six to
twelve origins spread across seasons, selecting on mean WAPE with the spread reported, would turn "the
Transformer wins by 0.0008" into a defensible statement. Cost scales linearly with the number of
origins, and a two-stage search — narrow on two or three origins, confirm the finalists on all of them
— keeps it affordable.

**Remove recursion from the tree path via output chunk shift.** The tree models forecast recursively,
so D+2's `lag_1` is D+1's prediction and errors compound along the horizon. A direct multi-horizon
formulation trains a separate head for each day, predicting D+k from origin-time features only, with
no feedback loop. Concretely, train heads for h in 1 to 14 on `demand_lag_{h, h+6, h+13, h+27}` so
every lag is genuinely observable at the origin. The sequence models already work this way, which may
be part of why they edge the trees on validation. Worth measuring rather than assuming, though: our
per-horizon error curve is currently flat, which suggests recursion is not yet costing much.

**Attack the per-series spread rather than the average.** Per-series WAPE ranges from 0.11 to 0.69.
Segment the worst decile and establish whether it is low-volume, promotion-heavy or stockout-prone;
`store_sku_forecast_and_metrics.csv` is already sorted worst-first. One detail points the way: SKU0045
appears among both the best and the worst series in different stores, so the difficulty is a
store-by-SKU property rather than a SKU property, which argues for store-level segmentation
experiments over SKU-level ones.

**More signal, not more capacity.** Since nothing overfits, the returns are in features. Promotional
depth interactions and lead/lag effects around campaigns are the obvious first target, since uplift runs
from 63% to 131% by category, yet promotions enter the model as a single binary flag, and promotional
days already carry double the error. Beyond that: price relative to category and to competing SKUs in
the same subcategory; a richer calendar covering paydays, school terms and holiday proximity rather
than a same-day binary; and enabling `price_known_future` if the business genuinely commits prices
fourteen days ahead, which is a commercial question rather than a modelling one.

**Test the stockout correction.** Three percent of rows still train on censored sales as though they
were demand. Both correction strategies are implemented and leakage-tested but were never compared.
This is cheap and well-scoped.

**Move to quantile forecasts.** Inventory decisions need a service level, not a mean. Pinball loss at
the median and the 90th percentile would make the output directly usable for safety-stock sizing, and
would turn the asymmetric cost proxy from descriptive into actionable — particularly given the bias
findings in section 9.2.

**Architectures worth a fair test.** N-BEATS and N-HiTS are strong on pure seasonality; the Temporal
Fusion Transformer handles known-future covariates natively and offers interpretable attention;
PatchTST and DeepAR are also reasonable candidates, the latter for native probabilistic output. Before
any of that, though, we should establish a **seasonal-naive baseline** using demand from seven days
prior. We never did, which means we currently cannot state how much these models beat "the same
weekday last week" — and it is the first question any reviewer will ask.

**Engineering follow-ups.** Score the training set in the sequence-model trainers so train WAPE is
populated for every family. Tune the tree lag and rolling windows, currently the only untuned input
window. Vectorise the windowed dataset's item lookup, which at 0.42 milliseconds per sample is
starving the GPU and forcing us to cap training windows. Add CatBoost's native categorical handling as
a comparison against the shared ordinal encoding. And schedule retraining, triggered by the drift
report.

**What is not worth doing:** a larger Transformer, or a wider hyperparameter sweep. The second run
already demonstrated that neither moves the number.
