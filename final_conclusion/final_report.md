# FMCG Multi-Store Demand Forecasting

**Himanshu Verma**

Forecasting daily units sold for every active store-SKU pair, 14 days ahead.

| | |
|---|---|
| Code (GitHub) | [github.com/Himanshu-Verma-ds/demand-forecasting](https://github.com/Himanshu-Verma-ds/demand-forecasting) |
| Code + data (DagsHub) | [dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment) |
| MLflow — run 1 (67 runs) | [experiment 2](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/2) |
| MLflow — run 2 (96 runs) | [experiment 3](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3) |

Data: 1.1M rows, 2021-01-01 to 2023-12-31, 13 stores, 102 SKUs, 1,005 store-SKU series.

---

## 1. EDA: what I asked, what I found, why it mattered

I ran the EDA to answer specific questions whose answers would become configuration values, not to
produce charts. Notebook: `notebooks/01_eda.ipynb`.

| Question I asked | What the data said | What I did about it |
|---|---|---|
| Is the data clean enough to build lag features directly? | No missing values; exactly one row per store-SKU-date | Built lags directly, no imputation layer needed underneath |
| Do stockouts equal zero inventory? | No. Almost every stockout row still shows positive stock on hand | Treated the stockout flag as authoritative; it cannot be re-derived from inventory |
| Is the product/store hierarchy stable? | Every SKU maps to one category, subcategory and brand; every store to one country, city, channel | Encoded static attributes once per series; no effective-dating needed |
| Is history dense enough for series-level forecasting? | Median series has all 1,095 days; 1,004 of 1,005 reach the final date | Chose a single global model over all series rather than ~1,000 local ones |
| Are any series dead? | One (store 13 / SKU 73) stops on 2022-09-12 | Fixed the active forecast set at 1,004 series and excluded it |
| What shape is demand? | Right-skewed: median 49, mean 59, max 704 | Used absolute-error objectives, and WAPE as the decision metric rather than RMSE |
| Is there weekly seasonality? | Autocorrelation peaks cleanly at 7, 14, 21 and 28 days | Set demand lags at 1, 7, 14, 28 days and rolling windows at 7, 14, 28 |
| Is there annual seasonality? | Peaks in Jun–Aug and Oct–Dec, repeating in all three years | Added cyclical day-of-week, month and day-of-year encodings |
| Do promotions matter? | Promo days show 63%–131% higher demand; demand rises with discount depth | Made the promo flag a known-future covariate and gave promo days their own error slice |
| How common are stockouts? | ~3% of rows | Built a configurable censored-demand target with down-weighting |
| Are price and weather usable? | Present historically, but not committed 14 days ahead | Excluded both from future covariates |

Two findings did most of the work.

**The autocorrelation set the lags.** Peaks at 7, 14, 21 and 28 days are why the lag set is 1, 7, 14
and 28 rather than anything else: one day for short-run level, seven for the same weekday last week,
then 14 and 28 to confirm the weekly signal is stable and to span a monthly cycle. The same logic
fixed the rolling windows at one, two and four weeks. The cyclical calendar encodings exist because
plain integers would place Sunday six units away from Monday, and December twelve from January.

I also let the sequence models choose their own history window rather than fixing it, constrained to
whole weeks. They picked 42 days (LSTM), 70 (Transformer) and 56 (TiDE) — six, ten and eight weeks.
Every winner is a multiple of seven, which is a quiet confirmation that the weekly structure is real
and the models are using it.

**The promo slice earned its place.** Promotions are roughly 7% of rows carrying close to double the
demand. On validation they score 0.2755 WAPE against 0.2639 on normal days, but the mean absolute
error is twice as large: 31.0 units against 15.4. Promotions are the weakest part of the model and
the aggregate number barely registers it.

---

## 2. How I prepared the dataset

Three steps, run identically in training and at inference, which is what makes train/serve skew
structurally impossible: load and sort by series and date, build the demand target with stockout
handling, then build point-in-time features.

The result is 1.1M rows and 72 columns, of which about 52 numeric and 9 categorical reach the models.

I kept the panel long rather than splitting it per series. Every row carries its store, SKU, channel,
category, subcategory and brand, and one global model reads all 1,005 series at once. Every
group-wise calculation is scoped to a single store-SKU pair, so no series can contaminate another's
lags.

### Historical features vs known-future covariates

The split between these two is the core contract of the whole system.

| Type | Features |
|---|---|
| **Historical only** (never known ahead) | Demand lags at 1, 7, 14, 28 days; rolling mean, standard deviation and max over 7, 14, 28 days; lagged stockout flags and 28-day stockout rate; lagged and rolling stock on hand; lagged price and discount plus their 28-day means; prior-day promo state and 28-day promo rate; day-lagged store-level and SKU-level average demand |
| **Known future** (available at forecast time) | Calendar: year, month, day, weekday, week of year, weekend flag, holiday flag, and six cyclical encodings. Commercial: promotion flag |
| **Static** | Store, SKU, store-SKU key, country, city, channel, category, subcategory, brand, series age |
| **Blocked entirely** | Gross and net sales (both are units times price, so they leak the target); same-day stockout flag and stock on hand (outcomes, not inputs); purchase cost and margin, kept only for the business-cost proxy |

Price and weather are historically available but excluded from the future set, because a planner
running the forecast on day zero would not have committed prices or a 14-day weather forecast. The
training contract and the serving contract are the same object.

---

## 3. How I made sure there was no leakage

Six controls, then proof.

1. Revenue columns are permanently blocked, since both are units multiplied by price.
2. Every rolling statistic derived from the target is shifted by one day before being rolled.
3. Same-day stockout and inventory are never features; only their lagged forms.
4. The split is chronological, never random.
5. Tree models forecast the validation and test windows **recursively from a fixed origin**, so day
   two's one-day lag is day one's *prediction*, never its actual value.
6. Scaling statistics and category mappings are fitted on training rows only and frozen into the
   saved model.

The failure mode I was guarding against is subtle: computing lags across the whole table and *then*
filtering out validation rows leaves December 5th holding December 4th's actual demand, which is
unknowable at a December 3rd origin. That single mistake can roughly halve reported error.

Unit tests only cover the feature builder, so I attacked the end-to-end forecast paths directly by
corrupting information that is unknowable at the origin and checking the forecast did not move.

| What I corrupted | Result |
|---|---|
| Future actual demand, multiplied by 1000 | No change to any forecast |
| Future stockout flags and inventory | No change |
| Future price, discount, temperature, rainfall | No change |
| **Promotion flag** (declared known) | Moved by 151 units — correctly *does* influence the forecast |
| Recursion check: day two's one-day lag | Equals day one's prediction, not its actual |
| Stockout target built from train-only vs full data | 0 of 21,340 training targets changed |

The promotion row is the control. Without it, six zero-deltas would be equally consistent with a
model that simply ignores all of its future inputs.

---

## 4. Train, validation and test split

| Split | Dates | Rows | Purpose |
|---|---|---|---|
| Train | 2021-01-01 to 2023-12-03 | 1,071,888 | Fitting |
| Validation | 2023-12-04 to 2023-12-17 | 14,056 | Tuning and model selection |
| Test | 2023-12-18 to 2023-12-31 | 14,056 | Scored once, at the end |

Each evaluation window is 1,004 active series across 14 days. Both are 14 days long because the
deployed task is a 14-day batch forecast, and selecting on the same geometry means validation
measures the real task.

I did not use k-fold. Random folds put future rows into training, and because the features are lags
and rolling windows, neighbouring rows are near-duplicates, so a random split leaves copies of nearly
every validation row in the training set. After validation froze the configuration, I refit on train
plus validation and forecast the test window once from the new origin.

---

## 5. Tech stack

| Purpose | Tool | How I used it |
|---|---|---|
| Data versioning | DVC | Tracks the 212 MB raw dataset against a DagsHub remote; only the pointer file is in Git |
| Experiment tracking | MLflow on DagsHub | 163 runs across two experiments; every tuning trial logged with parameters, metrics and its fitted model |
| Code hosting | GitHub + DagsHub | Same history on both; DagsHub also holds the data and experiments |
| Hyperparameter search | Optuna (TPE sampler) | Bayesian search scored on a real 14-day recursive forecast; studies persist so an interrupted sweep resumes |
| Tree models | LightGBM, XGBoost, CatBoost | Global models over the long panel |
| Sequence models | PyTorch (LSTM, Transformer) | Written from scratch, shared weights across all series |
| Library models | Darts (TiDE) | Global model with static and future covariates |
| Serving | FastAPI | Single forecast endpoint; routes to the right adapter based on the registered model family |
| Containerisation | Docker | Slim Python base, non-root user, data and artifacts mounted rather than baked in |
| Logging | Python logging | One midnight-rotating file per pipeline step, 14-day retention |
| Monitoring | Custom drift module | PSI and KS tests on features, plus an explicit promotional-regime check |
| Testing | pytest | 28 tests covering leakage, splits, metrics, hierarchy and a performance-equivalence check |

---

## 6. Pipeline flow

1. **Prepare** — load raw data, build the demand target and point-in-time features.
2. **Tune** — Bayesian search per model against a recursive 14-day validation forecast. Every trial
   logged to MLflow with its parameters, train and validation metrics, and its fitted model.
3. **Train** — fit on the tuned parameters, evaluate validation, refit on train plus validation,
   score test once.
4. **Compare** — rank all models on validation, and report the best in each family.
5. **Register** — promote the validation champion for serving, and separately register all six models
   with their metrics for audit.
6. **Forecast** — generate the next 14 days for every active store-SKU pair.
7. **Serve** — the API loads whichever model the registry names.

A single orchestration script runs the whole chain, recording pass/fail per stage and skipping work
that already exists so an interrupted run resumes.

---

## 7. Results

### Model comparison

| Rank | Model | Family | Train WAPE | Validation WAPE | Test WAPE | Test MAPE | Test MAE | MLflow run |
|---|---|---|---|---|---|---|---|---|
| 1 | Transformer | Deep learning | — | **0.2654** | 0.2747 | 0.5728 | 17.14 | [run](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3/runs/81f426ed1bd947788c3ef0d41299fd40) |
| 2 | LSTM | Deep learning | — | 0.2662 | **0.2673** | **0.5134** | **16.68** | [run](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3/runs/12c116700648433abd34144ee74b4e6f) |
| 3 | CatBoost | Tree | 0.2664 | 0.2719 | 0.2683 | 0.5283 | 16.75 | [run](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3/runs/ecf488259fdb4bb8a534b1c591d9e1a7) |
| 4 | XGBoost | Tree | 0.2584 | 0.2721 | 0.2714 | 0.5199 | 16.94 | [run](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3/runs/8efb226821904b339aefec9160ee057d) |
| 5 | LightGBM | Tree | 0.2637 | 0.2722 | 0.2704 | 0.5304 | 16.88 | [run](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3/runs/e37112b147c54425812c2722f2cda594) |
| 6 | TiDE | Darts | — | 0.2762 | 0.2694 | 0.5263 | 16.81 | [run](https://dagshub.com/Himanshu-Verma-ds/demand-forecasting-assignment.mlflow/#/experiments/3/runs/cc14be10653d45fa98f0f6ec11a90465) |

Train WAPE is blank for the sequence models because they early-stop on validation and never score the
training set.

### Forecast bias against actual units

WAPE says all six models are equivalent. Bias does not, and this is the table I would put in front of
a planning team.

| Model | Validation bias | Test bias | Test actual units | Test predicted units |
|---|---|---|---|---|
| LightGBM | **+0.06%** | +0.86% | 877,244 | 884,801 |
| CatBoost | +0.14% | +0.99% | 877,244 | 885,955 |
| LSTM | +0.43% | −1.10% | 877,244 | 867,556 |
| Transformer | +0.93% | **+7.32%** | 877,244 | 941,432 |
| XGBoost | −1.45% | −1.27% | 877,244 | 866,078 |
| TiDE | +3.29% | −0.00% | 877,244 | 877,224 |

### Out-of-sample forecast, 2024-01-01 to 2024-01-14

The 14 days after all available data, for all 1,004 active series.

| Model | Total units forecast | Mean units per series per day |
|---|---|---|
| Transformer | 808,023 | 57.49 |
| CatBoost | 804,418 | 57.23 |
| LightGBM | 790,566 | 56.24 |
| XGBoost | 779,889 | 55.48 |
| TiDE | 762,232 | 54.23 |
| LSTM | 761,973 | 54.21 |

All six forecast below the 877,244 units sold in the final fortnight of 2023, which matches the
seasonal profile: October to December is a peak, early January is not. The 6% spread between the
highest and lowest forecast, from models whose WAPEs differ by about one percent, is a real
uncertainty signal and argues for reporting a range to planners rather than a point estimate.

### Metric definitions

| Metric | What it is | What it means commercially |
|---|---|---|
| WAPE | Total absolute error divided by total actual units | Reads as "off by 27% of the units actually sold". Volume-weighted, always defined. My decision metric |
| MAPE | Average of per-row percentage errors | Familiar, but unstable here: 0.28% of rows are zero and 1.5% are under five units, where a two-unit miss reads as a 200% error. Reported, never used to select |
| MAE | Average absolute error in units | A miss of roughly 17 units per store-SKU-day. Translates directly into cases |
| RMSE | Root mean squared error | Penalises large misses; the promo-spike detector |
| Bias | Signed total error over total actuals | Positive means systematic over-forecasting, so overstock and markdown. The inventory-critical metric |

### What the pooled number hides

Per-series WAPE ranges from 0.11 to 0.69 — some store-SKU pairs are forecast six times worse than
others. Pooled and per-series-averaged figures are close, so high-volume series are not propping up
the headline, but that spread is where the next real accuracy gain is. Error is flat across the
horizon, ranging 0.25 to 0.29 from day one to day fourteen with no trend, so the 14-day horizon is
not the limiting factor.

---

## 8. Which model I would put in production, and why

The promoted champion is the Transformer, because it won on validation. **The model I would actually
deploy is CatBoost.**

Four reasons:

1. **The accuracy difference is inside the noise.** All six models span 0.2654 to 0.2762 on
   validation, about four percent relative. The gap between the Transformer and CatBoost is 0.0065.
2. **The validation ranking has failed to predict the test ranking three times.** In run 1 the
   Transformer won validation and CatBoost was best on test. In run 2 the Transformer won validation
   and the LSTM was best on test, with the Transformer finishing worst of the five non-Darts models.
   One 14-day December origin cannot resolve differences of half a percentage point.
3. **The bias result decides it.** The Transformer over-forecasts test by 7.3%, roughly 64,000 units
   of demand that did not exist. CatBoost is at 1.0% and LightGBM at 0.86%. For an inventory
   decision that dominates a 0.006 WAPE edge entirely.
4. **Operationally it is far cheaper.** CatBoost trains in about 25 seconds against 10 to 30 minutes,
   tunes at 40 seconds per trial against 7 to 30 minutes, runs inference on CPU across all 1,004
   series in about 22 seconds, and supports SHAP explanations without extra work.

CatBoost is also the only model that is top-three on both validation and test and byte-identical
between the two runs. LightGBM is the natural runner-up: last of six on validation WAPE, third on
test, and the least biased model in the study.

**The most useful finding is that the model barely matters.** Six architectures with very different
inductive biases land within four percent of each other; train and validation error are nearly
identical, so nothing is overfitting; and doubling the search budget while tripling the epoch ceiling
changed nothing. TiDE, which never received a tuned search, still lands within 0.4% of the tuned
Transformer on test. The ceiling is the information in the features, not the model on top of them.

---

## 9. Cold start and stockouts

**Stockouts (3% of rows).** On a stockout day the recorded units sold is what was *available to
sell*, not what customers wanted. Training on it directly teaches the model that demand was low
precisely when it may have been high. I built three strategies: leave the target alone, replace it
with the higher of observed sales and a trailing non-stockout rolling median, or apply a fixed
percentage uplift — with imputed rows down-weighted so the model trusts them less. Every strategy
emits the same columns, so nothing downstream changes when the strategy does.

Both production runs used the uncorrected target. Correcting censored demand is a modelling
assumption, and the honest first version ships the baseline and treats the correction as an
experiment to be measured. I did not get to that measurement, and it is the cheapest open item I have.

**Cold start.** This dataset has almost none — 1,004 of 1,005 series have three full years — but the
design still has to handle it. The global model is the main defence: a new SKU inherits everything
the model has learned about its category, subcategory, brand and channel, which a per-series model
could not do at all. Unseen categorical values map to a reserved "unknown" index rather than raising,
and unseen sequence-model identifiers fall back to a learned unknown embedding. Where history is
shorter than the longest lag, the hierarchy features and calendar still carry signal. For genuine
launches, a subcategory-level forecast allocated down by recent share is the intended fallback.

---

## 10. Future improvements

**Rolling-origin validation, first and by some distance.** Everything here rests on one December
fortnight, and the validation ranking has already contradicted the test ranking three times. Six to
twelve origins spread across seasons, selecting on mean WAPE with the spread reported, would turn
"the Transformer wins by 0.0008" into a defensible statement.

**Remove recursion from the tree path.** The tree models forecast recursively, so errors can compound
along the horizon. Training a separate head per horizon day, each predicting from origin-time
features only, removes the feedback loop entirely. The sequence models already work this way. Worth
measuring rather than assuming, since the per-horizon error curve is currently flat.

**Attack the per-series spread rather than the average.** Per-series WAPE runs from 0.11 to 0.69. One
detail points the way: the same SKU appears among both the best and worst series in different stores,
so the difficulty is a store-by-SKU property, which argues for store-level segmentation over
SKU-level.

**More signal, not more capacity.** Nothing overfits, so returns are in features: promotional depth
and lead/lag effects around campaigns (promotions currently enter as a single binary flag despite
carrying double the error), price relative to competing SKUs in the same subcategory, and a richer
calendar covering paydays and holiday proximity rather than a same-day flag.

**Test the stockout correction.** Three percent of rows still train on censored sales as though they
were demand. Both correction strategies are built and leakage-tested but were never compared.

**Move to quantile forecasts.** Inventory needs a service level, not a mean. Forecasting the median
and 90th percentile would make the output directly usable for safety-stock sizing and turn the
asymmetric cost proxy from descriptive into actionable.

**Establish a seasonal-naive baseline.** I never did, which means I cannot currently state how much
these models beat "the same weekday last week" — the first question any reviewer will ask.

**Other architectures worth a fair test.** N-BEATS and N-HiTS for pure seasonality, the Temporal
Fusion Transformer for native known-future handling and interpretable attention, and DeepAR for
probabilistic output.

**What is not worth doing:** a larger Transformer, or a wider hyperparameter sweep. The second run
already showed neither moves the number.
