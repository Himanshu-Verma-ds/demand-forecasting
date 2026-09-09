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

I went into the EDA with a list of things I needed settled before I could sensibly pick a model.
Most of the answers ended up as configuration values, which is why this table has three columns
rather than two. Working notebook: `notebooks/01_eda.ipynb`.

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

Two of these did most of the work.

**The autocorrelation picked the lags for me.** With clean peaks at 7, 14, 21 and 28 days, the lag
set almost writes itself: one day for short-run level, seven for the same weekday last week, then 14
and 28 to check the weekly signal holds and to cover a full monthly cycle. Rolling windows followed
the same logic at one, two and four weeks. The cyclical calendar terms are there for a duller reason:
as plain integers, Sunday sits six units from Monday and December twelve from January, which is
simply wrong.

Rather than pinning the sequence models to a fixed history window, I made it a tunable and restricted
the search to whole weeks. They came back with 42 days for the LSTM, 70 for the Transformer and 56
for TiDE. Every one is a multiple of seven. I did not force that, and I take it as a decent sign the
weekly structure is real and the models are actually leaning on it.

**The promo slice earned its place.** Promotions are about 7% of rows carrying roughly double the
demand. On validation they come in at 0.2755 WAPE against 0.2639 on ordinary days, which looks
harmless until you check absolute error: 31.0 units against 15.4. Promotions are the weakest part of
this model, and the headline number barely flinches.

---

## 2. How I prepared the dataset

Three steps: load and sort by series and date, build the demand target with stockout handling, then
build the point-in-time features. The same three run at inference, which is the only reliable way I
know to keep train/serve skew out — if the two paths are literally the same code, they cannot drift.

That gives 1.1M rows and 72 columns, of which roughly 52 numeric and 9 categorical actually reach the
models.

I kept the panel long instead of splitting it into a thousand separate series. Each row carries its
own store, SKU, channel, category, subcategory and brand, and one global model sees all 1,005 series
together. Every group-wise calculation is scoped to a single store-SKU pair, so nothing bleeds
between series.

### Historical features vs known-future covariates

This split is the contract the whole system rests on, so it is worth being explicit about.

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

The specific mistake I was watching for is an easy one to make. Compute lags across the whole table,
*then* filter out the validation rows, and December 5th ends up holding December 4th's actual demand
— which nobody has at a December 3rd origin. It does not throw an error. It just makes the model look
about twice as good as it is.

Unit tests only cover the feature builder, so rather than trust them I went after the end-to-end
forecast paths directly: corrupt something the model has no business knowing, and see whether the
forecast moves.

| What I corrupted | Result |
|---|---|
| Future actual demand, multiplied by 1000 | No change to any forecast |
| Future stockout flags and inventory | No change |
| Future price, discount, temperature, rainfall | No change |
| **Promotion flag** (declared known) | Moved by 151 units — correctly *does* influence the forecast |
| Recursion check: day two's one-day lag | Equals day one's prediction, not its actual |
| Stockout target built from train-only vs full data | 0 of 21,340 training targets changed |

The promotion row is the control, and it is the one that makes the rest mean anything. Six zero
deltas on their own would be equally consistent with a model that ignores every future input it is
given.

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

No k-fold here, deliberately. Random folds put future rows into training, and since the features are
lags and rolling windows, neighbouring rows are near-duplicates — a random split hands the model a
copy of almost every validation row. Once validation had frozen the configuration I refit on train
plus validation, then forecast the test window once from the new origin and left it alone.

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

The registry promoted the Transformer, because that is what won on validation. **I would ship
CatBoost instead**, and I think the case is fairly clear-cut.

1. **The accuracy gap is not real.** All six models sit between 0.2654 and 0.2762 on validation, so
   about four percent relative end to end. Transformer to CatBoost is 0.0065 of WAPE. I would not
   bet an architecture choice on that.
2. **Validation has already mispredicted test three times.** Run 1: Transformer won validation,
   CatBoost was best on test. Run 2: Transformer won validation, LSTM was best on test, and the
   Transformer came in worst of the five non-Darts models. A single 14-day December origin simply
   cannot separate models this close together, and I would rather say so than pretend the ranking is
   meaningful.
3. **Bias settles it.** The Transformer over-forecasts the test window by 7.3% — call it 64,000 units
   of demand that never existed. CatBoost sits at 1.0%, LightGBM at 0.86%. For anyone placing an
   order against this, that swamps a 0.006 WAPE edge.
4. **And it is far cheaper to live with.** CatBoost trains in roughly 25 seconds against 10 to 30
   minutes, tunes at 40 seconds a trial against 7 to 30 minutes, scores all 1,004 series on CPU in
   about 22 seconds, and gives SHAP explanations for free when someone inevitably asks why a number
   moved.

CatBoost is also the only model that is top-three on both validation and test *and* came out
byte-identical across the two runs, which counts for something when you have to operate it. LightGBM
is the obvious backup: last of six on validation WAPE, third on test, and the least biased model in
the study.

**The finding I would actually lead with is that the model barely matters here.** Six architectures
with very different inductive biases all land within four percent of each other. Train and validation
error are nearly identical, so nothing is overfitting. Doubling the search budget and tripling the
epoch ceiling changed nothing. And TiDE, which never got a tuned search at all, still finishes within
0.4% of the tuned Transformer on test. The ceiling here is the information in the features, not the
model sitting on top of them — which is where I would spend the next sprint.

---

## 9. Cold start and stockouts

**Stockouts, about 3% of rows.** On a stockout day the recorded number is what was *available to
sell*, not what customers wanted. Train on it as-is and you are teaching the model that demand
collapsed on exactly the days it may have spiked. I built three options behind one switch: leave the
target alone, take the higher of observed sales and a trailing non-stockout rolling median, or apply
a flat percentage uplift. Imputed rows get down-weighted so the model treats them as softer evidence.
All three emit the same columns, so switching strategy changes numbers and nothing else.

Both runs used the uncorrected target. Any correction here is a modelling assumption dressed up as a
fix, and I would rather ship the honest baseline and measure the correction as an experiment than
bake in a guess. I did not get to that experiment. It is the cheapest thing left on my list.

**Cold start.** There is barely any in this dataset — 1,004 of 1,005 series carry three full years —
but the design has to survive it anyway, because real assortments churn. The global model does most
of the work: a brand-new SKU inherits whatever the model already knows about its category,
subcategory, brand and channel, which is precisely what a per-series model cannot offer. Unseen
category values map to a reserved unknown index rather than throwing, and unseen identifiers in the
sequence models fall back to a learned unknown embedding. Where history is shorter than the longest
lag, the hierarchy and calendar features still carry signal. For a genuine launch with no history at
all, the intended fallback is a subcategory forecast allocated down by recent share.

---

## 10. Future improvements

**Rolling-origin validation, well ahead of anything else.** Every conclusion above hangs off a single
December fortnight, and validation has already contradicted test three times. Six to twelve origins
spread across the year, selecting on mean WAPE and reporting the spread, is what turns "the
Transformer wins by 0.0008" into something I would actually defend in a review. If I only get to do
one thing from this list, it is this.

**Take the recursion out of the tree path.** The tree models forecast recursively, so errors can
compound across the horizon. Training one head per horizon day, each predicting from origin-time
features only, removes the feedback loop outright — which is effectively how the sequence models
already operate. Worth measuring rather than assuming, mind: the per-horizon error curve is flat at
the moment, so recursion may not be costing much yet.

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

**Establish a seasonal-naive baseline.** I skipped this and I should not have. Without it I cannot
tell you how much any of these models beat "same weekday last week", and that is the first question
I would ask if someone handed me this report.

**Other architectures worth a fair test.** N-BEATS and N-HiTS for pure seasonality, the Temporal
Fusion Transformer for native known-future handling and interpretable attention, and DeepAR for
probabilistic output.

**What is not worth doing:** a larger Transformer, or a wider hyperparameter sweep. The second run
already showed neither moves the number.
