"""Empirical leakage probes.

The unit tests only check that build_causal_features is causal. These probes attack the
end-to-end forecast paths: if a model's forecast changes when we corrupt information that
is genuinely unknown at the forecast origin, that information is leaking.
"""
import numpy as np
import pandas as pd

from demand_forecasting.config import load_config
from demand_forecasting.data import read_raw, prepare_features
from demand_forecasting.features import ml_feature_columns
from demand_forecasting.inference_ml import recursive_forecast
from demand_forecasting.models.ml import build_ml_bundle
from demand_forecasting.splits import make_temporal_split

CFG = load_config("_audit/config.yaml")
DATE = CFG["data"]["date_col"]
SERIES = CFG["data"]["series_cols"]
TARGET = CFG["data"]["target_col"]

raw = read_raw(CFG["data"]["raw_path"], CFG)
split = make_temporal_split(raw, CFG)
feat = prepare_features(raw, CFG)
numeric, categorical = ml_feature_columns(feat, CFG)
cols = numeric + categorical

train = feat[feat[DATE] <= split.train_end].dropna(subset=["demand_lag_28"]).copy()
bundle = build_ml_bundle(train, "lightgbm", {"n_estimators": 60, "learning_rate": 0.1}, CFG, n_jobs=-1)
bundle.pipeline.fit(train[cols], train["demand_target"], model__sample_weight=train["sample_weight"])

history = raw[raw[DATE] <= split.train_end].copy()
val = raw[(raw[DATE] >= split.val_start) & (raw[DATE] <= split.val_end)].copy()

print("=" * 70)
print("PROBE 1: does the tree forecast depend on future TRUE demand?")
base = recursive_forecast(bundle, history, val.copy(), CFG)

corrupt = val.copy()
corrupt[TARGET] = corrupt[TARGET] * 1000 + 12345          # destroy the future labels
corrupt["gross_sales"] = corrupt["gross_sales"] * 999
corrupt["net_sales"] = corrupt["net_sales"] * 999
c1 = recursive_forecast(bundle, history, corrupt, CFG)

key = [DATE, *SERIES]
m = base[key + ["prediction"]].merge(c1[key + ["prediction"]], on=key, suffixes=("_base", "_corrupt"))
d = (m["prediction_base"] - m["prediction_corrupt"]).abs().max()
print(f"  rows compared      : {len(m)}")
print(f"  max |Δprediction|  : {d:.10f}")
print(f"  VERDICT            : {'PASS - no future-label leakage' if d == 0 else 'FAIL - LEAK'}")

print("=" * 70)
print("PROBE 2: does it depend on future stock_out_flag / stock_on_hand?")
c2f = val.copy()
c2f["stock_out_flag"] = 1 - c2f["stock_out_flag"]
c2f["stock_on_hand"] = 0
c2 = recursive_forecast(bundle, history, c2f, CFG)
m2 = base[key + ["prediction"]].merge(c2[key + ["prediction"]], on=key, suffixes=("_b", "_c"))
d2 = (m2["prediction_b"] - m2["prediction_c"]).abs().max()
print(f"  max |Δprediction|  : {d2:.10f}")
print(f"  VERDICT            : {'PASS - future inventory not used' if d2 == 0 else 'FAIL - LEAK'}")

print("=" * 70)
print("PROBE 3: config says price/weather are NOT known -> must not matter")
c3f = val.copy()
c3f["list_price"] = c3f["list_price"] * 5
c3f["discount_pct"] = 0.9
c3f["temperature"] = -50
c3f["rain_mm"] = 999
c3 = recursive_forecast(bundle, history, c3f, CFG)
m3 = base[key + ["prediction"]].merge(c3[key + ["prediction"]], on=key, suffixes=("_b", "_c"))
d3 = (m3["prediction_b"] - m3["prediction_c"]).abs().max()
print(f"  price_known_future={CFG['features']['price_known_future']} weather_known_future={CFG['features']['weather_known_future']}")
print(f"  max |Δprediction|  : {d3:.10f}")
print(f"  VERDICT            : {'PASS - unknown covariates ignored' if d3 == 0 else 'FAIL - LEAK'}")

print("=" * 70)
print("PROBE 4: promo IS declared known -> changing it SHOULD move the forecast")
c4f = val.copy()
c4f["promo_flag"] = 1 - c4f["promo_flag"]
c4 = recursive_forecast(bundle, history, c4f, CFG)
m4 = base[key + ["prediction"]].merge(c4[key + ["prediction"]], on=key, suffixes=("_b", "_c"))
d4 = (m4["prediction_b"] - m4["prediction_c"]).abs().max()
print(f"  promo_known_future={CFG['features']['promo_known_future']}")
print(f"  max |Δprediction|  : {d4:.6f}")
print(f"  VERDICT            : {'PASS - promo is actually used' if d4 > 0 else 'FAIL - promo declared known but ignored'}")

print("=" * 70)
print("PROBE 5: recursion - does day D+2 actually consume day D+1's PREDICTION?")
one = val[val[DATE] <= split.val_start + pd.Timedelta(days=1)].copy()
sub = recursive_forecast(bundle, history, one, CFG)
d1 = sub[sub[DATE] == split.val_start].sort_values(SERIES)
d2r = sub[sub[DATE] == split.val_start + pd.Timedelta(days=1)].sort_values(SERIES)
lag_matches = np.allclose(d2r["demand_lag_1"].to_numpy(float), d1["prediction"].to_numpy(float))
print(f"  D+2 demand_lag_1 == D+1 prediction : {lag_matches}")
true_d1 = val[val[DATE] == split.val_start].sort_values(SERIES)[TARGET].to_numpy(float)
equals_truth = np.allclose(d2r["demand_lag_1"].to_numpy(float), true_d1)
print(f"  D+2 demand_lag_1 == D+1 ACTUAL     : {equals_truth}  (must be False)")
print(f"  VERDICT            : {'PASS - genuine recursion' if lag_matches and not equals_truth else 'FAIL'}")

print("=" * 70)
print("PROBE 6: blocked columns absent from the model input")
blocked = ["gross_sales", "net_sales", "units_sold", "demand_target", "sample_weight",
           "stock_out_flag", "stock_on_hand", "stockout_imputed_amount"]
present = [c for c in blocked if c in cols]
print(f"  feature count      : {len(cols)}")
print(f"  blocked but present: {present}")
print(f"  VERDICT            : {'PASS' if not present else 'FAIL'}")
