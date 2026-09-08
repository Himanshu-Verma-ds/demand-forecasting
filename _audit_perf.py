import time

from demand_forecasting.config import load_config
from demand_forecasting.data import read_raw, prepare_features
from demand_forecasting.features import build_causal_features, ml_feature_columns
from demand_forecasting.inference_ml import recursive_forecast
from demand_forecasting.models.ml import build_ml_bundle
from demand_forecasting.splits import make_temporal_split

cfg = load_config("_audit/config.yaml")
DATE = cfg["data"]["date_col"]

t = time.time(); raw = read_raw(cfg["data"]["raw_path"], cfg); print(f"read_raw            {time.time()-t:6.2f}s  rows={len(raw)}")
split = make_temporal_split(raw, cfg)

t = time.time(); feat = prepare_features(raw, cfg); print(f"prepare_features    {time.time()-t:6.2f}s")
numeric, categorical = ml_feature_columns(feat, cfg)
cols = numeric + categorical

train = feat[feat[DATE] <= split.train_end].dropna(subset=["demand_lag_28"]).copy()
print(f"train rows={len(train)} features={len(cols)}")

t = time.time()
bundle = build_ml_bundle(train, "lightgbm", {"n_estimators": 700, "learning_rate": 0.04, "num_leaves": 63}, cfg, n_jobs=cfg["training"]["n_jobs"])
bundle.pipeline.fit(train[cols], train["demand_target"], model__sample_weight=train["sample_weight"])
print(f"fit                 {time.time()-t:6.2f}s")

history = raw[raw[DATE] <= split.train_end].copy()
val = raw[(raw[DATE] >= split.val_start) & (raw[DATE] <= split.val_end)].copy()

t = time.time()
combined = history.copy()
combined = prepare_features(combined, cfg)
print(f"one feature build   {time.time()-t:6.2f}s  (recursive_forecast does this 14x)")

t = time.time(); pred = recursive_forecast(bundle, history, val.copy(), cfg); print(f"recursive_forecast  {time.time()-t:6.2f}s  rows={len(pred)}")
print()
print("=> one Optuna trial ~= fit + recursive_forecast")
