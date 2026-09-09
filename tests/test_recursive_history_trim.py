"""The history trim in recursive_forecast must not change a single prediction.

Trimming is a memory/speed optimisation on the leakage-critical path, so it is pinned by an
equivalence test rather than trusted: forecasts built from the trimmed history must match
forecasts built from the full history exactly.
"""
import numpy as np
import pandas as pd
import pytest

from demand_forecasting import inference_ml
from demand_forecasting.features import ml_feature_columns
from demand_forecasting.inference_ml import recursive_forecast, required_history_days
from demand_forecasting.models.ml import build_ml_bundle


def make_history(days: int = 400, series: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=days)
    rows = []

    for s in range(series):
        store, sku = f"S{s}", f"K{s}"
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "store_id": store,
                "sku_id": sku,
                "units_sold": float(40 + 10 * np.sin(i / 7) + rng.integers(0, 8) + 5 * s),
                "stock_out_flag": 0,
                "stock_on_hand": 200,
                "list_price": 10.0,
                "discount_pct": 0.0,
                "promo_flag": int(i % 23 == 0),
                "temperature": 15.0,
                "rain_mm": 0.5,
                "is_holiday": 0,
                "country": "X", "city": "Y", "channel": "Store",
                "category": "C", "subcategory": "SC", "brand": "B",
                "purchase_cost": 5.0, "margin_pct": 0.5,
            })

    return pd.DataFrame(rows)


def make_future(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    last = history["date"].max()
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon)
    keys = history[["store_id", "sku_id"]].drop_duplicates()
    future = keys.merge(pd.DataFrame({"date": dates}), how="cross")

    for column, value in [
        ("units_sold", np.nan), ("stock_out_flag", np.nan), ("stock_on_hand", np.nan),
        ("list_price", 10.0), ("discount_pct", 0.0), ("promo_flag", 0),
        ("temperature", 15.0), ("rain_mm", 0.5), ("is_holiday", 0),
        ("country", "X"), ("city", "Y"), ("channel", "Store"),
        ("category", "C"), ("subcategory", "SC"), ("brand", "B"),
        ("purchase_cost", 5.0), ("margin_pct", 0.5),
    ]:
        future[column] = value

    return future


def fit_bundle(history: pd.DataFrame, cfg: dict):
    from demand_forecasting.data import prepare_features

    feat = prepare_features(history, cfg)
    numeric, categorical = ml_feature_columns(feat, cfg)
    cols = numeric + categorical
    train = feat.dropna(subset=["demand_lag_28"]).copy()

    bundle = build_ml_bundle(train, "lightgbm", {"n_estimators": 40, "num_leaves": 15}, cfg, n_jobs=2)
    bundle.pipeline.fit(train[cols], train["demand_target"], model__sample_weight=train["sample_weight"])

    return bundle


def test_required_history_days_covers_every_configured_window(cfg):
    needed = required_history_days(cfg)

    assert needed >= max(cfg["features"]["demand_lags"])
    assert needed >= max(cfg["features"]["demand_roll_windows"])
    assert needed >= cfg["features"]["stock_roll_window"]
    assert needed >= 28


def test_trimmed_history_gives_identical_forecasts(cfg, monkeypatch):
    """The whole point: trimming must be invisible in the output."""
    history = make_history()
    future = make_future(history, cfg["data"]["horizon"])
    bundle = fit_bundle(history, cfg)

    trimmed = recursive_forecast(bundle, history, future.copy(), cfg)

    # Force the trim window to exceed the history so nothing is dropped.
    monkeypatch.setattr(inference_ml, "required_history_days", lambda _cfg: 10_000)
    untrimmed = recursive_forecast(bundle, history, future.copy(), cfg)

    keys = ["date", "store_id", "sku_id"]
    merged = trimmed[keys + ["prediction"]].merge(
        untrimmed[keys + ["prediction"]], on=keys, suffixes=("_trim", "_full")
    )

    assert len(merged) == len(future)
    np.testing.assert_allclose(
        merged["prediction_trim"].to_numpy(float),
        merged["prediction_full"].to_numpy(float),
        rtol=0, atol=0,
    )


def test_series_age_days_survives_trimming(cfg, monkeypatch):
    """series_age_days is a running counter, so the dropped rows must be added back."""
    history = make_history(days=300)
    future = make_future(history, cfg["data"]["horizon"])
    bundle = fit_bundle(history, cfg)

    trimmed = recursive_forecast(bundle, history, future.copy(), cfg)

    monkeypatch.setattr(inference_ml, "required_history_days", lambda _cfg: 10_000)
    untrimmed = recursive_forecast(bundle, history, future.copy(), cfg)

    keys = ["date", "store_id", "sku_id"]
    merged = trimmed[keys + ["series_age_days"]].merge(
        untrimmed[keys + ["series_age_days"]], on=keys, suffixes=("_trim", "_full")
    )

    pd.testing.assert_series_equal(
        merged["series_age_days_trim"].astype(float),
        merged["series_age_days_full"].astype(float),
        check_names=False,
    )
