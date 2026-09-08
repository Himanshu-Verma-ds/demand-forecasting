from __future__ import annotations

import numpy as np
import pandas as pd


LEAKAGE_COLUMNS = ["gross_sales", "net_sales"]
STATIC_CATEGORICAL = ["store_id", "sku_id", "country", "city", "channel", "category", "subcategory", "brand"]


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Create deterministic calendar features directly from the date column."""
    out = df.copy()
    d = pd.to_datetime(out[date_col])

    out["year"] = d.dt.year
    out["month"] = d.dt.month
    out["day"] = d.dt.day
    out["weekday"] = d.dt.dayofweek
    out["weekofyear"] = d.dt.isocalendar().week.astype(int)
    out["is_weekend"] = d.dt.dayofweek.isin([5, 6]).astype(int)

    out["dow_sin"] = np.sin(2 * np.pi * d.dt.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * d.dt.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (d.dt.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (d.dt.month - 1) / 12)
    out["doy_sin"] = np.sin(2 * np.pi * (d.dt.dayofyear - 1) / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * (d.dt.dayofyear - 1) / 365.25)

    return out


def build_causal_features(df: pd.DataFrame, config: dict, demand_col: str = "demand_target") -> pd.DataFrame:
    """Create point-in-time-correct forecasting features using settings from config."""
    data_cfg = config["data"]
    feature_cfg = config["features"]

    series_cols = data_cfg["series_cols"]
    date_col = data_cfg["date_col"]
    demand_lags = feature_cfg["demand_lags"]
    roll_windows = feature_cfg["demand_roll_windows"]
    stockout_lags = feature_cfg["stockout_lags"]
    stock_roll_window = feature_cfg["stock_roll_window"]
    promo_known_future = feature_cfg["promo_known_future"]
    price_known_future = feature_cfg["price_known_future"]
    weather_known_future = feature_cfg["weather_known_future"]

    required_columns = [*series_cols, date_col, demand_col, "stock_out_flag", "stock_on_hand", "promo_flag", "list_price", "discount_pct"]

    if weather_known_future:
        required_columns.extend(["temperature", "rain_mm"])

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    out = df.sort_values([*series_cols, date_col]).copy()
    out = add_calendar_features(out, date_col=date_col)

    g = out.groupby(series_cols, sort=False)

    for lag in demand_lags:
        out[f"demand_lag_{lag}"] = g[demand_col].shift(lag)

    for window in roll_windows:
        out[f"demand_roll_mean_{window}"] = g[demand_col].transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
        out[f"demand_roll_std_{window}"] = g[demand_col].transform(lambda s: s.shift(1).rolling(window, min_periods=3).std())
        out[f"demand_roll_max_{window}"] = g[demand_col].transform(lambda s: s.shift(1).rolling(window, min_periods=3).max())

    for lag in stockout_lags:
        out[f"stockout_lag_{lag}"] = g["stock_out_flag"].shift(lag)

    stock_min_periods = min(7, stock_roll_window)
    out[f"stockout_rate_{stock_roll_window}"] = g["stock_out_flag"].transform(lambda s: s.shift(1).rolling(stock_roll_window, min_periods=stock_min_periods).mean())

    out["stock_on_hand_lag_1"] = g["stock_on_hand"].shift(1)
    out["stock_on_hand_mean_7"] = g["stock_on_hand"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())

    out["list_price_lag_1"] = g["list_price"].shift(1)
    out["discount_pct_lag_1"] = g["discount_pct"].shift(1)

    prior_price_28 = g["list_price"].transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())
    prior_discount_28 = g["discount_pct"].transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())

    out["list_price_mean_28"] = prior_price_28
    out["discount_pct_mean_28"] = prior_discount_28

    if price_known_future:
        out["price_vs_28d_mean"] = out["list_price"] / prior_price_28.replace(0, np.nan) - 1.0
        out["discount_change_1"] = out["discount_pct"] - g["discount_pct"].shift(1)

    out["promo_prev_1"] = g["promo_flag"].shift(1)
    out["promo_rate_28"] = g["promo_flag"].transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())

    if not promo_known_future:
        out["promo_flag"] = np.nan

    if not price_known_future:
        out["list_price"] = np.nan
        out["discount_pct"] = np.nan

    if not weather_known_future:
        if "temperature" in out.columns:
            out["temperature"] = np.nan
        if "rain_mm" in out.columns:
            out["rain_mm"] = np.nan

    store_daily = out.groupby([date_col, "store_id"], as_index=False)[demand_col].mean().rename(columns={demand_col: "store_demand_mean_lag_1"})
    store_daily[date_col] = pd.to_datetime(store_daily[date_col]) + pd.Timedelta(days=1)
    out = out.merge(store_daily, on=[date_col, "store_id"], how="left")

    sku_daily = out.groupby([date_col, "sku_id"], as_index=False)[demand_col].mean().rename(columns={demand_col: "sku_demand_mean_lag_1"})
    sku_daily[date_col] = pd.to_datetime(sku_daily[date_col]) + pd.Timedelta(days=1)
    out = out.merge(sku_daily, on=[date_col, "sku_id"], how="left")

    out["series_age_days"] = out.groupby(series_cols, sort=False).cumcount()
    out["store_sku_id"] = out["store_id"].astype(str) + "__" + out["sku_id"].astype(str)

    return out


def ml_feature_columns(df: pd.DataFrame, config: dict) -> tuple[list[str], list[str]]:
    """Return leakage-safe numeric and categorical feature lists according to config."""
    data_cfg = config["data"]
    feature_cfg = config["features"]

    target_col = data_cfg["target_col"]
    series_cols = data_cfg["series_cols"]

    categoricals = [column for column in STATIC_CATEGORICAL + ["store_sku_id"] if column in df.columns]

    for column in series_cols:
        if column in df.columns and column not in categoricals:
            categoricals.append(column)

    blocked = {
        data_cfg["date_col"],
        target_col,
        "demand_target",
        "sample_weight",
        "stockout_imputed_amount",
        "stock_out_flag",
        "stock_on_hand",
        "purchase_cost",
        "margin_pct",
        "supplier_id",
        "sku_name",
        "latitude",
        "longitude",
        *LEAKAGE_COLUMNS,
    }

    if not feature_cfg["promo_known_future"]:
        blocked.add("promo_flag")

    if not feature_cfg["price_known_future"]:
        blocked.update({"list_price", "discount_pct", "price_vs_28d_mean", "discount_change_1"})

    if not feature_cfg["weather_known_future"]:
        blocked.update({"temperature", "rain_mm"})

    numeric = [column for column in df.columns if column not in blocked and column not in categoricals and pd.api.types.is_numeric_dtype(df[column])]

    return numeric, categoricals