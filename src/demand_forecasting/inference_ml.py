from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import read_raw
from .features import build_causal_features
from .logging_utils import log_run_context, setup_logging
from .models.ml import MLBundle
from .stockout import add_stockout_target


def required_history_days(cfg: dict) -> int:
    """Days of history per series that build_causal_features actually needs.

    Every feature is a bounded lag or rolling window, so history older than the longest window
    cannot influence a forecast. Rebuilding features over the full multi-year history is therefore
    pure waste: on the full dataset it means a ~1.07M-row frame per horizon day instead of ~70k.
    """
    feature_cfg = cfg["features"]

    windows = [
        max(feature_cfg["demand_lags"]),
        max(feature_cfg["demand_roll_windows"]),
        max(feature_cfg["stockout_lags"]),
        int(feature_cfg["stock_roll_window"]),
        28,  # price / discount / promo trailing means are hardcoded to 28 days
    ]

    # Double the longest window plus one horizon as a safety margin for min_periods.
    return int(max(windows)) * 2 + int(cfg["data"]["horizon"])


def recursive_forecast(bundle: MLBundle, history: pd.DataFrame, future_covariates: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Forecast future dates recursively without using future target or unavailable covariate information.

    Each predicted horizon day is appended as synthetic demand history so later
    horizon days use earlier predictions rather than actual future demand.
    """
    data_cfg = cfg["data"]
    feature_cfg = cfg["features"]

    date_col = data_cfg["date_col"]
    series_cols = data_cfg["series_cols"]

    history = history.sort_values([*series_cols, date_col]).copy()
    history = add_stockout_target(history, cfg)

    # Trim to the window the features can actually see. series_age_days is a running counter,
    # so the rows dropped here are added back as a per-series offset after each feature build.
    warmup_days = required_history_days(cfg)
    full_counts = history.groupby(series_cols, sort=False, dropna=False).size()
    history = history.groupby(series_cols, sort=False, group_keys=False, dropna=False).tail(warmup_days)
    kept_counts = history.groupby(series_cols, sort=False, dropna=False).size()

    age_offset = (full_counts - kept_counts).rename("series_age_offset").reset_index()

    future = future_covariates.sort_values([*series_cols, date_col]).copy()
    future[date_col] = pd.to_datetime(future[date_col])

    if history.empty:
        raise ValueError("History dataframe cannot be empty")

    if future.empty:
        raise ValueError("Future dataframe cannot be empty")

    if pd.to_datetime(future[date_col]).min() <= pd.to_datetime(history[date_col]).max():
        raise ValueError("Future forecast dates must be after the final history date")

    forbidden_columns = ["units_sold", "demand_target", "gross_sales", "net_sales", "sample_weight", "stockout_imputed_amount", "stock_out_flag", "stock_on_hand"]

    for column in forbidden_columns:
        if column in future.columns:
            future[column] = np.nan

    if not feature_cfg["promo_known_future"] and "promo_flag" in future.columns:
        future["promo_flag"] = np.nan

    if not feature_cfg["price_known_future"]:
        for column in ["list_price", "discount_pct"]:
            if column in future.columns:
                future[column] = np.nan

    if not feature_cfg["weather_known_future"]:
        for column in ["temperature", "rain_mm"]:
            if column in future.columns:
                future[column] = np.nan

    outputs = []

    for forecast_date in sorted(future[date_col].dropna().unique()):
        day = future[future[date_col].eq(forecast_date)].copy()

        day["units_sold"] = np.nan
        day["demand_target"] = np.nan
        day["stock_out_flag"] = np.nan
        day["stock_on_hand"] = np.nan
        day["sample_weight"] = 1.0
        day["stockout_imputed_amount"] = 0.0

        combined = pd.concat([history, day], ignore_index=True, sort=False)
        features = build_causal_features(combined, cfg, demand_col="demand_target")

        xday = features[pd.to_datetime(features[date_col]).eq(pd.Timestamp(forecast_date))].copy()

        if xday.empty:
            raise ValueError(f"No feature rows generated for forecast date {pd.Timestamp(forecast_date).date()}")

        # Restore the true series age that trimming removed.
        xday = xday.merge(age_offset, on=series_cols, how="left", validate="many_to_one")
        xday["series_age_days"] = xday["series_age_days"] + xday["series_age_offset"].fillna(0)
        xday = xday.drop(columns=["series_age_offset"])

        predictions = bundle.predict(xday)
        xday["prediction"] = predictions
        outputs.append(xday)

        synthetic_history = day.copy()
        synthetic_history["units_sold"] = predictions
        synthetic_history["demand_target"] = predictions

        history = pd.concat([history, synthetic_history], ignore_index=True, sort=False)
        history = history.sort_values([*series_cols, date_col]).reset_index(drop=True)

    if not outputs:
        raise ValueError("No forecasts were generated")

    return pd.concat(outputs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Generate recursive leakage-safe ML forecasts.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--future", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output", default="artifacts/forecast_ml.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("inference_ml", cfg)

    log_run_context(
        logger,
        "inference_ml",
        config_path=args.config,
        model=args.model,
        history=args.history,
        future=args.future,
        output=args.output,
        horizon=cfg["data"]["horizon"],
        promo_known_future=cfg["features"]["promo_known_future"],
        price_known_future=cfg["features"]["price_known_future"],
        weather_known_future=cfg["features"]["weather_known_future"],
    )

    model = MLBundle.load(args.model)

    history = read_raw(args.history, cfg)
    future = read_raw(args.future, cfg)

    output = recursive_forecast(model, history, future, cfg)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output[[cfg["data"]["date_col"], *cfg["data"]["series_cols"], "prediction"]].to_csv(args.output, index=False)

    logger.info("Wrote %s forecast rows to %s", f"{len(output):,}", args.output)


if __name__ == "__main__":
    main()