from __future__ import annotations

import numpy as np
import pandas as pd


VALID_STOCKOUT_TARGET_MODES = {"none", "rolling_median", "percentage"}


def add_stockout_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create the training target and sample weights for stockout observations.

    Supported strategies are controlled by config["features"]["stockout_target_mode"]:

    none:
        Keep observed units_sold unchanged for all rows.

    rolling_median:
        Estimate stockout-day demand using the rolling median of prior
        non-stockout demand for the same Store-SKU series.

    percentage:
        Increase observed stockout-day sales by a configured percentage.

    The function always creates demand_target, sample_weight, and
    stockout_imputed_amount so downstream training code remains unchanged
    when the stockout strategy is switched.
    """
    features_cfg = config["features"]
    data_cfg = config["data"]

    series_cols = data_cfg["series_cols"]
    target_col = data_cfg["target_col"]
    date_col = data_cfg["date_col"]

    mode = features_cfg.get("stockout_target_mode", "none")
    window = features_cfg.get("stockout_imputation_window", 56)
    min_periods = features_cfg.get("stockout_imputation_min_periods", 7)
    percentage_uplift = features_cfg.get("stockout_percentage_uplift", 0.50)
    stockout_weight = features_cfg.get("stockout_sample_weight", 0.50)

    if mode not in VALID_STOCKOUT_TARGET_MODES:
        raise ValueError(f"Invalid stockout_target_mode: {mode}. Expected one of {sorted(VALID_STOCKOUT_TARGET_MODES)}")

    required_columns = [*series_cols, date_col, target_col, "stock_out_flag"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if window <= 0:
        raise ValueError("stockout_imputation_window must be greater than 0")

    if min_periods <= 0:
        raise ValueError("stockout_imputation_min_periods must be greater than 0")

    if min_periods > window:
        raise ValueError("stockout_imputation_min_periods cannot be greater than stockout_imputation_window")

    if percentage_uplift < 0:
        raise ValueError("stockout_percentage_uplift cannot be negative")

    if not 0 <= stockout_weight <= 1:
        raise ValueError("stockout_sample_weight must be between 0 and 1")

    out = df.sort_values([*series_cols, date_col]).copy()
    observed = out[target_col].astype(float)
    stockout_mask = out["stock_out_flag"].eq(1)

    out["demand_target"] = observed
    out["sample_weight"] = 1.0

    if mode == "rolling_median":
        nonstock_demand = observed.where(~stockout_mask)
        group_keys = [out[column] for column in series_cols]

        rolling_baseline = nonstock_demand.groupby(group_keys, sort=False).transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).median()
        )

        expanding_baseline = nonstock_demand.groupby(group_keys, sort=False).transform(
            lambda s: s.shift(1).expanding(min_periods=1).median()
        )

        nonstock_values = observed.loc[~stockout_mask]

        if nonstock_values.empty:
            raise ValueError("Cannot calculate stockout baseline because all observations are marked as stockouts")

        global_prior = float(nonstock_values.median())
        baseline = rolling_baseline.fillna(expanding_baseline).fillna(global_prior)

        out["demand_target"] = np.where(stockout_mask, np.maximum(observed, baseline), observed)
        out.loc[stockout_mask, "sample_weight"] = stockout_weight

    elif mode == "percentage":
        out["demand_target"] = np.where(stockout_mask, observed * (1.0 + percentage_uplift), observed)
        out.loc[stockout_mask, "sample_weight"] = stockout_weight

    out["demand_target"] = out["demand_target"].clip(lower=0).astype(float)
    out["stockout_imputed_amount"] = out["demand_target"] - observed

    return out


def add_stockout_adjusted_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Backward-compatible wrapper for add_stockout_target()."""
    return add_stockout_target(df=df, config=config)