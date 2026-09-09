from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


EPS = 1e-8


def _validate_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert actual and predicted values to numeric arrays."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.size == 0:
        raise ValueError("y_true and y_pred cannot be empty")

    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains NaN or infinite values")

    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred contains NaN or infinite values")

    return y_true, y_pred


def wape(y_true, y_pred) -> float:
    """Calculate Weighted Absolute Percentage Error."""
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    return float(np.abs(y_true - y_pred).sum() / max(np.abs(y_true).sum(), EPS))


def mape(y_true, y_pred) -> float:
    """Calculate Mean Absolute Percentage Error over rows with non-zero actuals.

    MAPE divides by each actual, so zero-demand rows are undefined and are excluded. Use
    mape_coverage() to see how many rows that removed. On this dataset the exclusion is small
    (~0.3% of rows) but MAPE remains unstable on low-demand rows: predicting 3 when the actual
    is 1 registers as a 200% error. WAPE is the more robust aggregate and stays the recommended
    selection metric.
    """
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    mask = np.abs(y_true) > EPS

    if not mask.any():
        return float("nan")

    return float((np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])).mean())


def mape_coverage(y_true) -> float:
    """Fraction of rows that MAPE is actually computed on (those with a non-zero actual)."""
    y_true = np.asarray(y_true, dtype=float)

    if y_true.size == 0:
        return float("nan")

    return float((np.abs(y_true) > EPS).mean())


def smape(y_true, y_pred) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    ratio = np.divide(2.0 * np.abs(y_true - y_pred), denominator, out=np.zeros_like(y_true), where=denominator > EPS)
    return float(ratio.mean())


def forecast_bias(y_true, y_pred) -> float:
    """Calculate normalized forecast bias where positive values indicate overforecasting."""
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    return float((y_pred - y_true).sum() / max(np.abs(y_true).sum(), EPS))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate common forecasting regression metrics."""
    y_true, y_pred = _validate_arrays(y_true, y_pred)

    return {
        "wape": wape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mape_coverage": mape_coverage(y_true),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "smape": smape(y_true, y_pred),
        "bias": forecast_bias(y_true, y_pred),
    }


def business_proxy(df: pd.DataFrame, y_col: str = "units_sold", pred_col: str = "prediction") -> dict[str, float]:
    """Estimate asymmetric business exposure from underforecasting and overforecasting."""
    required_columns = {y_col, pred_col, "list_price", "purchase_cost"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns for business proxy: {sorted(missing_columns)}")

    y, p = _validate_arrays(df[y_col], df[pred_col])

    list_price = df["list_price"].to_numpy(dtype=float)
    purchase_cost = df["purchase_cost"].to_numpy(dtype=float)

    if not np.isfinite(list_price).all():
        raise ValueError("list_price contains NaN or infinite values")

    if not np.isfinite(purchase_cost).all():
        raise ValueError("purchase_cost contains NaN or infinite values")

    under = np.maximum(y - p, 0.0)
    over = np.maximum(p - y, 0.0)

    unit_margin = np.maximum(list_price - purchase_cost, 0.0)

    return {
        "underforecast_margin_risk": float((under * unit_margin).sum()),
        "overforecast_purchase_cost": float((over * purchase_cost).sum()),
        "underforecast_unit_rate": float(under.sum() / max(np.abs(y).sum(), EPS)),
        "overforecast_unit_rate": float(over.sum() / max(np.abs(y).sum(), EPS)),
    }


def grouped_metrics(df: pd.DataFrame, group_col: str, y_col: str = "units_sold", pred_col: str = "prediction") -> pd.DataFrame:
    """Calculate forecasting metrics independently for each group."""
    required_columns = {group_col, y_col, pred_col}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    rows = []

    for key, group in df.groupby(group_col, dropna=False):
        metrics = regression_metrics(group[y_col], group[pred_col])
        rows.append({group_col: key, "n": len(group), **metrics})

    return pd.DataFrame(rows).sort_values("wape", ascending=False).reset_index(drop=True)


def full_evaluation(df: pd.DataFrame, y_col: str = "units_sold", pred_col: str = "prediction") -> dict[str, object]:
    """Run overall, business, and grouped forecasting evaluation."""
    if y_col not in df.columns:
        raise ValueError(f"Missing target column: {y_col}")

    if pred_col not in df.columns:
        raise ValueError(f"Missing prediction column: {pred_col}")

    overall = regression_metrics(df[y_col], df[pred_col])

    proxy = None

    if {"list_price", "purchase_cost"}.issubset(df.columns):
        proxy = business_proxy(df, y_col=y_col, pred_col=pred_col)

    breakdowns = {}

    for group_col in ["channel", "category", "promo_flag"]:
        if group_col in df.columns:
            breakdowns[group_col] = grouped_metrics(df, group_col, y_col, pred_col).to_dict("records")

    return {
        "overall": overall,
        "business_proxy": proxy,
        "breakdowns": breakdowns,
    }