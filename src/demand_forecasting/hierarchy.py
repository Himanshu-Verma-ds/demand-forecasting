from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_bottom_up(bottom_forecasts: pd.DataFrame, levels: list[str], cfg: dict, pred_col: str = "prediction") -> pd.DataFrame:
    """Aggregate bottom-level forecasts upward so parent forecasts equal the sum of child forecasts."""
    date_col = cfg["data"]["date_col"]
    required_columns = [date_col, *levels, pred_col]
    missing_columns = [column for column in required_columns if column not in bottom_forecasts.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return bottom_forecasts.groupby([date_col, *levels], as_index=False, dropna=False)[pred_col].sum()


def trailing_shares(history: pd.DataFrame, child_cols: list[str], parent_cols: list[str], cfg: dict) -> pd.DataFrame:
    """Estimate causal child allocation shares from configurable trailing historical demand."""
    date_col = cfg["data"]["date_col"]
    hierarchy_cfg = cfg["hierarchy"]

    window_days = int(hierarchy_cfg["share_window_days"])
    target_col = hierarchy_cfg["share_target_col"]
    exclude_stockouts = bool(hierarchy_cfg["exclude_stockouts_for_shares"])

    if window_days <= 0:
        raise ValueError("share_window_days must be greater than 0")

    if not parent_cols:
        raise ValueError("parent_cols cannot be empty")

    if not set(parent_cols).issubset(child_cols):
        raise ValueError("parent_cols must be contained within child_cols")

    required_columns = [date_col, target_col, *child_cols]

    if exclude_stockouts:
        required_columns.append("stock_out_flag")

    missing_columns = [column for column in required_columns if column not in history.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if history.empty:
        raise ValueError("History dataframe cannot be empty")

    h = history.copy()
    h[date_col] = pd.to_datetime(h[date_col])

    end = h[date_col].max()
    start = end - pd.Timedelta(days=window_days - 1)

    children = h[child_cols].drop_duplicates().copy()
    recent = h[(h[date_col] >= start) & (h[date_col] <= end)].copy()

    if exclude_stockouts:
        recent = recent[recent["stock_out_flag"].eq(0)].copy()

    recent_child_demand = recent.groupby(child_cols, as_index=False, dropna=False)[target_col].sum().rename(columns={target_col: "child_demand"})
    shares = children.merge(recent_child_demand, on=child_cols, how="left")
    shares["child_demand"] = shares["child_demand"].fillna(0.0)

    shares["parent_demand"] = shares.groupby(parent_cols, dropna=False)["child_demand"].transform("sum")
    shares["child_count"] = shares.groupby(parent_cols, dropna=False)["child_demand"].transform("size").clip(lower=1)

    shares["share"] = np.where(shares["parent_demand"] > 0, shares["child_demand"] / shares["parent_demand"], 1.0 / shares["child_count"])

    share_sum = shares.groupby(parent_cols, dropna=False)["share"].transform("sum")
    shares["share"] = shares["share"] / share_sum.replace(0, np.nan)

    if shares["share"].isna().any():
        raise ValueError("Unable to calculate valid hierarchy allocation shares")

    parent_share_check = shares.groupby(parent_cols, dropna=False)["share"].sum()

    if not np.allclose(parent_share_check.to_numpy(float), 1.0, atol=1e-8):
        raise ValueError("Hierarchy shares do not sum to 1 within each parent")

    return shares[child_cols + ["share"]]


def middle_out_allocate(parent_forecast: pd.DataFrame, shares: pd.DataFrame, parent_cols: list[str], child_cols: list[str], cfg: dict, pred_col: str = "prediction") -> pd.DataFrame:
    """Allocate parent forecasts to children using trailing causal shares while preserving parent totals."""
    date_col = cfg["data"]["date_col"]

    if not set(parent_cols).issubset(child_cols):
        raise ValueError("parent_cols must be contained within child_cols")

    forecast_required = [date_col, *parent_cols, pred_col]
    share_required = [*child_cols, "share"]

    missing_forecast_columns = [column for column in forecast_required if column not in parent_forecast.columns]
    missing_share_columns = [column for column in share_required if column not in shares.columns]

    if missing_forecast_columns:
        raise ValueError(f"Missing parent forecast columns: {missing_forecast_columns}")

    if missing_share_columns:
        raise ValueError(f"Missing share columns: {missing_share_columns}")

    if shares.duplicated(subset=child_cols).any():
        raise ValueError("shares must contain exactly one row per child")

    # The parent forecast holds one row per (date, parent), so the parent keys repeat across the
    # horizon. Only the share table needs to be unique, which is checked above; the merge itself
    # is legitimately many-to-many on parent_cols.
    out = parent_forecast.merge(shares, on=parent_cols, how="left", validate="many_to_many")

    if out["share"].isna().any():
        missing_parents = out.loc[out["share"].isna(), parent_cols].drop_duplicates().to_dict("records")
        raise ValueError(f"No child allocation shares available for parent(s): {missing_parents}")

    out[pred_col] = out[pred_col].astype(float) * out["share"].astype(float)

    allocated = out.groupby([date_col, *parent_cols], as_index=False, dropna=False)[pred_col].sum()
    expected = parent_forecast.groupby([date_col, *parent_cols], as_index=False, dropna=False)[pred_col].sum()

    check = expected.merge(allocated, on=[date_col, *parent_cols], suffixes=("_parent", "_children"), validate="one_to_one")

    if not np.allclose(check[f"{pred_col}_parent"].to_numpy(float), check[f"{pred_col}_children"].to_numpy(float), rtol=1e-7, atol=1e-8):
        raise ValueError("Allocated child forecasts do not sum back to parent forecasts")

    return out[[date_col, *child_cols, pred_col]]