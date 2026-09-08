from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import build_causal_features
from .stockout import add_stockout_target


def read_raw(path: str | Path, cfg: dict) -> pd.DataFrame:
    """Load and chronologically sort the raw forecasting dataset."""
    date_col = cfg["data"]["date_col"]
    series_cols = cfg["data"]["series_cols"]

    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values([*series_cols, date_col]).reset_index(drop=True)

    return df


def prepare_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create the configured training target and leakage-safe forecasting features."""
    df = add_stockout_target(df, cfg)
    return build_causal_features(df, cfg, demand_col="demand_target")