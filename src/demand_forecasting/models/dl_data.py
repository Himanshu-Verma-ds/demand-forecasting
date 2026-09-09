from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


BASE_PAST_NUMERIC = [
    "demand_target",
    "promo_flag",
    "list_price",
    "discount_pct",
    "temperature",
    "rain_mm",
    "stock_out_flag",
    "stock_on_hand",
    "is_holiday",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

BASE_FUTURE_NUMERIC = [
    "is_holiday",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

STATIC_COLS = ["store_id", "sku_id", "channel", "category", "subcategory", "brand"]


@dataclass
class DLMetadata:
    """Metadata required for scaling numeric features and encoding static categorical variables."""

    static_maps: dict[str, dict[str, int]]
    means: dict[str, float]
    stds: dict[str, float]
    target_mean: float
    target_std: float
    past_cols: list[str]
    future_cols: list[str]
    static_cols: list[str]


def get_dl_feature_columns(cfg: dict) -> tuple[list[str], list[str], list[str]]:
    """Return past, future, and static DL feature columns according to config."""
    feature_cfg = cfg["features"]

    past_cols = BASE_PAST_NUMERIC.copy()
    future_cols = BASE_FUTURE_NUMERIC.copy()

    if feature_cfg["promo_known_future"]:
        future_cols.append("promo_flag")

    if feature_cfg["price_known_future"]:
        future_cols.extend(["list_price", "discount_pct"])

    if feature_cfg["weather_known_future"]:
        future_cols.extend(["temperature", "rain_mm"])

    return past_cols, future_cols, STATIC_COLS.copy()


def fit_metadata(train: pd.DataFrame, cfg: dict) -> DLMetadata:
    """Fit scaling statistics and static mappings using training data only."""
    past_cols, future_cols, static_cols = get_dl_feature_columns(cfg)

    required_columns = [*past_cols, *future_cols, *static_cols, "demand_target", "sample_weight"]
    missing_columns = [column for column in required_columns if column not in train.columns]

    if missing_columns:
        raise ValueError(f"Missing required DL columns: {missing_columns}")

    if train.empty:
        raise ValueError("Training dataframe cannot be empty")

    static_maps = {}

    for column in static_cols:
        values = sorted(train[column].where(train[column].notna(), "__MISSING__").astype(str).unique())
        static_maps[column] = {value: index + 1 for index, value in enumerate(values)}

    numeric_cols = sorted(set(past_cols + future_cols))
    feature_numeric_cols = [column for column in numeric_cols if column != "demand_target"]

    means = {}
    stds = {}

    for column in feature_numeric_cols:
        numeric = pd.to_numeric(train[column], errors="coerce")
        mean = float(numeric.mean()) if numeric.notna().any() else 0.0
        std = float(numeric.std()) if numeric.notna().sum() > 1 else 1.0

        if not np.isfinite(mean):
            mean = 0.0

        if not np.isfinite(std) or std <= 0:
            std = 1.0

        means[column] = mean
        stds[column] = std

    target = pd.to_numeric(train["demand_target"], errors="coerce")

    if not target.notna().any():
        raise ValueError("demand_target contains no valid training values")

    target_mean = float(target.mean())
    target_std = float(target.std()) if target.notna().sum() > 1 else 1.0

    if not np.isfinite(target_std) or target_std <= 0:
        target_std = 1.0

    return DLMetadata(
        static_maps=static_maps,
        means=means,
        stds=stds,
        target_mean=target_mean,
        target_std=target_std,
        past_cols=past_cols,
        future_cols=future_cols,
        static_cols=static_cols,
    )


def _scale(frame: pd.DataFrame, meta: DLMetadata) -> pd.DataFrame:
    """Scale numeric features using training-only metadata."""
    out = frame.copy()

    out["demand_target"] = pd.to_numeric(out["demand_target"], errors="coerce")
    out["demand_target"] = (out["demand_target"] - meta.target_mean) / meta.target_std

    for column, mean in meta.means.items():
        out[column] = pd.to_numeric(out[column], errors="coerce")
        out[column] = (out[column] - mean) / meta.stds[column]
        out[column] = out[column].fillna(0.0)

    return out


class MultiSeriesWindowDataset(Dataset):
    """Shared-weight sequence dataset supporting both training and future inference.

    Returned shapes per sample:
      past_x:      [lookback, P]
      future_x:    [horizon, F]
      static_ids:  [S]
      y:           [horizon]
      weight:      [horizon]
    """

    def __init__(self, df: pd.DataFrame, meta: DLMetadata, cfg: dict, require_target: bool = True,
                 max_windows: int | None = None):
        data_cfg = cfg["data"]

        self.meta = meta
        self.lookback = int(data_cfg["lookback"])
        self.horizon = int(data_cfg["horizon"])
        self.series_cols = data_cfg["series_cols"]
        self.date_col = data_cfg["date_col"]
        self.require_target = require_target
        self.samples = []

        if self.lookback <= 0:
            raise ValueError("lookback must be greater than 0")

        if self.horizon <= 0:
            raise ValueError("horizon must be greater than 0")

        required_columns = [
            *self.series_cols,
            self.date_col,
            *meta.past_cols,
            *meta.future_cols,
            *meta.static_cols,
            "demand_target",
            "sample_weight",
        ]

        missing_columns = [column for column in required_columns if column not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required dataset columns: {missing_columns}")

        scaled = _scale(df, meta)

        for _, group in scaled.groupby(self.series_cols, sort=False, dropna=False):
            group = group.sort_values(self.date_col).reset_index(drop=True)
            n = len(group)

            if n < self.lookback + self.horizon:
                continue

            for end in range(self.lookback, n - self.horizon + 1):
                past = group.iloc[end - self.lookback:end]
                future = group.iloc[end:end + self.horizon]

                if past["demand_target"].isna().any():
                    continue

                if self.require_target and future["demand_target"].isna().any():
                    continue

                if self.require_target and future["sample_weight"].isna().any():
                    continue

                self.samples.append((group, end))

        # Consecutive windows overlap in 55 of 56 history days, so the full set is highly
        # redundant. Subsampling keeps every series represented while making training
        # tractable; it is applied to training windows only, never to evaluation windows.
        if max_windows and len(self.samples) > max_windows:
            rng = np.random.default_rng(int(cfg["project"]["random_seed"]))
            keep = rng.choice(len(self.samples), size=max_windows, replace=False)
            self.samples = [self.samples[i] for i in sorted(keep)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        group, end = self.samples[idx]

        past = group.iloc[end - self.lookback:end]
        future = group.iloc[end:end + self.horizon]

        past_x = torch.tensor(past[self.meta.past_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        future_x = torch.tensor(future[self.meta.future_cols].to_numpy(dtype=np.float32), dtype=torch.float32)

        static_row = group.iloc[end - 1]

        static_ids = torch.tensor(
            [
                self.meta.static_maps[column].get(
                    str(static_row[column]) if pd.notna(static_row[column]) else "__MISSING__",
                    0,
                )
                for column in self.meta.static_cols
            ],
            dtype=torch.long,
        )

        if self.require_target:
            y_values = future["demand_target"].to_numpy(dtype=np.float32)
            weight_values = future["sample_weight"].to_numpy(dtype=np.float32)
        else:
            y_values = np.zeros(self.horizon, dtype=np.float32)
            weight_values = np.ones(self.horizon, dtype=np.float32)

        return {
            "past_x": past_x,
            "future_x": future_x,
            "static_ids": static_ids,
            "y": torch.tensor(y_values, dtype=torch.float32),
            "weight": torch.tensor(weight_values, dtype=torch.float32),
        }


def static_cardinalities(meta: DLMetadata) -> list[int]:
    """Return embedding cardinalities including the reserved unseen category."""
    return [len(meta.static_maps[column]) + 1 for column in meta.static_cols]