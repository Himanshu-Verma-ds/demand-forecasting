from __future__ import annotations

"""Train optional high-level global forecasting models using Darts.

TiDE and TSMixer are trained as global/shared-weight models across all
Store-SKU series. Past-observed and future-known covariates are selected
according to configuration.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from darts import TimeSeries
from darts.models import TiDEModel, TSMixerModel

from .config import load_config
from .data import read_raw
from .features import add_calendar_features
from .logging_utils import log_run_context, log_settings, setup_logging
from .splits import make_temporal_split
from .stockout import add_stockout_target


BASE_FUTURE_COLS = [
    "is_holiday",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

PAST_COLS = [
    "stock_out_flag",
    "stock_on_hand",
]


def get_future_cols(cfg: dict) -> list[str]:
    """Return future-known Darts covariates according to config."""
    feature_cfg = cfg["features"]
    future_cols = BASE_FUTURE_COLS.copy()

    if feature_cfg["promo_known_future"]:
        future_cols.append("promo_flag")

    if feature_cfg["price_known_future"]:
        future_cols.extend(["list_price", "discount_pct"])

    if feature_cfg["weather_known_future"]:
        future_cols.extend(["temperature", "rain_mm"])

    return future_cols


def make_static_maps(df: pd.DataFrame, series_cols: list[str]) -> dict[str, dict[str, int]]:
    """Create deterministic categorical mappings used as Darts static covariates."""
    static_maps = {}

    for column in series_cols:
        values = sorted(df[column].where(df[column].notna(), "__MISSING__").astype(str).unique())
        static_maps[column] = {value: index for index, value in enumerate(values)}

    return static_maps


def make_series(df: pd.DataFrame, cfg: dict, static_maps: dict[str, dict[str, int]], end_date: pd.Timestamp | None = None):
    """Create global target, covariate, weight, and key lists for Darts training."""
    data_cfg = cfg["data"]

    date_col = data_cfg["date_col"]
    series_cols = data_cfg["series_cols"]
    lookback = int(data_cfg["lookback"])
    horizon = int(data_cfg["horizon"])

    future_cols = get_future_cols(cfg)
    minimum_length = lookback + horizon

    required_columns = [
        date_col,
        *series_cols,
        "demand_target",
        "sample_weight",
        *PAST_COLS,
        *future_cols,
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required Darts columns: {missing_columns}")

    work = df[df[date_col] <= end_date].copy() if end_date is not None else df.copy()

    targets = []
    past_covariates = []
    future_covariates = []
    sample_weights = []
    keys = []

    for key, group in work.groupby(series_cols, sort=False, dropna=False):
        group = group.sort_values(date_col).copy()

        if len(group) < minimum_length:
            continue

        static_data = {}

        key_values = key if isinstance(key, tuple) else (key,)

        for column, value in zip(series_cols, key_values):
            encoded_value = static_maps[column].get(str(value) if pd.notna(value) else "__MISSING__", -1)
            static_data[f"{column}_code"] = [encoded_value]

        static_covariates = pd.DataFrame(static_data)

        target = TimeSeries.from_dataframe(
            group,
            time_col=date_col,
            value_cols="demand_target",
            fill_missing_dates=True,
            freq="D",
        ).with_static_covariates(static_covariates)

        past_cov = TimeSeries.from_dataframe(
            group,
            time_col=date_col,
            value_cols=PAST_COLS,
            fill_missing_dates=True,
            freq="D",
        )

        future_cov = TimeSeries.from_dataframe(
            group,
            time_col=date_col,
            value_cols=future_cols,
            fill_missing_dates=True,
            freq="D",
        )

        weight = TimeSeries.from_dataframe(
            group,
            time_col=date_col,
            value_cols="sample_weight",
            fill_missing_dates=True,
            freq="D",
        )

        targets.append(target)
        past_covariates.append(past_cov)
        future_covariates.append(future_cov)
        sample_weights.append(weight)
        keys.append(tuple(key_values))

    if not targets:
        raise ValueError("No Store-SKU series contain enough history for Darts training")

    return targets, past_covariates, future_covariates, sample_weights, keys


def build_model(model_name: str, cfg: dict):
    """Build the configured TiDE or TSMixer global forecasting model."""
    data_cfg = cfg["data"]
    dl_cfg = cfg["dl"]

    common = {
        "input_chunk_length": int(data_cfg["lookback"]),
        "output_chunk_length": int(data_cfg["horizon"]),
        "n_epochs": int(dl_cfg["epochs"]),
        "batch_size": int(dl_cfg["batch_size"]),
        "random_state": int(cfg["project"]["random_seed"]),
        "force_reset": True,
        "save_checkpoints": False,
    }

    if model_name == "tide":
        return TiDEModel(
            hidden_size=128,
            decoder_output_dim=32,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=dl_cfg["dropout"],
            use_static_covariates=True,
            **common,
        )

    if model_name == "tsmixer":
        return TSMixerModel(
            hidden_size=128,
            ff_size=256,
            num_blocks=3,
            dropout=dl_cfg["dropout"],
            use_static_covariates=True,
            **common,
        )

    raise ValueError(f"Unknown Darts model: {model_name}")


def main():
    ap = argparse.ArgumentParser(description="Train global TiDE or TSMixer demand forecasting models.")
    ap.add_argument("--model", choices=["tide", "tsmixer"], required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("train_darts", cfg)
    log_run_context(logger, "train_darts", cfg, config_path=args.config, model=args.model)

    data_cfg = cfg["data"]
    feature_cfg = cfg["features"]
    training_cfg = cfg["training"]

    date_col = data_cfg["date_col"]
    series_cols = data_cfg["series_cols"]

    raw = read_raw(data_cfg["raw_path"], cfg)
    split = make_temporal_split(raw, cfg)

    full = add_stockout_target(raw, cfg)
    full = add_calendar_features(full, date_col=date_col)

    static_maps = make_static_maps(full, series_cols)

    targets, past_covariates, future_covariates, sample_weights, keys = make_series(
        full,
        cfg,
        static_maps,
        end_date=split.train_end,
    )

    log_settings(logger, "Darts training setup", {
        "model": args.model,
        "series_count": len(targets),
        "train_end": str(split.train_end.date()),
        "lookback": int(data_cfg["lookback"]),
        "horizon": int(data_cfg["horizon"]),
        "epochs": int(cfg["dl"]["epochs"]),
        "batch_size": int(cfg["dl"]["batch_size"]),
        "past_cols": PAST_COLS,
        "future_cols": get_future_cols(cfg),
    })

    model = build_model(args.model, cfg)

    if not model.supports_sample_weight:
        raise ValueError(f"{args.model} does not support sample_weight in the installed Darts version")

    model.fit(
        series=targets,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        sample_weight=sample_weights,
        verbose=True,
    )

    outdir = Path(training_cfg["save_dir"]) / args.model
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / "model.pt"
    metadata_path = outdir / "metadata.joblib"

    model.save(str(model_path))

    metadata = {
        "model_type": args.model,
        "keys": keys,
        "past_cols": PAST_COLS,
        "future_cols": get_future_cols(cfg),
        "static_maps": static_maps,
        "series_cols": series_cols,
        "date_col": date_col,
        "target_col": data_cfg["target_col"],
        "lookback": int(data_cfg["lookback"]),
        "horizon": int(data_cfg["horizon"]),
        "stockout_target_mode": feature_cfg["stockout_target_mode"],
        "promo_known_future": feature_cfg["promo_known_future"],
        "price_known_future": feature_cfg["price_known_future"],
        "weather_known_future": feature_cfg["weather_known_future"],
        "train_end": split.train_end,
    }

    joblib.dump(metadata, metadata_path)

    log_settings(logger, "Saved artifacts", {
        "model": str(model_path),
        "metadata": str(metadata_path),
        "training_series": len(targets),
    })

    logger.info("Finished %s global training", args.model)


if __name__ == "__main__":
    main()