from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import TiDEModel, TSMixerModel

from .config import load_config
from .data import read_raw
from .features import add_calendar_features
from .logging_utils import log_run_context, setup_logging
from .stockout import add_stockout_target
from .train_darts import get_future_cols


def validate_metadata(metadata: dict, cfg: dict, model_type: str) -> None:
    """Validate that inference config matches the schema used during training."""
    if metadata["model_type"] != model_type:
        raise ValueError(f"Model type mismatch: metadata={metadata['model_type']} requested={model_type}")

    # Lookback is tuned per model, so the saved metadata is authoritative rather than config.
    if int(metadata["horizon"]) != int(cfg["data"]["horizon"]):
        raise ValueError("Configured horizon does not match trained model horizon")

    if metadata["future_cols"] != get_future_cols(cfg):
        raise ValueError("Current future-covariate configuration does not match the model training schema")

    if metadata["stockout_target_mode"] != cfg["features"]["stockout_target_mode"]:
        raise ValueError("Current stockout_target_mode does not match the model training schema")


def load_model(model_type: str, model_path: str):
    """Load the saved TiDE or TSMixer model."""
    if model_type == "tide":
        return TiDEModel.load(model_path)

    if model_type == "tsmixer":
        return TSMixerModel.load(model_path)

    raise ValueError(f"Unknown Darts model type: {model_type}")


def prepare_future(future: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Prepare future rows while removing targets and variables unavailable at forecast time."""
    feature_cfg = cfg["features"]
    date_col = cfg["data"]["date_col"]

    out = future.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    for column in ["units_sold", "demand_target", "sample_weight", "stockout_imputed_amount", "gross_sales", "net_sales", "stock_out_flag", "stock_on_hand"]:
        if column in out.columns:
            out[column] = np.nan

    if not feature_cfg["promo_known_future"] and "promo_flag" in out.columns:
        out["promo_flag"] = np.nan

    if not feature_cfg["price_known_future"]:
        for column in ["list_price", "discount_pct"]:
            if column in out.columns:
                out[column] = np.nan

    if not feature_cfg["weather_known_future"]:
        for column in ["temperature", "rain_mm"]:
            if column in out.columns:
                out[column] = np.nan

    return add_calendar_features(out, date_col=date_col)


def forecastable_keys(future: pd.DataFrame, metadata: dict) -> list:
    """Trained series that the request actually supplies a full horizon for.

    A model trains on every series with enough history, including ones that later went
    inactive (this dataset has one ending 2022-09-12). Those cannot be forecast and must be
    dropped rather than failing the whole batch.
    """
    series_cols = metadata["series_cols"]
    horizon = int(metadata["horizon"])

    counts = future.groupby(series_cols, dropna=False).size()
    available = {key for key, n in counts.items() if n == horizon}

    return [key for key in metadata["keys"] if (key if isinstance(key, tuple) else (key,)) in available]


def validate_future_coverage(history: pd.DataFrame, future: pd.DataFrame, metadata: dict, keys: list) -> None:
    """Require one complete contiguous forecast horizon for each requested series."""
    date_col = metadata["date_col"]
    series_cols = metadata["series_cols"]
    horizon = int(metadata["horizon"])

    history_end = pd.to_datetime(history[date_col]).max()
    expected_dates = pd.date_range(history_end + pd.Timedelta(days=1), periods=horizon, freq="D")

    for key in keys:
        key_values = key if isinstance(key, tuple) else (key,)
        mask = pd.Series(True, index=future.index)

        for column, value in zip(series_cols, key_values):
            mask &= future[column].eq(value)

        group = future.loc[mask].copy()
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(group[date_col]).unique()))

        if len(group) != horizon:
            raise ValueError(f"Expected {horizon} future rows for series {key}, found {len(group)}")

        if not dates.equals(expected_dates):
            raise ValueError(f"Future dates for series {key} do not match the required forecast horizon")


def build_inference_series(history: pd.DataFrame, future: pd.DataFrame, metadata: dict, keys_to_use: list | None = None):
    """Build historical targets, past covariates, and horizon-extended future covariates for Darts."""
    date_col = metadata["date_col"]
    series_cols = metadata["series_cols"]
    past_cols = metadata["past_cols"]
    future_cols = metadata["future_cols"]
    static_maps = metadata["static_maps"]
    lookback = int(metadata["lookback"])

    targets = []
    past_covariates = []
    future_covariates = []
    keys = []

    for key in (keys_to_use if keys_to_use is not None else metadata["keys"]):
        key_values = key if isinstance(key, tuple) else (key,)

        hist_mask = pd.Series(True, index=history.index)
        future_mask = pd.Series(True, index=future.index)

        for column, value in zip(series_cols, key_values):
            hist_mask &= history[column].eq(value)
            future_mask &= future[column].eq(value)

        hist_group = history.loc[hist_mask].sort_values(date_col).copy()
        future_group = future.loc[future_mask].sort_values(date_col).copy()

        if hist_group.empty:
            raise ValueError(f"No historical data available for trained series {key}")

        if len(hist_group) < lookback:
            raise ValueError(f"Series {key} has only {len(hist_group)} history rows; at least {lookback} are required")

        missing_past = [column for column in past_cols if column not in hist_group.columns]
        missing_future = [column for column in future_cols if column not in future_group.columns]

        if missing_past:
            raise ValueError(f"Missing historical covariates for series {key}: {missing_past}")

        if missing_future:
            raise ValueError(f"Missing future covariates for series {key}: {missing_future}")

        if future_group[future_cols].isna().any().any():
            bad_columns = future_group[future_cols].columns[future_group[future_cols].isna().any()].tolist()
            raise ValueError(f"Missing required future-known covariates for series {key}: {bad_columns}")

        static_data = {}

        for column, value in zip(series_cols, key_values):
            lookup_value = str(value) if pd.notna(value) else "__MISSING__"

            if lookup_value not in static_maps[column]:
                raise ValueError(f"Unknown static value {lookup_value!r} for column {column}")

            static_data[f"{column}_code"] = [static_maps[column][lookup_value]]

        static_covariates = pd.DataFrame(static_data)

        target = TimeSeries.from_dataframe(
            hist_group,
            time_col=date_col,
            value_cols="demand_target",
            fill_missing_dates=True,
            freq="D",
        ).with_static_covariates(static_covariates)

        past_cov = TimeSeries.from_dataframe(
            hist_group,
            time_col=date_col,
            value_cols=past_cols,
            fill_missing_dates=True,
            freq="D",
        )

        future_source = pd.concat(
            [hist_group[[date_col, *future_cols]], future_group[[date_col, *future_cols]]],
            ignore_index=True,
        ).drop_duplicates(subset=[date_col], keep="last").sort_values(date_col)

        future_cov = TimeSeries.from_dataframe(
            future_source,
            time_col=date_col,
            value_cols=future_cols,
            fill_missing_dates=True,
            freq="D",
        )

        targets.append(target)
        past_covariates.append(past_cov)
        future_covariates.append(future_cov)
        keys.append(tuple(key_values))

    return targets, past_covariates, future_covariates, keys


def forecast(model_type: str, model_path: str, metadata_path: str, history: pd.DataFrame, future: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Generate leakage-safe TiDE or TSMixer forecasts."""
    metadata = joblib.load(metadata_path)
    validate_metadata(metadata, cfg, model_type)

    date_col = metadata["date_col"]
    series_cols = metadata["series_cols"]
    horizon = int(metadata["horizon"])

    history = history.sort_values([*series_cols, date_col]).copy()
    history = add_stockout_target(history, cfg)
    history = add_calendar_features(history, date_col=date_col)

    future = prepare_future(future, cfg)

    requested_keys = forecastable_keys(future, metadata)

    if not requested_keys:
        raise ValueError("No trained series has a complete forecast horizon in the request")

    validate_future_coverage(history, future, metadata, requested_keys)

    series, past_covariates, future_covariates, keys = build_inference_series(
        history,
        future,
        metadata,
        keys_to_use=requested_keys,
    )

    model = load_model(model_type, model_path)

    predictions = model.predict(
        n=horizon,
        series=series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        verbose=False,
    )

    if not isinstance(predictions, list):
        predictions = [predictions]

    if len(predictions) != len(keys):
        raise ValueError(f"Prediction series count {len(predictions)} does not match expected series count {len(keys)}")

    rows = []

    for key, prediction_series in zip(keys, predictions):
        pred_df = prediction_series.to_dataframe().reset_index()
        prediction_col = [column for column in pred_df.columns if column != date_col][0]
        pred_df = pred_df.rename(columns={prediction_col: "prediction"})
        pred_df["prediction"] = pred_df["prediction"].astype(float).clip(lower=0.0)

        for column, value in zip(series_cols, key):
            pred_df[column] = value

        rows.append(pred_df[[date_col, *series_cols, "prediction"]])

    output = pd.concat(rows, ignore_index=True)
    output = output.sort_values([*series_cols, date_col]).reset_index(drop=True)

    expected_rows = len(keys) * horizon

    if len(output) != expected_rows:
        raise ValueError(f"Expected {expected_rows} forecast rows, generated {len(output)}")

    return output


def main():
    ap = argparse.ArgumentParser(description="Generate leakage-safe TiDE or TSMixer forecasts.")
    ap.add_argument("--model-type", choices=["tide", "tsmixer"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--future", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output", default="artifacts/forecast_darts.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("inference_darts", cfg)

    log_run_context(
        logger,
        "inference_darts",
        config_path=args.config,
        model_type=args.model_type,
        model=args.model,
        metadata=args.metadata,
        history=args.history,
        future=args.future,
        output=args.output,
        horizon=cfg["data"]["horizon"],
    )

    history = read_raw(args.history, cfg)
    future = read_raw(args.future, cfg)

    output = forecast(
        model_type=args.model_type,
        model_path=args.model,
        metadata_path=args.metadata,
        history=history,
        future=future,
        cfg=cfg,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    logger.info("Wrote %s forecast rows to %s", f"{len(output):,}", args.output)


if __name__ == "__main__":
    main()