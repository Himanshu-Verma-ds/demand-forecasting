from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import read_raw
from .features import add_calendar_features
from .logging_utils import log_run_context, setup_logging
from .models.dl_data import DLMetadata, MultiSeriesWindowDataset, get_dl_feature_columns, static_cardinalities
from .models.lstm import GlobalLSTMForecaster
from .models.transformer import GlobalTemporalTransformer
from .stockout import add_stockout_target


def load_checkpoint(path: str, device: torch.device):
    """Load the saved model, metadata, and training configuration."""
    checkpoint = torch.load(path, map_location=device)
    meta = DLMetadata(**checkpoint["meta"])
    dl_cfg = checkpoint["dl_config"]
    cardinalities = static_cardinalities(meta)

    past_dim = len(meta.past_cols)
    future_dim = len(meta.future_cols)

    if checkpoint["model_type"] == "lstm":
        model = GlobalLSTMForecaster(
            past_dim,
            future_dim,
            cardinalities,
            hidden=dl_cfg["hidden_size"],
            emb_dim=dl_cfg["embedding_dim"],
            num_layers=dl_cfg["num_layers"],
            dropout=dl_cfg["dropout"],
        )

    elif checkpoint["model_type"] == "transformer":
        model = GlobalTemporalTransformer(
            past_dim,
            future_dim,
            cardinalities,
            d_model=dl_cfg["transformer_d_model"],
            nhead=dl_cfg["transformer_heads"],
            num_layers=dl_cfg["transformer_layers"],
            emb_dim=dl_cfg["embedding_dim"],
            dropout=dl_cfg["dropout"],
        )

    else:
        raise ValueError(f"Unknown model type in checkpoint: {checkpoint['model_type']}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    return model, meta, checkpoint


def validate_checkpoint(meta: DLMetadata, checkpoint: dict, cfg: dict) -> None:
    """Ensure inference config matches the feature schema used during training."""
    expected_past, expected_future, expected_static = get_dl_feature_columns(cfg)

    if meta.past_cols != expected_past:
        raise ValueError("Current config produces different past features from those stored in the checkpoint")

    if meta.future_cols != expected_future:
        raise ValueError("Current config produces different future features from those stored in the checkpoint")

    if meta.static_cols != expected_static:
        raise ValueError("Current config produces different static features from those stored in the checkpoint")

    if int(checkpoint["lookback"]) != int(cfg["data"]["lookback"]):
        raise ValueError("Config lookback does not match the trained checkpoint")

    if int(checkpoint["horizon"]) != int(cfg["data"]["horizon"]):
        raise ValueError("Config horizon does not match the trained checkpoint")


def prepare_future(future: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Remove unavailable future information while retaining genuinely known future covariates."""
    data_cfg = cfg["data"]
    feature_cfg = cfg["features"]

    target_col = data_cfg["target_col"]
    out = future.copy()

    for column in [target_col, "demand_target", "gross_sales", "net_sales", "stock_out_flag", "stock_on_hand"]:
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

    out["demand_target"] = np.nan
    out["sample_weight"] = 1.0
    out["stockout_imputed_amount"] = 0.0

    return out


def validate_future_structure(history: pd.DataFrame, future: pd.DataFrame, cfg: dict) -> None:
    """Validate that each future series contains exactly one configured forecast horizon."""
    data_cfg = cfg["data"]

    date_col = data_cfg["date_col"]
    series_cols = data_cfg["series_cols"]
    horizon = int(data_cfg["horizon"])

    if history.empty:
        raise ValueError("History dataframe cannot be empty")

    if future.empty:
        raise ValueError("Future dataframe cannot be empty")

    if pd.to_datetime(future[date_col]).min() <= pd.to_datetime(history[date_col]).max():
        raise ValueError("All future forecast dates must be after the final history date")

    counts = future.groupby(series_cols, dropna=False).size()

    if not counts.eq(horizon).all():
        bad = counts[counts.ne(horizon)].to_dict()
        raise ValueError(f"Each future series must contain exactly {horizon} rows. Invalid series: {bad}")

    unique_dates = pd.to_datetime(future[date_col]).nunique()

    if unique_dates != horizon:
        raise ValueError(f"Future dataframe must contain exactly {horizon} unique forecast dates")


def forecast(model_path: str, history: pd.DataFrame, future: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Generate direct multi-horizon DL forecasts without using unavailable future information."""
    data_cfg = cfg["data"]
    dl_cfg = cfg["dl"]

    date_col = data_cfg["date_col"]
    series_cols = data_cfg["series_cols"]
    lookback = int(data_cfg["lookback"])
    horizon = int(data_cfg["horizon"])

    history = history.sort_values([*series_cols, date_col]).copy()
    future = future.sort_values([*series_cols, date_col]).copy()

    validate_future_structure(history, future, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, checkpoint = load_checkpoint(model_path, device)
    validate_checkpoint(meta, checkpoint, cfg)

    output = future.copy()
    future_model = prepare_future(future, cfg)

    history_model = add_stockout_target(history, cfg)
    history_model = add_calendar_features(history_model, date_col=date_col)
    future_model = add_calendar_features(future_model, date_col=date_col)

    start = pd.to_datetime(future_model[date_col]).min()
    end = pd.to_datetime(future_model[date_col]).max()
    history_start = start - pd.Timedelta(days=lookback)

    combined = pd.concat([history_model, future_model], ignore_index=True, sort=False)
    combined = combined.sort_values([*series_cols, date_col]).reset_index(drop=True)

    frame = combined[(combined[date_col] >= history_start) & (combined[date_col] <= end)].copy()

    dataset = MultiSeriesWindowDataset(frame, meta, cfg, require_target=False)

    if len(dataset) == 0:
        raise ValueError("No valid DL inference windows were generated")

    loader = DataLoader(
        dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=False,
        num_workers=dl_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    predictions = []

    with torch.no_grad():
        for batch in loader:
            past = batch["past_x"].to(device)
            future_x = batch["future_x"].to(device)
            static = batch["static_ids"].to(device)

            if checkpoint["model_type"] == "lstm":
                pred = model(past, future_x, static, teacher_y=None, teacher_forcing=0.0)
            else:
                pred = model(past, future_x, static)

            pred = pred.cpu().numpy() * meta.target_std + meta.target_mean
            predictions.extend(np.clip(pred.reshape(-1), 0.0, None))

    output = output.sort_values([*series_cols, date_col]).reset_index(drop=True)

    if len(predictions) != len(output):
        raise ValueError(f"Prediction rows {len(predictions)} != future rows {len(output)}")

    output["prediction"] = np.asarray(predictions, dtype=float)

    return output


def main():
    ap = argparse.ArgumentParser(description="Generate leakage-safe direct multi-horizon DL forecasts.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--future", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output", default="artifacts/forecast_dl.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("inference_dl", cfg)

    log_run_context(
        logger,
        "inference_dl",
        config_path=args.config,
        model=args.model,
        history=args.history,
        future=args.future,
        output=args.output,
        lookback=cfg["data"]["lookback"],
        horizon=cfg["data"]["horizon"],
        cuda_available=torch.cuda.is_available(),
    )

    history = read_raw(args.history, cfg)
    future = read_raw(args.future, cfg)

    output = forecast(args.model, history, future, cfg)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output[[cfg["data"]["date_col"], *cfg["data"]["series_cols"], "prediction"]].to_csv(args.output, index=False)

    logger.info("Wrote %s forecast rows to %s", f"{len(output):,}", args.output)


if __name__ == "__main__":
    main()