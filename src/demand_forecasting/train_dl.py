from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import random

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import apply_dl_params, load_config
from .data import read_raw
from .features import add_calendar_features
from .logging_utils import log_metrics, log_run_context, log_settings, setup_logging
from .metrics import full_evaluation
from .models.dl_data import MultiSeriesWindowDataset, fit_metadata, static_cardinalities
from .models.lstm import GlobalLSTMForecaster
from .models.transformer import GlobalTemporalTransformer
from .splits import make_temporal_split
from .stockout import add_stockout_target
from .tracking import configure_mlflow


LOGGER = logging.getLogger("demand_forecasting.train_dl")


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enrich(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create configurable stockout targets and deterministic calendar features."""
    date_col = cfg["data"]["date_col"]
    out = add_stockout_target(df, cfg)
    return add_calendar_features(out, date_col=date_col)


def origin_frame(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    """Return exactly the historical lookback plus one forecast horizon around an origin."""
    date_col = cfg["data"]["date_col"]
    lookback = int(cfg["data"]["lookback"])
    history_start = start - pd.Timedelta(days=lookback)

    return df[(df[date_col] >= history_start) & (df[date_col] <= end)].copy()


def build_model(kind: str, meta, cfg: dict):
    """Build the configured LSTM or Transformer forecaster."""
    dl_cfg = cfg["dl"]
    cardinalities = static_cardinalities(meta)
    past_dim = len(meta.past_cols)
    future_dim = len(meta.future_cols)

    if kind == "lstm":
        return GlobalLSTMForecaster(
            past_dim,
            future_dim,
            cardinalities,
            hidden=dl_cfg["hidden_size"],
            emb_dim=dl_cfg["embedding_dim"],
            num_layers=dl_cfg["num_layers"],
            dropout=dl_cfg["dropout"],
        )

    if kind == "transformer":
        return GlobalTemporalTransformer(
            past_dim,
            future_dim,
            cardinalities,
            d_model=dl_cfg["transformer_d_model"],
            nhead=dl_cfg["transformer_heads"],
            num_layers=dl_cfg["transformer_layers"],
            emb_dim=dl_cfg["embedding_dim"],
            dropout=dl_cfg["dropout"],
        )

    raise ValueError(f"Unknown DL model: {kind}")


def weighted_l1(pred: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Calculate weighted MAE, supporting reduced confidence for imputed stockout targets."""
    return (torch.abs(pred - y) * weight).sum() / weight.sum().clamp_min(1.0)


def run_epoch(model, loader: DataLoader, device: torch.device, optimizer=None, kind: str = "lstm") -> float:
    """Run one training or evaluation epoch."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        past = batch["past_x"].to(device)
        future = batch["future_x"].to(device)
        static = batch["static_ids"].to(device)
        y = batch["y"].to(device)
        weight = batch["weight"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        if kind == "lstm":
            pred = model(past, future, static, teacher_y=y if training else None, teacher_forcing=0.2 if training else 0.0)
        else:
            pred = model(past, future, static)

        loss = weighted_l1(pred, y, weight)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(y)
        total_samples += len(y)

    return total_loss / max(total_samples, 1)


def predict_loader(model, loader: DataLoader, device: torch.device, kind: str, meta) -> np.ndarray:
    """Generate forecasts and convert scaled predictions back to original demand units."""
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            past = batch["past_x"].to(device)
            future = batch["future_x"].to(device)
            static = batch["static_ids"].to(device)

            if kind == "lstm":
                pred = model(past, future, static, teacher_y=None, teacher_forcing=0.0)
            else:
                pred = model(past, future, static)

            pred = pred.cpu().numpy() * meta.target_std + meta.target_mean
            y = batch["y"].numpy() * meta.target_std + meta.target_mean
            pred = np.clip(pred, 0.0, None)

            rows.extend(zip(y.reshape(-1), pred.reshape(-1)))

    return np.asarray(rows, dtype=float)


def make_loader(dataset, cfg: dict, shuffle: bool) -> DataLoader:
    """Create a configured PyTorch DataLoader."""
    dl_cfg = cfg["dl"]

    return DataLoader(
        dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=dl_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )


def train_with_validation(kind: str, train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: dict):
    """Train with validation-based early stopping and return the best checkpoint and epoch."""
    dl_cfg = cfg["dl"]
    meta = fit_metadata(train_df, cfg)

    # Training windows may be subsampled for tractability; validation windows never are.
    train_ds = MultiSeriesWindowDataset(train_df, meta, cfg, max_windows=dl_cfg.get("max_train_windows"))
    val_ds = MultiSeriesWindowDataset(val_df, meta, cfg)

    if len(train_ds) == 0:
        raise ValueError("No valid DL training windows were generated")

    if len(val_ds) == 0:
        raise ValueError("No valid DL validation windows were generated")

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(kind, meta, cfg).to(device)

    LOGGER.info(
        "Training %s on %s device with %s train windows and %s validation windows",
        kind,
        device.type,
        f"{len(train_ds):,}",
        f"{len(val_ds):,}",
    )

    LOGGER.info(
        "Trainable parameters: %s",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=dl_cfg["learning_rate"],
        weight_decay=dl_cfg["weight_decay"],
    )

    best_loss = float("inf")
    best_train_loss = float("nan")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(dl_cfg["epochs"]):
        train_loss = run_epoch(model, train_loader, device, optimizer, kind)
        val_loss = run_epoch(model, val_loader, device, optimizer=None, kind=kind)

        mlflow.log_metric("epoch_train_weighted_mae_scaled", train_loss, step=epoch)
        mlflow.log_metric("epoch_val_weighted_mae_scaled", val_loss, step=epoch)

        LOGGER.info("epoch=%02d train_weighted_mae_scaled=%.4f val_weighted_mae_scaled=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_counter += 1

            if patience_counter >= dl_cfg["patience"]:
                LOGGER.info("Early stopping at epoch %d; best epoch was %d", epoch, best_epoch)
                break

    if best_state is None:
        raise RuntimeError("Training finished without producing a valid checkpoint")

    model.load_state_dict(best_state)

    # Train and validation loss at the selected epoch, so overfitting is visible per run.
    mlflow.log_metric("train_weighted_mae_scaled", best_train_loss)
    mlflow.log_metric("val_weighted_mae_scaled", best_loss)

    return model, meta, val_loader, device, best_epoch, best_train_loss


def refit_fixed_epochs(kind: str, train_df: pd.DataFrame, cfg: dict, epochs: int):
    """Refit a fresh model on train plus validation using the validation-selected epoch count."""
    dl_cfg = cfg["dl"]
    meta = fit_metadata(train_df, cfg)
    train_ds = MultiSeriesWindowDataset(train_df, meta, cfg, max_windows=dl_cfg.get("max_train_windows"))

    if len(train_ds) == 0:
        raise ValueError("No valid DL refit windows were generated")

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(kind, meta, cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=dl_cfg["learning_rate"],
        weight_decay=dl_cfg["weight_decay"],
    )

    for epoch in range(epochs):
        loss = run_epoch(model, train_loader, device, optimizer, kind)
        mlflow.log_metric("refit_train_weighted_mae_scaled", loss, step=epoch)
        LOGGER.info("refit_epoch=%02d train_weighted_mae_scaled=%.4f", epoch, loss)

    return model, meta, device


def attach_prediction_metadata(truth: pd.DataFrame, flat_predictions: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Attach flattened horizon predictions to the corresponding truth rows."""
    date_col = cfg["data"]["date_col"]
    series_cols = cfg["data"]["series_cols"]

    ordered = truth.sort_values([*series_cols, date_col]).copy()

    if len(ordered) != len(flat_predictions):
        raise ValueError(f"Prediction rows {len(flat_predictions)} != truth rows {len(ordered)}")

    ordered["prediction"] = flat_predictions[:, 1]

    return ordered


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate global LSTM or Transformer demand forecasting models.")
    ap.add_argument("--model", choices=["lstm", "transformer"], required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--params-json", default=None, help="Tuned parameters from tune_dl.py")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tuned_params = json.loads(Path(args.params_json).read_text(encoding="utf-8")) if args.params_json else None

    if tuned_params:
        cfg = apply_dl_params(cfg, tuned_params)

    logger = setup_logging("train_dl", cfg)

    log_run_context(
        logger,
        "train_dl",
        cfg,
        config_path=args.config,
        model=args.model,
        params_json=args.params_json,
        tuned_params=tuned_params,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )

    seed_everything(int(cfg["project"]["random_seed"]))

    data_cfg = cfg["data"]
    training_cfg = cfg["training"]

    date_col = data_cfg["date_col"]
    target_col = data_cfg["target_col"]

    raw = read_raw(data_cfg["raw_path"], cfg)
    split = make_temporal_split(raw, cfg)
    full = enrich(raw, cfg)

    train = full[full[date_col] <= split.train_end].copy()
    val_frame = origin_frame(full, split.val_start, split.val_end, cfg)
    val_truth = raw[(raw[date_col] >= split.val_start) & (raw[date_col] <= split.val_end)].copy()

    if train.empty:
        raise ValueError("Training dataframe is empty")

    if val_truth.empty:
        raise ValueError("Validation dataframe is empty")

    configure_mlflow(training_cfg["mlflow_experiment"])

    with mlflow.start_run(run_name=f"{args.model}-global") as run:
        mlflow.set_tags({"stage": "final", "model": args.model, "family": "dl"})

        log_settings(logger, "Model settings", {
            "model": args.model,
            "dl": cfg["dl"],
            "lookback": data_cfg["lookback"],
            "horizon": data_cfg["horizon"],
            "mlflow_run_id": run.info.run_id,
            "mlflow_experiment": training_cfg["mlflow_experiment"],
            "split": {
                "train_end": str(split.train_end.date()),
                "validation": f"{split.val_start.date()}..{split.val_end.date()}",
                "test": f"{split.test_start.date()}..{split.test_end.date()}",
            },
        })

        mlflow.log_params({
            "model": args.model,
            "lookback": data_cfg["lookback"],
            "horizon": data_cfg["horizon"],
            "stockout_target_mode": cfg["features"]["stockout_target_mode"],
            "promo_known_future": cfg["features"]["promo_known_future"],
            "price_known_future": cfg["features"]["price_known_future"],
            "weather_known_future": cfg["features"]["weather_known_future"],
        })

        model, meta, val_loader, device, best_epoch, best_train_loss = train_with_validation(args.model, train, val_frame, cfg)
        logger.info("Best epoch %d: train_weighted_mae_scaled=%.4f", best_epoch, best_train_loss)

        val_predictions = predict_loader(model, val_loader, device, args.model, meta)
        val_eval = attach_prediction_metadata(val_truth, val_predictions, cfg)
        val_report = full_evaluation(val_eval, y_col=target_col, pred_col="prediction")

        log_metrics(logger, "Validation metrics (model selection)", val_report["overall"])

        for key, value in val_report["overall"].items():
            mlflow.log_metric(f"val_{key}", value)

        mlflow.log_metric("best_epoch", best_epoch)
        logger.info("Refitting on train plus validation for %d epochs", best_epoch + 1)

        refit_df = full[full[date_col] <= split.val_end].copy()

        seed_everything(int(cfg["project"]["random_seed"]))

        refit_model, refit_meta, refit_device = refit_fixed_epochs(
            args.model,
            refit_df,
            cfg,
            epochs=best_epoch + 1,
        )

        test_frame = origin_frame(full, split.test_start, split.test_end, cfg)
        test_truth = raw[(raw[date_col] >= split.test_start) & (raw[date_col] <= split.test_end)].copy()

        if test_truth.empty:
            raise ValueError("Test dataframe is empty")

        test_ds = MultiSeriesWindowDataset(test_frame, refit_meta, cfg)

        if len(test_ds) == 0:
            raise ValueError("No valid DL test windows were generated")

        test_loader = make_loader(test_ds, cfg, shuffle=False)

        test_predictions = predict_loader(refit_model, test_loader, refit_device, args.model, refit_meta)
        test_eval = attach_prediction_metadata(test_truth, test_predictions, cfg)
        test_report = full_evaluation(test_eval, y_col=target_col, pred_col="prediction")

        log_metrics(logger, "Test metrics (final reporting only)", test_report["overall"])

        for key, value in test_report["overall"].items():
            mlflow.log_metric(f"test_{key}", value)

        if test_report["business_proxy"] is not None:
            for key, value in test_report["business_proxy"].items():
                mlflow.log_metric(f"test_business_{key}", value)

        mlflow.set_tags({
            "split.train_end": str(split.train_end.date()),
            "split.val": f"{split.val_start.date()}..{split.val_end.date()}",
            "split.test": f"{split.test_start.date()}..{split.test_end.date()}",
            "evaluation": "direct_multi_horizon_no_test_early_stopping",
        })

        outdir = Path(training_cfg["save_dir"]) / args.model
        outdir.mkdir(parents=True, exist_ok=True)

        model_path = outdir / "model.pt"
        validation_path = outdir / "validation_predictions.csv"
        test_path = outdir / "test_predictions.csv"
        metrics_path = outdir / "metrics.json"

        checkpoint = {
            "model_type": args.model,
            "state_dict": refit_model.state_dict(),
            "meta": asdict(refit_meta),
            "dl_config": cfg["dl"],
            "lookback": data_cfg["lookback"],
            "horizon": data_cfg["horizon"],
            "best_epoch": best_epoch,
        }

        torch.save(checkpoint, model_path)

        val_eval.to_csv(validation_path, index=False)
        test_eval.to_csv(test_path, index=False)

        metrics_payload = {
            "best_epoch": best_epoch,
            "validation": val_report,
            "test": test_report,
        }

        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(validation_path), artifact_path="predictions")
        mlflow.log_artifact(str(test_path), artifact_path="predictions")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        log_settings(logger, "Saved artifacts", {
            "model": str(model_path),
            "validation_predictions": str(validation_path),
            "test_predictions": str(test_path),
            "metrics": str(metrics_path),
            "best_epoch": best_epoch,
        })

        logger.info("Finished %s training run %s", args.model, run.info.run_id)


if __name__ == "__main__":
    main()