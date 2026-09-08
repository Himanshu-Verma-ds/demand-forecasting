from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import pandas as pd

from .config import load_config
from .data import read_raw, prepare_features
from .features import ml_feature_columns
from .inference_ml import recursive_forecast
from .logging_utils import log_metrics, log_run_context, log_settings, setup_logging
from .metrics import full_evaluation, regression_metrics
from .models.ml import build_ml_bundle, default_params
from .splits import make_temporal_split
from .tracking import configure_mlflow, flatten_dict, log_json_artifact


LOGGER = logging.getLogger("demand_forecasting.train_ml")


def train_one(model_name: str, cfg: dict, raw: pd.DataFrame, params: dict | None = None):
    """Train, recursively validate, refit, and evaluate one ML forecasting model."""
    data_cfg = cfg["data"]
    feature_cfg = cfg["features"]
    training_cfg = cfg["training"]

    date_col = data_cfg["date_col"]
    series_cols = data_cfg["series_cols"]
    target_col = data_cfg["target_col"]
    merge_cols = [date_col, *series_cols]

    split = make_temporal_split(raw, cfg)
    params = params or default_params(model_name)

    feat = prepare_features(raw, cfg)
    numeric, categorical = ml_feature_columns(feat, cfg)
    feature_cols = numeric + categorical

    max_demand_lag = max(feature_cfg["demand_lags"])
    warmup_col = f"demand_lag_{max_demand_lag}"

    if warmup_col not in feat.columns:
        raise ValueError(f"Missing expected warm-up feature: {warmup_col}")

    train = feat[feat[date_col] <= split.train_end].dropna(subset=[warmup_col]).copy()

    max_train_rows = training_cfg.get("max_train_rows")

    if max_train_rows and len(train) > max_train_rows:
        train = train.sample(max_train_rows, random_state=int(cfg["project"]["random_seed"]))

    if train.empty:
        raise ValueError("Training dataset is empty after applying split and warm-up filtering")

    LOGGER.info(
        "Fitting %s on %s training rows with %d features (%d numeric, %d categorical)",
        model_name,
        f"{len(train):,}",
        len(feature_cols),
        len(numeric),
        len(categorical),
    )

    bundle = build_ml_bundle(train, model_name, params, cfg, n_jobs=training_cfg["n_jobs"])
    bundle.pipeline.fit(train[feature_cols], train["demand_target"], model__sample_weight=train["sample_weight"])

    train_pred = bundle.predict(train[feature_cols])
    train_metrics = regression_metrics(train[target_col], train_pred)

    history_train = raw[raw[date_col] <= split.train_end].copy()
    val_truth = raw[(raw[date_col] >= split.val_start) & (raw[date_col] <= split.val_end)].copy()

    if val_truth.empty:
        raise ValueError("Validation dataset is empty")

    LOGGER.info(
        "Running recursive validation forecast for %s..%s",
        split.val_start.date(),
        split.val_end.date(),
    )

    val_pred = recursive_forecast(bundle, history_train, val_truth.copy(), cfg)
    val_eval = val_truth.merge(val_pred[merge_cols + ["prediction"]], on=merge_cols, how="left", validate="one_to_one")

    if val_eval["prediction"].isna().any():
        raise ValueError("Missing validation predictions after merge")

    val_report = full_evaluation(val_eval, y_col=target_col, pred_col="prediction")

    refit_end = split.val_end
    refit_feat = feat[feat[date_col] <= refit_end].dropna(subset=[warmup_col]).copy()

    if refit_feat.empty:
        raise ValueError("Refit dataset is empty")

    refit_numeric, refit_categorical = ml_feature_columns(refit_feat, cfg)
    refit_feature_cols = refit_numeric + refit_categorical

    if refit_feature_cols != feature_cols:
        raise ValueError("Feature columns changed between initial training and refit")

    LOGGER.info("Refitting %s on %s rows through %s", model_name, f"{len(refit_feat):,}", refit_end.date())

    refit_bundle = build_ml_bundle(refit_feat, model_name, params, cfg, n_jobs=training_cfg["n_jobs"])
    refit_bundle.pipeline.fit(refit_feat[refit_feature_cols], refit_feat["demand_target"], model__sample_weight=refit_feat["sample_weight"])

    test_truth = raw[(raw[date_col] >= split.test_start) & (raw[date_col] <= split.test_end)].copy()

    if test_truth.empty:
        raise ValueError("Test dataset is empty")

    LOGGER.info(
        "Running recursive test forecast for %s..%s",
        split.test_start.date(),
        split.test_end.date(),
    )

    test_history = raw[raw[date_col] <= refit_end].copy()
    test_pred = recursive_forecast(refit_bundle, test_history, test_truth.copy(), cfg)
    test_eval = test_truth.merge(test_pred[merge_cols + ["prediction"]], on=merge_cols, how="left", validate="one_to_one")

    if test_eval["prediction"].isna().any():
        raise ValueError("Missing test predictions after merge")

    test_report = full_evaluation(test_eval, y_col=target_col, pred_col="prediction")

    return refit_bundle, train_metrics, val_report, test_report, val_eval, test_eval, split


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate classical ML demand forecasting models.")
    ap.add_argument("--model", choices=["lightgbm", "xgboost", "catboost"], required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--params-json", default=None, help="Optional tuned parameter JSON")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("train_ml", cfg)

    log_run_context(
        logger,
        "train_ml",
        cfg,
        config_path=args.config,
        model=args.model,
        params_json=args.params_json,
    )

    raw = read_raw(cfg["data"]["raw_path"], cfg)
    logger.info("Loaded %s raw rows from %s", f"{len(raw):,}", cfg["data"]["raw_path"])

    params = json.loads(Path(args.params_json).read_text(encoding="utf-8")) if args.params_json else None

    configure_mlflow(cfg["training"]["mlflow_experiment"])

    with mlflow.start_run(run_name=f"{args.model}-final") as run:
        effective_params = params or default_params(args.model)

        log_settings(logger, "Model hyperparameters", {
            "model": args.model,
            "source": "tuned_json" if params else "default_params",
            "params": effective_params,
            "mlflow_run_id": run.info.run_id,
            "mlflow_experiment": cfg["training"]["mlflow_experiment"],
        })

        mlflow.log_params(flatten_dict({
            "model": args.model,
            "params": effective_params,
            "stockout_target_mode": cfg["features"]["stockout_target_mode"],
            "promo_known_future": cfg["features"]["promo_known_future"],
            "price_known_future": cfg["features"]["price_known_future"],
            "weather_known_future": cfg["features"]["weather_known_future"],
        }))

        bundle, train_metrics, val_report, test_report, val_df, test_df, split = train_one(args.model, cfg, raw, params)

        log_settings(logger, "Temporal split", {
            "train_end": str(split.train_end.date()),
            "validation": f"{split.val_start.date()}..{split.val_end.date()}",
            "test": f"{split.test_start.date()}..{split.test_end.date()}",
        })

        log_metrics(logger, "Train metrics (in-sample fit)", train_metrics)
        log_metrics(logger, "Validation metrics (model selection)", val_report["overall"])
        log_metrics(logger, "Test metrics (final reporting only)", test_report["overall"])

        for key, value in train_metrics.items():
            mlflow.log_metric(f"train_{key}", value)

        for key, value in val_report["overall"].items():
            mlflow.log_metric(f"val_{key}", value)

        for key, value in test_report["overall"].items():
            mlflow.log_metric(f"test_{key}", value)

        if test_report["business_proxy"] is not None:
            for key, value in test_report["business_proxy"].items():
                mlflow.log_metric(f"test_business_{key}", value)

        mlflow.set_tags({
            "split.train_end": str(split.train_end.date()),
            "split.val": f"{split.val_start.date()}..{split.val_end.date()}",
            "split.test": f"{split.test_start.date()}..{split.test_end.date()}",
            "evaluation": "recursive_multi_day_no_label_leakage",
        })

        outdir = Path(cfg["training"]["save_dir"]) / args.model
        outdir.mkdir(parents=True, exist_ok=True)

        model_path = outdir / "model.joblib"
        validation_path = outdir / "validation_predictions.csv"
        test_path = outdir / "test_predictions.csv"
        metrics_path = outdir / "metrics.json"

        bundle.save(str(model_path))
        val_df.to_csv(validation_path, index=False)
        test_df.to_csv(test_path, index=False)

        metrics_payload = {
            "train": train_metrics,
            "validation": val_report,
            "test": test_report,
        }

        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(validation_path), artifact_path="predictions")
        mlflow.log_artifact(str(test_path), artifact_path="predictions")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        log_json_artifact({"validation": val_report, "test": test_report}, f"{args.model}_evaluation.json")

        log_settings(logger, "Saved artifacts", {
            "model": str(model_path),
            "validation_predictions": str(validation_path),
            "test_predictions": str(test_path),
            "metrics": str(metrics_path),
        })

        logger.info("Finished %s training run %s", args.model, run.info.run_id)


if __name__ == "__main__":
    main()