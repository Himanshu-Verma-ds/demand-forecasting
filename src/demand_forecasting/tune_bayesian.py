from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import pandas as pd

from .config import load_config
from .data import prepare_features, read_raw
from .features import ml_feature_columns
from .inference_ml import recursive_forecast
from .logging_utils import log_run_context, log_settings, setup_logging
from .metrics import wape
from .models.ml import build_ml_bundle
from .splits import make_temporal_split


def suggest(trial: optuna.Trial, model: str) -> dict:
    """Suggest Bayesian hyperparameters for the requested model."""
    model = model.lower()

    if model == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 5, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    if model == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }

    if model == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        }

    raise ValueError(f"Unknown model: {model}")


def main():
    ap = argparse.ArgumentParser(description="Optuna TPE Bayesian hyperparameter search using recursive leakage-safe validation.")
    ap.add_argument("--model", choices=["lightgbm", "xgboost", "catboost"], required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("tune_bayesian", cfg)

    log_run_context(
        logger,
        "tune_bayesian",
        cfg,
        config_path=args.config,
        model=args.model,
        sampler="optuna.TPESampler(multivariate=True)",
        objective="validation_wape_recursive_14_day",
    )

    data_cfg = cfg["data"]
    feature_cfg = cfg["features"]
    training_cfg = cfg["training"]
    bayes_cfg = cfg["bayes"]

    date_col = data_cfg["date_col"]
    series_cols = data_cfg["series_cols"]
    target_col = data_cfg["target_col"]
    merge_cols = [date_col, *series_cols]

    raw = read_raw(data_cfg["raw_path"], cfg)
    split = make_temporal_split(raw, cfg)

    feat = prepare_features(raw, cfg)

    numeric, categorical = ml_feature_columns(feat, cfg)
    feature_cols = numeric + categorical

    max_demand_lag = max(feature_cfg["demand_lags"])
    warmup_col = f"demand_lag_{max_demand_lag}"

    if warmup_col not in feat.columns:
        raise ValueError(f"Missing expected warm-up feature: {warmup_col}")

    train = feat[feat[date_col] <= split.train_end].dropna(subset=[warmup_col]).copy()

    if train.empty:
        raise ValueError("Training dataset is empty after applying split and warm-up filtering")

    cap = training_cfg.get("max_train_rows") or 400_000

    if len(train) > cap:
        train = train.sample(cap, random_state=int(cfg["project"]["random_seed"]))

    history = raw[raw[date_col] <= split.train_end].copy()
    val_truth = raw[(raw[date_col] >= split.val_start) & (raw[date_col] <= split.val_end)].copy()

    if val_truth.empty:
        raise ValueError("Validation dataset is empty")

    log_settings(logger, "Tuning setup", {
        "model": args.model,
        "n_trials": bayes_cfg["n_trials"],
        "timeout_seconds": bayes_cfg.get("timeout_seconds"),
        "train_rows": len(train),
        "train_row_cap": cap,
        "feature_count": len(feature_cols),
        "validation_window": f"{split.val_start.date()}..{split.val_end.date()}",
        "validation_rows": len(val_truth),
    })

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, args.model)

        bundle = build_ml_bundle(train, args.model, params, cfg, n_jobs=training_cfg["n_jobs"])
        bundle.pipeline.fit(train[feature_cols], train["demand_target"], model__sample_weight=train["sample_weight"])

        pred = recursive_forecast(bundle, history, val_truth.copy(), cfg)

        merged = val_truth.merge(
            pred[merge_cols + ["prediction"]],
            on=merge_cols,
            how="left",
            validate="one_to_one",
        )

        if merged["prediction"].isna().any():
            raise ValueError("Missing validation predictions after merge")

        score = wape(merged[target_col], merged["prediction"])

        trial.set_user_attr("validation_rows", len(merged))
        trial.set_user_attr("validation_start", str(split.val_start.date()))
        trial.set_user_attr("validation_end", str(split.val_end.date()))

        logger.info("Trial %d validation WAPE %.6f with params %s", trial.number, score, params)

        return score

    sampler = optuna.samplers.TPESampler(
        seed=int(cfg["project"]["random_seed"]),
        multivariate=True,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"{args.model}-wape",
    )

    study.optimize(
        objective,
        n_trials=bayes_cfg["n_trials"],
        timeout=bayes_cfg.get("timeout_seconds"),
    )

    outdir = Path(training_cfg["save_dir"]) / "tuning"
    outdir.mkdir(parents=True, exist_ok=True)

    best_params_path = outdir / f"{args.model}_best_params.json"
    trials_path = outdir / f"{args.model}_trials.csv"

    best_params_path.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    study.trials_dataframe().to_csv(trials_path, index=False)

    log_settings(logger, "Best trial", {
        "model": args.model,
        "best_validation_wape": study.best_value,
        "best_trial_number": study.best_trial.number,
        "completed_trials": len(study.trials),
        "best_params": study.best_params,
        "best_params_path": str(best_params_path),
        "trials_path": str(trials_path),
    })


if __name__ == "__main__":
    main()