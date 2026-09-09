from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlflow
import optuna
import pandas as pd

from .config import load_config
from .data import prepare_features, read_raw
from .features import ml_feature_columns
from .inference_ml import recursive_forecast
from .logging_utils import log_run_context, log_settings, setup_logging
from .metrics import regression_metrics
from .models.ml import build_ml_bundle
from .splits import make_temporal_split
from .tracking import configure_mlflow, flatten_dict


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
    ap.add_argument("--metric", default=None, help="Validation metric to minimise (default: config bayes.objective_metric)")
    ap.add_argument("--n-trials", type=int, default=None, help="Override bayes.n_trials")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("tune_bayesian", cfg)

    objective_metric = args.metric or cfg["bayes"].get("objective_metric", "wape")
    n_trials = args.n_trials or cfg["bayes"]["n_trials"]

    log_run_context(
        logger,
        "tune_bayesian",
        cfg,
        config_path=args.config,
        model=args.model,
        sampler="optuna.TPESampler(multivariate=True)",
        objective=f"validation_{objective_metric}_recursive_14_day",
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

    # Tuning only needs enough rows to rank configurations, not to fit the final model. A
    # smaller cap keeps peak memory well below the box limit; final training uses all rows.
    cap = bayes_cfg.get("max_tune_rows") or training_cfg.get("max_train_rows") or 400_000

    if len(train) > cap:
        train = train.sample(cap, random_state=int(cfg["project"]["random_seed"]))

    history = raw[raw[date_col] <= split.train_end].copy()
    val_truth = raw[(raw[date_col] >= split.val_start) & (raw[date_col] <= split.val_end)].copy()

    # The full feature table is ~1.1M x 72 and is not needed again: every trial refits on `train`
    # and recursive_forecast rebuilds its own features from `history`. Dropping it here frees
    # several hundred MB for the whole study.
    del feat
    gc.collect()

    if val_truth.empty:
        raise ValueError("Validation dataset is empty")

    outdir = Path(training_cfg["save_dir"]) / "tuning"
    outdir.mkdir(parents=True, exist_ok=True)

    log_trial_models = bool(bayes_cfg.get("log_trial_models", False))

    log_settings(logger, "Tuning setup", {
        "model": args.model,
        "objective_metric": objective_metric,
        "n_trials": n_trials,
        "timeout_seconds": bayes_cfg.get("timeout_seconds"),
        "train_rows": len(train),
        "train_row_cap": cap,
        "feature_count": len(feature_cols),
        "validation_window": f"{split.val_start.date()}..{split.val_end.date()}",
        "validation_rows": len(val_truth),
        "log_trial_models": log_trial_models,
    })

    configure_mlflow(training_cfg["mlflow_experiment"])

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, args.model)

        # One nested MLflow run per trial, so every hyperparameter set ever evaluated is
        # recorded next to the metrics it produced.
        with mlflow.start_run(run_name=f"{args.model}-tune-trial-{trial.number:03d}", nested=True):
            mlflow.set_tags({
                "stage": "tuning",
                "model": args.model,
                "trial_number": trial.number,
                "objective_metric": objective_metric,
            })

            mlflow.log_params(flatten_dict({"model": args.model, "params": params}))

            bundle = build_ml_bundle(train, args.model, params, cfg, n_jobs=training_cfg["n_jobs"])
            bundle.pipeline.fit(train[feature_cols], train["demand_target"], model__sample_weight=train["sample_weight"])

            # In-sample fit, so overfitting is visible per trial rather than only at the end.
            train_metrics = regression_metrics(train[target_col], bundle.predict(train[feature_cols]))

            for key, value in train_metrics.items():
                mlflow.log_metric(f"train_{key}", value)
                trial.set_user_attr(f"train_{key}", value)

            pred = recursive_forecast(bundle, history, val_truth.copy(), cfg)

            merged = val_truth.merge(
                pred[merge_cols + ["prediction"]],
                on=merge_cols,
                how="left",
                validate="one_to_one",
            )

            if merged["prediction"].isna().any():
                raise ValueError("Missing validation predictions after merge")

            metrics = regression_metrics(merged[target_col], merged["prediction"])

            if objective_metric not in metrics:
                raise ValueError(f"Unknown objective metric {objective_metric!r}. Available: {sorted(metrics)}")

            for key, value in metrics.items():
                mlflow.log_metric(f"val_{key}", value)

            if log_trial_models:
                trial_model_path = outdir / f"{args.model}_trial_{trial.number:03d}.joblib"
                bundle.save(str(trial_model_path))
                mlflow.log_artifact(str(trial_model_path), artifact_path="model")

            score = float(metrics[objective_metric])

            for key, value in metrics.items():
                trial.set_user_attr(f"val_{key}", value)

            trial.set_user_attr("validation_rows", len(merged))
            trial.set_user_attr("validation_start", str(split.val_start.date()))
            trial.set_user_attr("validation_end", str(split.val_end.date()))

            logger.info(
                "Trial %d val_%s=%.6f (val wape=%.6f mape=%.6f mae=%.4f | train wape=%.6f) params=%s",
                trial.number,
                objective_metric,
                score,
                metrics["wape"],
                metrics["mape"],
                metrics["mae"],
                train_metrics["wape"],
                params,
            )

            # A fitted booster can be hundreds of MB; release it before the next trial
            # rather than relying on the collector to keep up across many trials.
            del bundle, pred, merged
            gc.collect()

        return score

    sampler = optuna.samplers.TPESampler(
        seed=int(cfg["project"]["random_seed"]),
        multivariate=True,
    )

    # Persist the study so an interrupted run resumes from its completed trials instead of
    # starting over. Long searches on a memory-constrained box get killed; this makes that cheap.
    storage = f"sqlite:///{outdir / f'{args.model}_study.db'}"

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"{args.model}-{objective_metric}",
        storage=storage,
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)

    if completed:
        logger.info("Resuming study: %d trials already complete, %d remaining", completed, remaining)

    best_params_path = outdir / f"{args.model}_best_params.json"
    trials_path = outdir / f"{args.model}_trials.csv"

    with mlflow.start_run(run_name=f"{args.model}-tuning") as parent_run:
        mlflow.set_tags({
            "stage": "tuning_parent",
            "model": args.model,
            "objective_metric": objective_metric,
        })

        mlflow.log_params({
            "model": args.model,
            "objective_metric": objective_metric,
            "n_trials": n_trials,
            "sampler": "TPESampler(multivariate=True)",
            "train_rows": len(train),
            "validation_window": f"{split.val_start.date()}..{split.val_end.date()}",
        })

        if remaining:
            study.optimize(
                objective,
                n_trials=remaining,
                timeout=bayes_cfg.get("timeout_seconds"),
            )

        best_params_path.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
        study.trials_dataframe().to_csv(trials_path, index=False)

        best_metrics = {
            key.removeprefix("val_"): value
            for key, value in study.best_trial.user_attrs.items()
            if key.startswith("val_")
        }

        for key, value in best_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"best_val_{key}", value)

        mlflow.log_params(flatten_dict({"best_params": study.best_params}))
        mlflow.log_artifact(str(best_params_path), artifact_path="tuning")
        mlflow.log_artifact(str(trials_path), artifact_path="tuning")

        log_settings(logger, "Best trial", {
            "model": args.model,
            "objective_metric": objective_metric,
            f"best_validation_{objective_metric}": study.best_value,
            "best_trial_number": study.best_trial.number,
            "best_trial_metrics": best_metrics,
            "completed_trials": len(study.trials),
            "best_params": study.best_params,
            "best_params_path": str(best_params_path),
            "trials_path": str(trials_path),
            "mlflow_parent_run_id": parent_run.info.run_id,
        })


if __name__ == "__main__":
    main()