from __future__ import annotations

"""Bayesian hyperparameter search for the sequence models.

Covers the from-scratch PyTorch models (lstm, transformer) and the Darts global models
(tide, tsmixer). The search space includes **lookback**, i.e. the input chunk length, which is
the sequence-model analogue of how much history a tree model's lag features can see.

Every trial runs a real 14-day validation forecast from the train-end origin and is logged to
MLflow as a nested run, exactly like tune_bayesian.py does for the tree models.
"""

import argparse
import json
from pathlib import Path

import mlflow
import optuna

from .config import apply_dl_params, load_config
from .data import read_raw
from .logging_utils import log_run_context, log_settings, setup_logging
from .metrics import full_evaluation
from .splits import make_temporal_split
from .tracking import configure_mlflow, flatten_dict


TORCH_MODELS = {"lstm", "transformer"}
DARTS_MODELS = {"tide", "tsmixer"}


def suggest(trial: optuna.Trial, model: str, cfg: dict) -> dict:
    """Suggest sequence-model hyperparameters, including the input chunk length."""
    horizon = int(cfg["data"]["horizon"])

    # Lookback must cover at least one horizon and enough history for weekly seasonality.
    params = {
        "lookback": trial.suggest_int("lookback", max(14, horizon), 112, step=14),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    }

    if model == "lstm":
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 256])
        params["num_layers"] = trial.suggest_int("num_layers", 1, 3)
        params["embedding_dim"] = trial.suggest_categorical("embedding_dim", [8, 16, 32])
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    elif model == "transformer":
        params["transformer_d_model"] = trial.suggest_categorical("transformer_d_model", [64, 128, 256])
        params["transformer_heads"] = trial.suggest_categorical("transformer_heads", [2, 4, 8])
        params["transformer_layers"] = trial.suggest_int("transformer_layers", 2, 4)
        params["embedding_dim"] = trial.suggest_categorical("embedding_dim", [8, 16, 32])
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    elif model in DARTS_MODELS:
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 256])

    else:
        raise ValueError(f"Unknown sequence model: {model}")

    return params


def evaluate_torch(model_name: str, trial_cfg: dict, raw, split) -> dict:
    """Train a torch model on train only and score the validation horizon."""
    from .train_dl import (
        attach_prediction_metadata,
        enrich,
        origin_frame,
        predict_loader,
        seed_everything,
        train_with_validation,
    )

    date_col = trial_cfg["data"]["date_col"]
    target_col = trial_cfg["data"]["target_col"]

    seed_everything(int(trial_cfg["project"]["random_seed"]))

    full = enrich(raw, trial_cfg)
    train = full[full[date_col] <= split.train_end].copy()
    val_frame = origin_frame(full, split.val_start, split.val_end, trial_cfg)
    val_truth = raw[(raw[date_col] >= split.val_start) & (raw[date_col] <= split.val_end)].copy()

    model, meta, val_loader, device, best_epoch, best_train_loss = train_with_validation(
        model_name, train, val_frame, trial_cfg
    )

    predictions = predict_loader(model, val_loader, device, model_name, meta)
    evaluation = attach_prediction_metadata(val_truth, predictions, trial_cfg)
    report = full_evaluation(evaluation, y_col=target_col, pred_col="prediction")

    report["overall"]["best_epoch"] = float(best_epoch)
    report["overall"]["train_weighted_mae_scaled"] = float(best_train_loss)

    # Hand the fitted model back so the trial can persist it as an artifact.
    return report["overall"], (model, meta, best_epoch, trial_cfg)


def evaluate_darts(model_name: str, trial_cfg: dict, raw, split) -> dict:
    """Train a Darts model on train only and score the validation horizon."""
    from .features import add_calendar_features
    from .stockout import add_stockout_target
    from .train_darts import build_model, evaluate_window, make_series, make_static_maps

    date_col = trial_cfg["data"]["date_col"]
    series_cols = trial_cfg["data"]["series_cols"]

    full = add_stockout_target(raw, trial_cfg)
    full = add_calendar_features(full, date_col=date_col)

    static_maps = make_static_maps(full, series_cols)

    targets, past_cov, future_cov, weights, _ = make_series(full, trial_cfg, static_maps, end_date=split.train_end)

    model = build_model(model_name, trial_cfg)
    model.fit(series=targets, past_covariates=past_cov, future_covariates=future_cov, sample_weight=weights, verbose=False)

    report, _ = evaluate_window(
        model, full, raw, trial_cfg, static_maps,
        origin=split.train_end, start=split.val_start, end=split.val_end,
    )

    return report["overall"], (model, None, None, trial_cfg)


def save_trial_model(model_name: str, fitted, trial_number: int, outdir: Path, logger) -> None:
    """Persist one trial's fitted model and log it as an MLflow artifact.

    Every trial is kept, not just the winner, so any configuration in the sweep can be
    reloaded and inspected without retraining it.
    """
    import torch

    model, meta, best_epoch, trial_cfg = fitted
    path = outdir / f"{model_name}_trial_{trial_number:03d}.pt"

    try:
        if model_name in TORCH_MODELS:
            from dataclasses import asdict

            torch.save({
                "model_type": model_name,
                "state_dict": model.state_dict(),
                "meta": asdict(meta),
                "dl_config": trial_cfg["dl"],
                "lookback": trial_cfg["data"]["lookback"],
                "horizon": trial_cfg["data"]["horizon"],
                "best_epoch": best_epoch,
            }, path)
        else:
            model.save(str(path))

        mlflow.log_artifact(str(path), artifact_path="model")
    except Exception:
        # A failed artifact upload must not lose the trial's metrics.
        logger.exception("Could not persist trial %d model", trial_number)


def main():
    ap = argparse.ArgumentParser(description="Bayesian hyperparameter search for LSTM/Transformer/TiDE/TSMixer, including lookback.")
    ap.add_argument("--model", choices=sorted(TORCH_MODELS | DARTS_MODELS), required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--metric", default=None, help="Validation metric to minimise (default: config bayes.objective_metric)")
    ap.add_argument("--n-trials", type=int, default=None, help="Override bayes.n_trials_dl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("tune_dl", cfg)

    bayes_cfg = cfg["bayes"]
    training_cfg = cfg["training"]

    objective_metric = args.metric or bayes_cfg.get("objective_metric", "wape")
    n_trials = args.n_trials or int(bayes_cfg.get("n_trials_dl", 10))

    log_run_context(
        logger,
        "tune_dl",
        cfg,
        config_path=args.config,
        model=args.model,
        objective_metric=objective_metric,
        n_trials=n_trials,
    )

    raw = read_raw(cfg["data"]["raw_path"], cfg)
    split = make_temporal_split(raw, cfg)

    outdir = Path(training_cfg["save_dir"]) / "tuning"
    outdir.mkdir(parents=True, exist_ok=True)

    log_trial_models = bool(bayes_cfg.get("log_trial_models", False))

    best_params_path = outdir / f"{args.model}_best_params.json"
    trials_path = outdir / f"{args.model}_trials.csv"

    configure_mlflow(training_cfg["mlflow_experiment"])

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, args.model, cfg)
        trial_cfg = apply_dl_params(cfg, params)

        with mlflow.start_run(run_name=f"{args.model}-tune-trial-{trial.number:03d}", nested=True):
            mlflow.set_tags({
                "stage": "tuning",
                "model": args.model,
                "family": "dl" if args.model in TORCH_MODELS else "darts",
                "trial_number": trial.number,
                "objective_metric": objective_metric,
            })

            mlflow.log_params(flatten_dict({"model": args.model, "params": params}))

            if args.model in TORCH_MODELS:
                metrics, fitted = evaluate_torch(args.model, trial_cfg, raw, split)
            else:
                metrics, fitted = evaluate_darts(args.model, trial_cfg, raw, split)

            if objective_metric not in metrics:
                raise ValueError(f"Unknown objective metric {objective_metric!r}. Available: {sorted(metrics)}")

            for key, value in metrics.items():
                # train_* stays a train metric; everything else describes the validation forecast.
                prefix = "" if key.startswith("train_") else "val_"
                mlflow.log_metric(f"{prefix}{key}", value)
                trial.set_user_attr(f"{prefix}{key}", value)

            if log_trial_models:
                save_trial_model(args.model, fitted, trial.number, outdir, logger)

            score = float(metrics[objective_metric])

            logger.info(
                "Trial %d val_%s=%.6f (lookback=%d) params=%s",
                trial.number,
                objective_metric,
                score,
                params["lookback"],
                params,
            )

        return score

    sampler = optuna.samplers.TPESampler(seed=int(cfg["project"]["random_seed"]), multivariate=True)

    # Each trial trains a network, so losing completed trials to an interruption is expensive.
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

    with mlflow.start_run(run_name=f"{args.model}-tuning") as parent_run:
        mlflow.set_tags({
            "stage": "tuning_parent",
            "model": args.model,
            "family": "dl" if args.model in TORCH_MODELS else "darts",
            "objective_metric": objective_metric,
        })

        mlflow.log_params({
            "model": args.model,
            "objective_metric": objective_metric,
            "n_trials": n_trials,
            "sampler": "TPESampler(multivariate=True)",
            "tunes_lookback": True,
        })

        if remaining:
            study.optimize(objective, n_trials=remaining, timeout=bayes_cfg.get("timeout_seconds"))

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
            "best_lookback": study.best_params.get("lookback"),
            "best_trial_metrics": best_metrics,
            "completed_trials": len(study.trials),
            "best_params": study.best_params,
            "best_params_path": str(best_params_path),
            "trials_path": str(trials_path),
            "mlflow_parent_run_id": parent_run.info.run_id,
        })


if __name__ == "__main__":
    main()
