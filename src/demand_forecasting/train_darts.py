from __future__ import annotations

"""Train optional high-level global forecasting models using Darts.

TiDE and TSMixer are trained as global/shared-weight models across all
Store-SKU series. Past-observed and future-known covariates are selected
according to configuration.
"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from darts import TimeSeries
from darts.models import TiDEModel, TSMixerModel

from .config import apply_dl_params, load_config
from .data import read_raw
from .features import add_calendar_features
from .logging_utils import log_metrics, log_run_context, log_settings, setup_logging
from .metrics import full_evaluation
from .splits import make_temporal_split
from .stockout import add_stockout_target
from .tracking import configure_mlflow, flatten_dict


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

    # Read from config so tune_dl.py can search these.
    hidden_size = int(dl_cfg.get("hidden_size", 128))

    if model_name == "tide":
        return TiDEModel(
            hidden_size=hidden_size,
            decoder_output_dim=32,
            num_encoder_layers=int(dl_cfg.get("num_layers", 2)),
            num_decoder_layers=int(dl_cfg.get("num_layers", 2)),
            dropout=dl_cfg["dropout"],
            use_static_covariates=True,
            **common,
        )

    if model_name == "tsmixer":
        return TSMixerModel(
            hidden_size=hidden_size,
            ff_size=hidden_size * 2,
            num_blocks=int(dl_cfg.get("num_layers", 3)),
            dropout=dl_cfg["dropout"],
            use_static_covariates=True,
            **common,
        )

    raise ValueError(f"Unknown Darts model: {model_name}")


def forecast_from_origin(model, full: pd.DataFrame, cfg: dict, static_maps: dict, origin: pd.Timestamp) -> pd.DataFrame:
    """Forecast one horizon starting the day after origin, using history up to origin only.

    Target and past covariates stop at the origin. Future covariates are the genuinely-known ones
    (calendar plus whatever the *_known_future flags allow), so they legitimately extend across the
    horizon.
    """
    data_cfg = cfg["data"]

    date_col = data_cfg["date_col"]
    series_cols = data_cfg["series_cols"]
    lookback = int(data_cfg["lookback"])
    horizon = int(data_cfg["horizon"])
    future_cols = get_future_cols(cfg)

    history = full[full[date_col] <= origin]

    # Known-future covariates must span the whole horizon. Series that stopped before the
    # origin (this dataset has one that ends in 2022) cannot be forecast and are excluded
    # rather than allowed to fail the whole batch inside Darts.
    horizon_end = pd.Timestamp(origin) + pd.Timedelta(days=horizon)
    series_end = full.groupby(series_cols, sort=False, dropna=False)[date_col].max()
    forecastable = set(series_end[series_end >= horizon_end].index)

    targets = []
    past_covariates = []
    future_covariates = []
    keys = []

    for key, group in history.groupby(series_cols, sort=False, dropna=False):
        group = group.sort_values(date_col)

        if len(group) < lookback:
            continue

        if key not in forecastable:
            continue

        key_values = key if isinstance(key, tuple) else (key,)

        static_data = {}

        for column, value in zip(series_cols, key_values):
            encoded = static_maps[column].get(str(value) if pd.notna(value) else "__MISSING__", -1)
            static_data[f"{column}_code"] = [encoded]

        targets.append(
            TimeSeries.from_dataframe(group, time_col=date_col, value_cols="demand_target", fill_missing_dates=True, freq="D")
            .with_static_covariates(pd.DataFrame(static_data))
        )

        past_covariates.append(
            TimeSeries.from_dataframe(group, time_col=date_col, value_cols=PAST_COLS, fill_missing_dates=True, freq="D")
        )

        # Known-future covariates may span the horizon; they carry no target information.
        series_future = full[
            (full[series_cols[0]] == key_values[0]) & (full[series_cols[1]] == key_values[1])
        ].sort_values(date_col)

        future_covariates.append(
            TimeSeries.from_dataframe(series_future, time_col=date_col, value_cols=future_cols, fill_missing_dates=True, freq="D")
        )

        keys.append(tuple(key_values))

    if not targets:
        raise ValueError("No series contain enough history for the requested forecast origin")

    predictions = model.predict(
        n=horizon,
        series=targets,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        verbose=False,
    )

    if not isinstance(predictions, list):
        predictions = [predictions]

    rows = []

    for key_values, prediction_series in zip(keys, predictions):
        frame = prediction_series.to_dataframe().reset_index()
        value_col = [column for column in frame.columns if column != date_col][0]
        frame = frame.rename(columns={value_col: "prediction"})
        frame["prediction"] = frame["prediction"].astype(float).clip(lower=0.0)

        for column, value in zip(series_cols, key_values):
            frame[column] = value

        rows.append(frame[[date_col, *series_cols, "prediction"]])

    return pd.concat(rows, ignore_index=True)


def evaluate_window(model, full: pd.DataFrame, raw: pd.DataFrame, cfg: dict, static_maps: dict,
                    origin: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp):
    """Forecast from origin and score the result against the observed target."""
    data_cfg = cfg["data"]

    date_col = data_cfg["date_col"]
    series_cols = data_cfg["series_cols"]
    target_col = data_cfg["target_col"]
    merge_cols = [date_col, *series_cols]

    truth = raw[(raw[date_col] >= start) & (raw[date_col] <= end)].copy()

    if truth.empty:
        raise ValueError(f"No observations between {start.date()} and {end.date()}")

    predictions = forecast_from_origin(model, full, cfg, static_maps, origin)

    evaluation = truth.merge(predictions[merge_cols + ["prediction"]], on=merge_cols, how="inner", validate="one_to_one")

    if evaluation.empty:
        raise ValueError("No overlap between predictions and observed target")

    return full_evaluation(evaluation, y_col=target_col, pred_col="prediction"), evaluation


def main():
    ap = argparse.ArgumentParser(description="Train global TiDE or TSMixer demand forecasting models.")
    ap.add_argument("--model", choices=["tide", "tsmixer"], required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--params-json", default=None, help="Tuned parameters from tune_dl.py")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tuned_params = json.loads(Path(args.params_json).read_text(encoding="utf-8")) if args.params_json else None

    if tuned_params:
        cfg = apply_dl_params(cfg, tuned_params)

    logger = setup_logging("train_darts", cfg)

    log_run_context(
        logger,
        "train_darts",
        cfg,
        config_path=args.config,
        model=args.model,
        params_json=args.params_json,
        tuned_params=tuned_params,
    )

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

    configure_mlflow(training_cfg["mlflow_experiment"])

    with mlflow.start_run(run_name=f"{args.model}-global") as run:
        mlflow.set_tags({"stage": "final", "model": args.model, "family": "darts"})

        mlflow.log_params(flatten_dict({
            "model": args.model,
            "lookback": data_cfg["lookback"],
            "horizon": data_cfg["horizon"],
            "dl": cfg["dl"],
            "stockout_target_mode": feature_cfg["stockout_target_mode"],
            "promo_known_future": feature_cfg["promo_known_future"],
            "price_known_future": feature_cfg["price_known_future"],
            "weather_known_future": feature_cfg["weather_known_future"],
        }))

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

        # Validation is forecast from the train-end origin, matching train_ml and train_dl.
        logger.info("Forecasting validation window from origin %s", split.train_end.date())

        val_report, val_eval = evaluate_window(
            model, full, raw, cfg, static_maps,
            origin=split.train_end, start=split.val_start, end=split.val_end,
        )

        log_metrics(logger, "Validation metrics (model selection)", val_report["overall"])

        for key, value in val_report["overall"].items():
            mlflow.log_metric(f"val_{key}", value)

        # Refit on train + validation so the test window is a genuine 14-day-ahead forecast.
        logger.info("Refitting on train plus validation through %s", split.val_end.date())

        refit_targets, refit_past, refit_future, refit_weights, _ = make_series(
            full, cfg, static_maps, end_date=split.val_end,
        )

        refit_model = build_model(args.model, cfg)
        refit_model.fit(
            series=refit_targets,
            past_covariates=refit_past,
            future_covariates=refit_future,
            sample_weight=refit_weights,
            verbose=True,
        )

        test_report, test_eval = evaluate_window(
            refit_model, full, raw, cfg, static_maps,
            origin=split.val_end, start=split.test_start, end=split.test_end,
        )

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
            "evaluation": "direct_multi_horizon_no_label_leakage",
        })

        outdir = Path(training_cfg["save_dir"]) / args.model
        outdir.mkdir(parents=True, exist_ok=True)

        model_path = outdir / "model.pt"
        metadata_path = outdir / "metadata.joblib"
        validation_path = outdir / "validation_predictions.csv"
        test_path = outdir / "test_predictions.csv"
        metrics_path = outdir / "metrics.json"

        # The refit model is the one that would be deployed, matching train_ml/train_dl.
        refit_model.save(str(model_path))

        val_eval.to_csv(validation_path, index=False)
        test_eval.to_csv(test_path, index=False)

        metrics_path.write_text(
            json.dumps({"validation": val_report, "test": test_report}, indent=2),
            encoding="utf-8",
        )

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
            # The saved model is refit through val_end, so that is the inference origin.
            "train_end": split.val_end,
        }

        joblib.dump(metadata, metadata_path)

        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metadata_path), artifact_path="model")
        mlflow.log_artifact(str(validation_path), artifact_path="predictions")
        mlflow.log_artifact(str(test_path), artifact_path="predictions")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        log_settings(logger, "Saved artifacts", {
            "model": str(model_path),
            "metadata": str(metadata_path),
            "validation_predictions": str(validation_path),
            "test_predictions": str(test_path),
            "metrics": str(metrics_path),
            "training_series": len(targets),
            "mlflow_run_id": run.info.run_id,
        })

        logger.info("Finished %s global training run %s", args.model, run.info.run_id)


if __name__ == "__main__":
    main()