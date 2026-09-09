from __future__ import annotations

"""Baselines, scored on the same windows and metrics as the trained models.

Without a baseline a WAPE of 0.27 is an uninterpretable number. The question is never "is the
error small" but "does this beat something simpler", and for a demand-forecasting problem the
statistical reference point is SARIMA.

  sarima          - SARIMA(1,1,1)(1,1,1)[7] fitted independently per Store-SKU series. One
                    non-seasonal difference for level drift, one seasonal difference at lag 7
                    for the weekly cycle the EDA found, plus AR and MA terms at both scales.
                    This is the headline baseline.

Three naive references are computed alongside it, to show where the floor is:

  seasonal_naive  - the same weekday from the most recent complete week available at the
                    forecast origin. For horizon day h this is y[T + h - 7*ceil(h/7)], so
                    days 1-7 look back one week and days 8-14 look back two. Every value is
                    observed at the origin, so it is leakage-free by construction.
  naive           - the last observed value, held flat across the horizon.
  moving_average  - the mean of the last 28 observed days, held flat.

Every baseline forecasts from the same origin as the models and is scored on identical rows,
and each is logged to MLflow as its own run so it sits beside the models in the experiment.
"""

import argparse
import json
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from .config import load_config
from .data import read_raw
from .logging_utils import log_run_context, log_settings, setup_logging
from .metrics import full_evaluation
from .splits import make_temporal_split
from .tracking import configure_mlflow


def seasonal_naive(history: pd.DataFrame, truth: pd.DataFrame, cfg: dict, period: int = 7) -> pd.Series:
    """Same weekday from the most recent complete week available at the origin."""
    date_col = cfg["data"]["date_col"]
    series_cols = cfg["data"]["series_cols"]
    target_col = cfg["data"]["target_col"]

    origin = pd.to_datetime(history[date_col]).max()
    lookup = history.set_index([*series_cols, date_col])[target_col]

    horizon_day = (pd.to_datetime(truth[date_col]) - origin).dt.days
    # ceil(h / period) whole weeks back, so the source date is always at or before the origin.
    weeks_back = np.ceil(horizon_day / period).astype(int)
    source_date = pd.to_datetime(truth[date_col]) - pd.to_timedelta(weeks_back * period, unit="D")

    keys = pd.MultiIndex.from_arrays(
        [truth[c] for c in series_cols] + [source_date],
        names=[*series_cols, date_col],
    )

    return pd.Series(lookup.reindex(keys).to_numpy(), index=truth.index, name="prediction")


def flat_baseline(history: pd.DataFrame, truth: pd.DataFrame, cfg: dict, window: int | None) -> pd.Series:
    """Last observed value (window=None) or the mean of the last `window` days, held flat."""
    date_col = cfg["data"]["date_col"]
    series_cols = cfg["data"]["series_cols"]
    target_col = cfg["data"]["target_col"]

    ordered = history.sort_values([*series_cols, date_col])

    if window is None:
        level = ordered.groupby(series_cols, dropna=False)[target_col].last()
    else:
        level = ordered.groupby(series_cols, dropna=False)[target_col].apply(lambda s: s.tail(window).mean())

    keys = pd.MultiIndex.from_arrays([truth[c] for c in series_cols], names=series_cols)

    return pd.Series(level.reindex(keys).to_numpy(), index=truth.index, name="prediction")


def sarima_forecast(history: pd.DataFrame, truth: pd.DataFrame, cfg: dict, logger,
                    order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)) -> tuple[pd.Series, dict]:
    """Fit one SARIMA per Store-SKU on history and forecast the horizon.

    A global model this is not: SARIMA has no way to share strength across series, so each of
    the ~1,000 series gets its own fit. That is precisely the cost the global models avoid, and
    part of what the comparison is meant to expose.

    Fits that fail to converge fall back to the seasonal naive for that series rather than
    dropping it, so the baseline is scored on exactly the same rows as everything else.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    date_col = cfg["data"]["date_col"]
    series_cols = cfg["data"]["series_cols"]
    target_col = cfg["data"]["target_col"]
    horizon = int(cfg["data"]["horizon"])

    fallback = seasonal_naive(history, truth, cfg)
    predictions = pd.Series(np.nan, index=truth.index, dtype=float)

    truth_positions = {key: idx for key, idx in truth.groupby(series_cols, dropna=False).groups.items()}
    stats = {"fitted": 0, "failed": 0, "too_short": 0}

    minimum_length = seasonal_order[3] * 2 + 10
    total = history.groupby(series_cols, dropna=False).ngroups

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for n, (key, group) in enumerate(history.groupby(series_cols, sort=False, dropna=False), start=1):
            if n % 200 == 0:
                logger.info("  SARIMA %d/%d series fitted", n, total)

            index = truth_positions.get(key)

            if index is None or len(index) == 0:
                continue

            y = group.sort_values(date_col)[target_col].astype(float).to_numpy()

            if len(y) < minimum_length:
                stats["too_short"] += 1
                continue

            try:
                fitted = SARIMAX(
                    y,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                forecast = np.asarray(fitted.forecast(steps=horizon), dtype=float)

                if not np.isfinite(forecast).all():
                    raise ValueError("non-finite forecast")

                ordered = truth.loc[index].sort_values(date_col).index
                predictions.loc[ordered] = forecast[:len(ordered)]
                stats["fitted"] += 1

            except Exception:
                stats["failed"] += 1

    # Anything not fitted keeps the seasonal-naive value.
    unresolved = predictions.isna()
    stats["fallback_rows"] = int(unresolved.sum())
    predictions[unresolved] = fallback[unresolved]

    return predictions, stats


def evaluate(history: pd.DataFrame, truth: pd.DataFrame, predictions: pd.Series, cfg: dict) -> dict:
    frame = truth.copy()
    frame["prediction"] = predictions.to_numpy(dtype=float)

    if frame["prediction"].isna().any():
        missing = int(frame["prediction"].isna().sum())
        raise ValueError(f"{missing} baseline predictions could not be resolved from history")

    frame["prediction"] = frame["prediction"].clip(lower=0.0)

    return full_evaluation(frame, y_col=cfg["data"]["target_col"], pred_col="prediction")


def main():
    ap = argparse.ArgumentParser(description="Score naive baselines on the validation and test windows.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--skip-sarima", action="store_true", help="Naive baselines only (SARIMA is slow)")
    ap.add_argument("--order", default="1,1,1")
    ap.add_argument("--seasonal-order", default="1,1,1,7")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("baseline", cfg)

    date_col = cfg["data"]["date_col"]
    target_col = cfg["data"]["target_col"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    order = tuple(int(x) for x in args.order.split(","))
    seasonal_order = tuple(int(x) for x in args.seasonal_order.split(","))

    log_run_context(logger, "baseline", cfg, config_path=args.config, outdir=str(outdir),
                    sarima_order=str(order), sarima_seasonal_order=str(seasonal_order))

    raw = read_raw(cfg["data"]["raw_path"], cfg)
    split = make_temporal_split(raw, cfg)

    windows = {
        "validation": (split.train_end, split.val_start, split.val_end),
        "test": (split.val_end, split.test_start, split.test_end),
    }

    methods = {
        "seasonal_naive": lambda h, t: (seasonal_naive(h, t, cfg), {}),
        "naive_last_value": lambda h, t: (flat_baseline(h, t, cfg, None), {}),
        "moving_average_28": lambda h, t: (flat_baseline(h, t, cfg, 28), {}),
    }

    if not args.skip_sarima:
        methods["sarima"] = lambda h, t: sarima_forecast(h, t, cfg, logger, order, seasonal_order)

    rows = []
    payload = {}
    fit_stats = {}
    predictions_out = []

    for split_name, (origin, start, end) in windows.items():
        history = raw[raw[date_col] <= origin]
        truth = raw[(raw[date_col] >= start) & (raw[date_col] <= end)].copy()

        for method_name, fn in methods.items():
            logger.info("Running %s on %s (origin %s)", method_name, split_name, pd.Timestamp(origin).date())

            predicted_series, stats = fn(history, truth)
            report = evaluate(history, truth, predicted_series, cfg)
            overall = report["overall"]

            actual = float(truth[target_col].sum())
            predicted_total = float(predicted_series.clip(lower=0).sum())

            rows.append({
                "baseline": method_name,
                "split": split_name,
                "origin": str(pd.Timestamp(origin).date()),
                "rows": len(truth),
                **{k: overall[k] for k in ["wape", "mape", "mae", "rmse", "bias"]},
                "actual_units": round(actual, 2),
                "predicted_units": round(predicted_total, 2),
                "bias_pct": round((predicted_total - actual) / actual * 100, 4),
            })

            payload.setdefault(method_name, {})[split_name] = report

            if stats:
                fit_stats.setdefault(method_name, {})[split_name] = stats
                logger.info("  %s fit stats: %s", method_name, stats)

            frame = truth[[date_col, *cfg["data"]["series_cols"], target_col]].copy()
            frame["prediction"] = predicted_series.clip(lower=0).to_numpy()
            frame["baseline"] = method_name
            frame["split"] = split_name
            predictions_out.append(frame)

            logger.info("%s / %s: WAPE %.4f  MAE %.2f", method_name, split_name, overall["wape"], overall["mae"])

    table = pd.DataFrame(rows)

    predictions_path = outdir / "baseline_predictions.csv"
    pd.concat(predictions_out, ignore_index=True).to_csv(predictions_path, index=False)

    # One MLflow run per baseline, tagged so it sits alongside the models in the experiment.
    configure_mlflow(cfg["training"]["mlflow_experiment"])

    for method_name in methods:
        with mlflow.start_run(run_name=f"baseline-{method_name}"):
            mlflow.set_tags({"stage": "baseline", "model": method_name, "family": "baseline"})

            params = {"baseline": method_name, "horizon": cfg["data"]["horizon"]}

            if method_name == "sarima":
                params.update({"order": str(order), "seasonal_order": str(seasonal_order),
                               "fit_per_series": True})

            mlflow.log_params(params)

            for _, r in table[table["baseline"] == method_name].iterrows():
                prefix = "val" if r["split"] == "validation" else "test"
                for metric in ["wape", "mape", "mae", "rmse", "bias"]:
                    mlflow.log_metric(f"{prefix}_{metric}", float(r[metric]))
                mlflow.log_metric(f"{prefix}_bias_pct", float(r["bias_pct"]))

            for split_name, stats in fit_stats.get(method_name, {}).items():
                for key, value in stats.items():
                    mlflow.log_metric(f"{'val' if split_name == 'validation' else 'test'}_{key}", float(value))

            mlflow.log_artifact(str(predictions_path), artifact_path="predictions")

    logger.info("Logged %d baseline runs to MLflow experiment %s",
                len(methods), cfg["training"]["mlflow_experiment"])
    csv_path = outdir / "baseline_metrics.csv"
    json_path = outdir / "baseline_evaluation.json"

    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    logger.info("Baselines:\n%s", table[["baseline", "split", "wape", "mape", "mae", "bias_pct"]].round(4).to_string(index=False))
    log_settings(logger, "Baseline output", {"csv": str(csv_path), "json": str(json_path),
                                            "predictions": str(predictions_path), "fit_stats": fit_stats})


if __name__ == "__main__":
    main()
