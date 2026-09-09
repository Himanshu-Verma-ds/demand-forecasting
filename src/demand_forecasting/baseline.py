from __future__ import annotations

"""Naive baselines, scored on the same windows and metrics as the trained models.

Without one of these, a WAPE of 0.27 is an uninterpretable number. The relevant question is
never "is the error small" but "does this beat the cheapest thing that could possibly work",
and for daily retail demand with weekly seasonality that is the seasonal naive.

Three baselines are computed:

  seasonal_naive  - the same weekday from the most recent complete week available at the
                    forecast origin. For horizon day h this is y[T + h - 7*ceil(h/7)], so
                    days 1-7 look back one week and days 8-14 look back two. Every value is
                    observed at the origin, so it is leakage-free by construction.
  naive           - the last observed value, held flat across the horizon.
  moving_average  - the mean of the last 28 observed days, held flat.

All three forecast from the same origin as the models and are scored on the identical rows.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import read_raw
from .logging_utils import log_run_context, log_settings, setup_logging
from .metrics import full_evaluation
from .splits import make_temporal_split


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
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("baseline", cfg)

    date_col = cfg["data"]["date_col"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_run_context(logger, "baseline", cfg, config_path=args.config, outdir=str(outdir))

    raw = read_raw(cfg["data"]["raw_path"], cfg)
    split = make_temporal_split(raw, cfg)

    windows = {
        "validation": (split.train_end, split.val_start, split.val_end),
        "test": (split.val_end, split.test_start, split.test_end),
    }

    methods = {
        "seasonal_naive": lambda h, t: seasonal_naive(h, t, cfg),
        "naive_last_value": lambda h, t: flat_baseline(h, t, cfg, None),
        "moving_average_28": lambda h, t: flat_baseline(h, t, cfg, 28),
    }

    rows = []
    payload = {}

    for split_name, (origin, start, end) in windows.items():
        history = raw[raw[date_col] <= origin]
        truth = raw[(raw[date_col] >= start) & (raw[date_col] <= end)].copy()

        for method_name, fn in methods.items():
            report = evaluate(history, truth, fn(history, truth), cfg)
            overall = report["overall"]

            actual = float(truth[cfg["data"]["target_col"]].sum())
            predicted = float(pd.Series(fn(history, truth)).clip(lower=0).sum())

            rows.append({
                "baseline": method_name,
                "split": split_name,
                "origin": str(pd.Timestamp(origin).date()),
                "rows": len(truth),
                **{k: overall[k] for k in ["wape", "mape", "mae", "rmse", "bias"]},
                "actual_units": round(actual, 2),
                "predicted_units": round(predicted, 2),
                "bias_pct": round((predicted - actual) / actual * 100, 4),
            })

            payload.setdefault(method_name, {})[split_name] = report
            logger.info("%s / %s: WAPE %.4f  MAE %.2f", method_name, split_name, overall["wape"], overall["mae"])

    table = pd.DataFrame(rows)
    csv_path = outdir / "baseline_metrics.csv"
    json_path = outdir / "baseline_evaluation.json"

    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    logger.info("Baselines:\n%s", table[["baseline", "split", "wape", "mape", "mae", "bias_pct"]].round(4).to_string(index=False))
    log_settings(logger, "Baseline output", {"csv": str(csv_path), "json": str(json_path)})


if __name__ == "__main__":
    main()
