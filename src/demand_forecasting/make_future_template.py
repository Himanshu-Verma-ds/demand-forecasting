from __future__ import annotations

"""Build the known-future covariate rows needed to forecast the next horizon.

The API and the batch inference CLIs both require one contiguous row per Store-SKU per
future date, carrying only the fields that are genuinely known at the forecast origin.
This builds that frame from history so the contract is never guessed at by hand.

Outputs a CSV for the batch CLIs and a JSON body for POST /forecast.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import read_raw
from .logging_utils import log_run_context, log_settings, setup_logging


def required_future_fields(cfg: dict) -> list[str]:
    """Fields the request must carry, driven by the *_known_future contract."""
    data_cfg = cfg["data"]
    feature_cfg = cfg["features"]

    fields = [data_cfg["date_col"], *data_cfg["series_cols"], "is_holiday"]

    if feature_cfg["promo_known_future"]:
        fields.append("promo_flag")

    if feature_cfg["price_known_future"]:
        fields.extend(["list_price", "discount_pct"])

    if feature_cfg["weather_known_future"]:
        fields.extend(["temperature", "rain_mm"])

    return fields


def build_future_frame(history: pd.DataFrame, cfg: dict, series_limit: int | None = None) -> pd.DataFrame:
    """Create one contiguous horizon of known-future rows for every active Store-SKU."""
    data_cfg = cfg["data"]

    date_col = data_cfg["date_col"]
    series_cols = data_cfg["series_cols"]
    horizon = int(data_cfg["horizon"])

    history_end = pd.to_datetime(history[date_col]).max()
    future_dates = pd.date_range(history_end + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Only series still alive at the final history date are forecastable.
    last_seen = history.groupby(series_cols, dropna=False)[date_col].max().reset_index()
    active = last_seen[last_seen[date_col].eq(history_end)][series_cols]

    if active.empty:
        raise ValueError("No Store-SKU series are active at the final history date")

    if series_limit:
        active = active.head(series_limit)

    frame = active.merge(pd.DataFrame({date_col: future_dates}), how="cross")

    # Calendar fields are deterministic from the date.
    frame["is_holiday"] = 0
    frame["is_weekend"] = frame[date_col].dt.dayofweek.isin([5, 6]).astype(int)

    feature_cfg = cfg["features"]

    if feature_cfg["promo_known_future"]:
        # Default to "no promotion planned"; overwrite with the real promo calendar.
        frame["promo_flag"] = 0

    if feature_cfg["price_known_future"]:
        latest = history.sort_values(date_col).groupby(series_cols, as_index=False).tail(1)
        frame = frame.merge(latest[[*series_cols, "list_price", "discount_pct"]], on=series_cols, how="left")

    if feature_cfg["weather_known_future"]:
        frame["temperature"] = float(history["temperature"].tail(28).mean())
        frame["rain_mm"] = float(history["rain_mm"].tail(28).mean())

    return frame.sort_values([*series_cols, date_col]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Generate known-future covariate rows for the next forecast horizon.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--history", default=None, help="History CSV (default: config data.raw_path)")
    ap.add_argument("--output-csv", default="reports/future_covariates.csv")
    ap.add_argument("--output-json", default="reports/forecast_request.json")
    ap.add_argument("--series-limit", type=int, default=None, help="Only emit the first N series (for a small API smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("make_future_template", cfg)

    history_path = args.history or cfg["data"]["raw_path"]

    log_run_context(
        logger,
        "make_future_template",
        cfg,
        config_path=args.config,
        history=history_path,
        series_limit=args.series_limit,
    )

    history = read_raw(history_path, cfg)
    frame = build_future_frame(history, cfg, series_limit=args.series_limit)

    date_col = cfg["data"]["date_col"]
    fields = required_future_fields(cfg)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    payload = frame[fields].copy()
    payload[date_col] = payload[date_col].dt.strftime("%Y-%m-%d")

    body = {"future_covariates": payload.to_dict("records")}
    Path(args.output_json).write_text(json.dumps(body, indent=2), encoding="utf-8")

    log_settings(logger, "Generated forecast inputs", {
        "history_end": str(pd.to_datetime(history[date_col]).max().date()),
        "forecast_dates": f"{frame[date_col].min().date()}..{frame[date_col].max().date()}",
        "series": int(frame.groupby(cfg["data"]["series_cols"], dropna=False).ngroups),
        "rows": len(frame),
        "required_fields": fields,
        "csv": args.output_csv,
        "json": args.output_json,
    })


if __name__ == "__main__":
    main()
