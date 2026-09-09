from __future__ import annotations

"""Produce the deliverable 14-day Store-SKU forecast from each family's champion model.

Takes the champions chosen by report_best_models.py, builds the known-future covariates for the
horizon after history ends, and runs each champion through its own inference adapter. Emits one
tidy forecast file covering every active Store-SKU pair, plus a consolidated metrics file.
"""

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import read_raw
from .logging_utils import log_run_context, log_settings, setup_logging
from .make_future_template import build_future_frame


def forecast_one(model_row: pd.Series, history: pd.DataFrame, future: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Route one champion to its family's inference implementation."""
    family = model_row["family"]
    artifact = model_row["artifact_path"]

    if family == "ml":
        from .inference_ml import recursive_forecast
        from .models.ml import MLBundle

        return recursive_forecast(MLBundle.load(artifact), history, future.copy(), cfg)

    if family == "dl":
        from .inference_dl import forecast as dl_forecast

        return dl_forecast(artifact, history, future.copy(), cfg)

    if family == "darts":
        from .inference_darts import forecast as darts_forecast

        metadata = str(Path(artifact).parent / "metadata.joblib")

        return darts_forecast(
            model_type=model_row["model"],
            model_path=artifact,
            metadata_path=metadata,
            history=history,
            future=future.copy(),
            cfg=cfg,
        )

    raise ValueError(f"Unknown family: {family}")


def main():
    ap = argparse.ArgumentParser(description="Generate the final 14-day Store-SKU forecast from each family champion.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--best-models", default="reports/best_models.csv")
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("run_final_inference", cfg)

    date_col = cfg["data"]["date_col"]
    series_cols = cfg["data"]["series_cols"]

    log_run_context(logger, "run_final_inference", cfg, config_path=args.config, best_models=args.best_models)

    best = pd.read_csv(args.best_models)
    history = read_raw(cfg["data"]["raw_path"], cfg)
    future = build_future_frame(history, cfg)

    logger.info(
        "Forecasting %s..%s for %d Store-SKU series",
        future[date_col].min().date(),
        future[date_col].max().date(),
        future.groupby(series_cols, dropna=False).ngroups,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []

    for _, row in best.iterrows():
        if not Path(row["artifact_path"]).exists():
            logger.warning("Skipping %s - artifact missing at %s", row["model"], row["artifact_path"])
            continue

        logger.info("Forecasting with %s (%s)", row["model"], row["family"])

        try:
            output = forecast_one(row, history, future, cfg)
        except Exception:
            # One family failing must not lose the others' forecasts.
            logger.exception("Inference failed for %s", row["model"])
            continue

        frame = output[[date_col, *series_cols, "prediction"]].copy()
        frame["model"] = row["model"]
        frame["family"] = row["family"]
        frames.append(frame)

        logger.info("  -> %s rows", f"{len(frame):,}")

    if not frames:
        raise RuntimeError("No champion produced a forecast")

    forecasts = pd.concat(frames, ignore_index=True)
    forecasts = forecasts.sort_values(["model", *series_cols, date_col]).reset_index(drop=True)

    long_path = outdir / "forecast_store_sku.csv"
    forecasts.to_csv(long_path, index=False)

    # Wide view: one row per Store-SKU-date, one column per model, for easy comparison.
    wide = forecasts.pivot_table(
        index=[date_col, *series_cols],
        columns="model",
        values="prediction",
    ).reset_index()

    wide.columns.name = None
    wide_path = outdir / "forecast_store_sku_wide.csv"
    wide.to_csv(wide_path, index=False)

    # Consolidated Store-SKU view: how each series is forecast to behave next, alongside how
    # accurately that series was predicted on the held-out test window.
    summary = (
        forecasts.groupby(["model", "family", *series_cols], as_index=False)["prediction"]
        .agg(forecast_total_units="sum", forecast_mean_units="mean")
    )

    per_series_path = Path(args.outdir) / "best_models_per_series_metrics.csv"

    if per_series_path.exists():
        per_series = pd.read_csv(per_series_path)
        keep = ["model", *series_cols, "n", "actual_units", "wape", "mape", "mae", "rmse", "bias"]
        keep = [c for c in keep if c in per_series.columns]

        summary = summary.merge(
            per_series[keep].rename(columns={
                "n": "test_rows",
                "actual_units": "test_actual_units",
                "wape": "test_wape",
                "mape": "test_mape",
                "mae": "test_mae",
                "rmse": "test_rmse",
                "bias": "test_bias",
            }),
            on=["model", *series_cols],
            how="left",
        )

    summary = summary.sort_values(["model", "test_wape"], ascending=[True, False], na_position="last")
    summary_path = Path(args.outdir) / "store_sku_forecast_and_metrics.csv"
    summary.to_csv(summary_path, index=False)

    logger.info(
        "Store-SKU summary: %d rows across %d models",
        len(summary),
        summary["model"].nunique(),
    )

    log_settings(logger, "Final forecast written", {
        "store_sku_summary": str(summary_path),
        "long": str(long_path),
        "wide": str(wide_path),
        "models": sorted(forecasts["model"].unique()),
        "rows": len(forecasts),
        "series": int(forecasts.groupby(series_cols, dropna=False).ngroups),
        "horizon_dates": f"{forecasts[date_col].min().date()}..{forecasts[date_col].max().date()}",
    })


if __name__ == "__main__":
    main()
