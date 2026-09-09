from __future__ import annotations

"""Pick the best model in each family and report its test-set inference against actuals.

Answers three operational questions in one place:

  1. Which model won inside each family (ml / dl / darts), by the configured validation metric?
  2. What are its train / validation / test metrics side by side?
  3. How do its test predictions compare to the observed target, per row, per Store-SKU
     series and per horizon day?

Per-series and per-horizon breakdowns are computed here rather than in metrics.full_evaluation()
because they only matter once a champion exists, and they are large (one row per series).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .logging_utils import log_run_context, log_settings, setup_logging
from .metrics import regression_metrics


FAMILY_OF = {
    "lightgbm": "ml",
    "xgboost": "ml",
    "catboost": "ml",
    "lstm": "dl",
    "transformer": "dl",
    "tide": "darts",
    "tsmixer": "darts",
}

ARTIFACT_OF = {
    "ml": "model.joblib",
    "dl": "model.pt",
    "darts": "model.pt",
}

METRIC_KEYS = ["wape", "mape", "mae", "rmse", "smape", "bias"]


def collect_models(artifacts_dir: Path) -> pd.DataFrame:
    """Read every metrics.json under the artifact directory into one flat table."""
    rows = []

    for path in sorted(artifacts_dir.glob("*/metrics.json")):
        model = path.parent.name
        payload = json.loads(path.read_text(encoding="utf-8"))

        train = payload.get("train", {})
        validation = payload.get("validation", {})
        test = payload.get("test", {})

        val_overall = validation.get("overall", validation)
        test_overall = test.get("overall", test)

        row = {
            "model": model,
            "family": FAMILY_OF.get(model, "unknown"),
            "artifact_path": str(path.parent / ARTIFACT_OF.get(FAMILY_OF.get(model, ""), "model.pt")),
            "metrics_path": str(path),
            "test_predictions_path": str(path.parent / "test_predictions.csv"),
        }

        for key in METRIC_KEYS:
            row[f"train_{key}"] = train.get(key)
            row[f"val_{key}"] = val_overall.get(key)
            row[f"test_{key}"] = test_overall.get(key)

        row["val_mape_coverage"] = val_overall.get("mape_coverage")
        row["test_mape_coverage"] = test_overall.get("mape_coverage")

        rows.append(row)

    if not rows:
        raise ValueError(f"No metrics.json files found under {artifacts_dir}")

    return pd.DataFrame(rows)


def best_per_family(table: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    """Return the single best model within each family, ranked by the selection metric."""
    valid = table.dropna(subset=[rank_col])

    if valid.empty:
        raise ValueError(f"No model has a usable {rank_col}")

    best = (
        valid.sort_values(rank_col, ascending=True)
        .groupby("family", as_index=False, sort=False)
        .head(1)
        .sort_values(rank_col, ascending=True)
        .reset_index(drop=True)
    )

    best.insert(0, "overall_rank", range(1, len(best) + 1))

    return best


def load_test_predictions(best: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stack the per-row test predictions of every family champion against the actuals."""
    date_col = cfg["data"]["date_col"]
    series_cols = cfg["data"]["series_cols"]
    target_col = cfg["data"]["target_col"]

    frames = []

    for _, row in best.iterrows():
        path = Path(row["test_predictions_path"])

        if not path.exists():
            continue

        frame = pd.read_csv(path, parse_dates=[date_col])
        keep = [date_col, *series_cols, target_col, "prediction"]
        missing = [column for column in keep if column not in frame.columns]

        if missing:
            raise ValueError(f"{path} is missing columns {missing}")

        frame = frame[keep].copy()
        frame = frame.rename(columns={target_col: "actual"})
        frame["model"] = row["model"]
        frame["family"] = row["family"]
        frame["error"] = frame["prediction"] - frame["actual"]
        frame["abs_error"] = frame["error"].abs()

        # D+1 .. D+14 relative to the first scored date.
        frame["horizon_day"] = (frame[date_col] - frame[date_col].min()).dt.days + 1

        frames.append(frame)

    if not frames:
        raise ValueError("No test_predictions.csv found for any champion model")

    return pd.concat(frames, ignore_index=True)


def breakdown(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Metrics computed independently within each group, for every model."""
    rows = []

    for keys, group in predictions.groupby(["model", *group_cols], dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(["model", *group_cols], keys))
        record["n"] = len(group)
        record["actual_units"] = float(group["actual"].sum())
        record.update(regression_metrics(group["actual"], group["prediction"]))
        rows.append(record)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Report the best model per family and its test inference against actuals.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--artifacts", default=None, help="Artifact directory (default: config training.save_dir)")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--metric", default=None, help="Validation metric defining 'best' (default: config training.selection_metric)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("report_best_models", cfg)

    metric = args.metric or cfg["training"].get("selection_metric", "wape")
    rank_col = f"val_{metric}"
    artifacts_dir = Path(args.artifacts or cfg["training"]["save_dir"])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_run_context(
        logger,
        "report_best_models",
        cfg,
        config_path=args.config,
        artifacts_dir=str(artifacts_dir),
        outdir=str(outdir),
        selection_metric=metric,
    )

    table = collect_models(artifacts_dir)
    best = best_per_family(table, rank_col)

    all_models_path = outdir / "all_models_metrics.csv"
    best_path = outdir / "best_models.csv"

    table.sort_values(rank_col, na_position="last").to_csv(all_models_path, index=False)
    best.to_csv(best_path, index=False)

    logger.info(
        "Best model per family by %s:\n%s",
        rank_col,
        best[["overall_rank", "family", "model", f"train_{metric}", rank_col, f"test_{metric}"]].to_string(index=False),
    )

    predictions = load_test_predictions(best, cfg)

    predictions_path = outdir / "best_models_test_predictions.csv"
    per_series_path = outdir / "best_models_per_series_metrics.csv"
    per_horizon_path = outdir / "best_models_per_horizon_metrics.csv"

    predictions.to_csv(predictions_path, index=False)

    series_cols = cfg["data"]["series_cols"]
    per_series = breakdown(predictions, series_cols).sort_values(["model", "wape"], ascending=[True, False])
    per_horizon = breakdown(predictions, ["horizon_day"]).sort_values(["model", "horizon_day"])

    per_series.to_csv(per_series_path, index=False)
    per_horizon.to_csv(per_horizon_path, index=False)

    # Pooled WAPE is volume weighted; the macro average treats every series equally and
    # exposes failures on low-volume Store-SKU pairs that the pooled figure hides.
    summary = []

    for model, group in per_series.groupby("model"):
        pooled = predictions[predictions["model"].eq(model)]

        summary.append({
            "model": model,
            "series_count": len(group),
            "pooled_test_wape": regression_metrics(pooled["actual"], pooled["prediction"])["wape"],
            "macro_test_wape": float(group["wape"].mean()),
            "worst_series_wape": float(group["wape"].max()),
            "best_series_wape": float(group["wape"].min()),
        })

    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(outdir / "best_models_pooled_vs_macro.csv", index=False)

    logger.info("Pooled vs macro test WAPE:\n%s", summary_frame.to_string(index=False))
    logger.info("Per-horizon test WAPE:\n%s", per_horizon[["model", "horizon_day", "n", "wape", "mape"]].to_string(index=False))

    log_settings(logger, "Written reports", {
        "all_models": str(all_models_path),
        "best_models": str(best_path),
        "test_predictions": str(predictions_path),
        "per_series_metrics": str(per_series_path),
        "per_horizon_metrics": str(per_horizon_path),
        "pooled_vs_macro": str(outdir / "best_models_pooled_vs_macro.csv"),
    })


if __name__ == "__main__":
    main()
