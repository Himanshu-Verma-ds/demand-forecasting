from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .logging_utils import log_run_context, setup_logging


def main():
    ap = argparse.ArgumentParser(description="Compare forecasting models using validation metrics for model selection.")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--output", default="reports/model_comparison.csv")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("evaluate_compare", cfg)
    log_run_context(
        logger,
        "evaluate_compare",
        cfg,
        config_path=args.config,
        artifacts_dir=args.artifacts,
        output=args.output,
    )

    rows = []

    for path in Path(args.artifacts).glob("*/metrics.json"):
        metrics = json.loads(path.read_text(encoding="utf-8"))

        train = metrics.get("train", {})
        validation = metrics.get("validation", {})
        test = metrics.get("test", {})

        val_overall = validation.get("overall", validation)
        test_overall = test.get("overall", test)
        val_business = validation.get("business_proxy") or {}
        test_business = test.get("business_proxy") or {}

        rows.append({
            "model": path.parent.name,
            "train_wape": train.get("wape"),
            "val_wape": val_overall.get("wape"),
            "val_mae": val_overall.get("mae"),
            "val_rmse": val_overall.get("rmse"),
            "val_smape": val_overall.get("smape"),
            "val_bias": val_overall.get("bias"),
            "val_underforecast_margin_risk": val_business.get("underforecast_margin_risk"),
            "val_overforecast_purchase_cost": val_business.get("overforecast_purchase_cost"),
            "test_wape": test_overall.get("wape"),
            "test_mae": test_overall.get("mae"),
            "test_rmse": test_overall.get("rmse"),
            "test_smape": test_overall.get("smape"),
            "test_bias": test_overall.get("bias"),
            "test_underforecast_margin_risk": test_business.get("underforecast_margin_risk"),
            "test_overforecast_purchase_cost": test_business.get("overforecast_purchase_cost"),
            "metrics_path": str(path),
        })

    if not rows:
        raise ValueError(f"No metrics.json files found under {args.artifacts}")

    out = pd.DataFrame(rows)

    if out["val_wape"].notna().sum() == 0:
        raise ValueError("No models contain validation WAPE values")

    out = out.sort_values("val_wape", ascending=True, na_position="last").reset_index(drop=True)
    out["selection_rank"] = out["val_wape"].rank(method="min", ascending=True, na_option="bottom").astype(int)

    columns = [
        "selection_rank",
        "model",
        "train_wape",
        "val_wape",
        "val_mae",
        "val_rmse",
        "val_smape",
        "val_bias",
        "val_underforecast_margin_risk",
        "val_overforecast_purchase_cost",
        "test_wape",
        "test_mae",
        "test_rmse",
        "test_smape",
        "test_bias",
        "test_underforecast_margin_risk",
        "test_overforecast_purchase_cost",
        "metrics_path",
    ]

    out = out[columns]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    logger.info("Model comparison (ranked by validation WAPE):\n%s", out.to_string(index=False))
    logger.info("Wrote comparison table to %s", args.output)
    logger.info("Validation-selected champion: %s", out.iloc[0]["model"])


if __name__ == "__main__":
    main()