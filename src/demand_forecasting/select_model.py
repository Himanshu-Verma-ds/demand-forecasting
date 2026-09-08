from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import yaml

from .config import load_config
from .logging_utils import log_run_context, log_settings, setup_logging


def main():
    ap = argparse.ArgumentParser(description="Promote the validation-selected champion to the model registry.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--comparison", default="reports/model_comparison.csv")
    ap.add_argument("--metadata-path", default=None)
    ap.add_argument("--model-type", default=None)
    ap.add_argument("--registry", default="configs/model_registry.yaml")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("select_model", cfg)

    log_run_context(
        logger,
        "select_model",
        cfg,
        config_path=args.config,
        name=args.name,
        family=args.family,
        artifact=args.artifact,
        comparison=args.comparison,
        registry=args.registry,
    )

    comparison = pd.read_csv(args.comparison)

    if comparison.empty:
        raise ValueError("Model comparison file is empty")

    if "model" not in comparison.columns or "val_wape" not in comparison.columns:
        raise ValueError("Comparison file must contain model and val_wape columns")

    valid = comparison.dropna(subset=["val_wape"]).sort_values("val_wape", ascending=True)

    if valid.empty:
        raise ValueError("No model has a valid validation WAPE")

    champion = valid.iloc[0]["model"]

    if args.name != champion:
        raise ValueError(f"{args.name} is not the validation-selected champion. Current champion is {champion}")

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    validation = metrics.get("validation", {})
    test = metrics.get("test", {})

    validation_metrics = validation.get("overall", validation)
    test_metrics = test.get("overall", test)

    selected_model = {
        "name": args.name,
        "family": args.family.lower(),
        "artifact_path": args.artifact,
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "validation_wape",
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "notes": "Selected using validation performance only. Test metrics are stored for final unbiased reporting and were not used for model selection.",
    }

    if args.metadata_path:
        selected_model["metadata_path"] = args.metadata_path

    if args.model_type:
        selected_model["model_type"] = args.model_type.lower()

    document = {"selected_model": selected_model}

    registry_path = Path(args.registry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    log_settings(logger, "Promoted model", selected_model)
    logger.info("Updated model registry at %s", registry_path)


if __name__ == "__main__":
    main()