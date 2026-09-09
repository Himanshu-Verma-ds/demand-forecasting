from __future__ import annotations

"""Register every trained model, not just the promoted champion.

select_model.py records the single model the API serves. That is the right contract for serving,
but it discards the rest of the evidence: which models were trained, how they scored, and where
their artifacts live. This writes a full registry so any model can be audited or reloaded, with
the champion flagged rather than being the only entry.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .config import load_config
from .logging_utils import log_run_context, log_settings, setup_logging


FAMILY_OF = {
    "lightgbm": "ml",
    "xgboost": "ml",
    "catboost": "ml",
    "lstm": "dl",
    "transformer": "dl",
    "tide": "darts",
    "tsmixer": "darts",
}

ARTIFACT_OF = {"ml": "model.joblib", "dl": "model.pt", "darts": "model.pt"}

METRIC_KEYS = ["wape", "mape", "mae", "rmse", "smape", "bias"]


def bias_against_actuals(path: Path, target_col: str) -> dict | None:
    """Total predicted vs total actual units - the number inventory decisions care about."""
    if not path.exists():
        return None

    frame = pd.read_csv(path)

    if target_col not in frame.columns or "prediction" not in frame.columns:
        return None

    actual = float(frame[target_col].sum())
    predicted = float(frame["prediction"].sum())

    return {
        "actual_units": round(actual, 2),
        "predicted_units": round(predicted, 2),
        "bias_pct": round((predicted - actual) / actual * 100, 4) if actual else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Register every trained model with its metrics and artifact path.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--artifacts", default=None, help="Artifact directory (default: config training.save_dir)")
    ap.add_argument("--output", default="configs/model_registry_all.yaml")
    ap.add_argument("--metric", default=None, help="Metric defining the champion (default: config training.selection_metric)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("register_all_models", cfg)

    metric = args.metric or cfg["training"].get("selection_metric", "wape")
    artifacts_dir = Path(args.artifacts or cfg["training"]["save_dir"])
    target_col = cfg["data"]["target_col"]

    log_run_context(
        logger,
        "register_all_models",
        cfg,
        config_path=args.config,
        artifacts_dir=str(artifacts_dir),
        output=args.output,
        selection_metric=metric,
    )

    entries = []

    for metrics_path in sorted(artifacts_dir.glob("*/metrics.json")):
        name = metrics_path.parent.name
        family = FAMILY_OF.get(name, "unknown")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        validation = payload.get("validation", {}).get("overall", {})
        test = payload.get("test", {}).get("overall", {})
        train = payload.get("train", {})

        artifact = metrics_path.parent / ARTIFACT_OF.get(family, "model.pt")

        entry = {
            "name": name,
            "family": family,
            "model_type": name,
            "artifact_path": str(artifact),
            "trained": artifact.exists(),
            "train_metrics": {k: round(v, 6) for k, v in train.items() if k in METRIC_KEYS},
            "validation_metrics": {k: round(v, 6) for k, v in validation.items() if k in METRIC_KEYS},
            "test_metrics": {k: round(v, 6) for k, v in test.items() if k in METRIC_KEYS},
        }

        params_path = artifacts_dir / "tuning" / f"{name}_best_params.json"

        if params_path.exists():
            entry["tuned_params"] = json.loads(params_path.read_text(encoding="utf-8"))
            entry["params_path"] = str(params_path)

        if family == "darts":
            entry["metadata_path"] = str(metrics_path.parent / "metadata.joblib")

        for split in ["validation", "test"]:
            bias = bias_against_actuals(metrics_path.parent / f"{split}_predictions.csv", target_col)
            if bias:
                entry[f"{split}_totals"] = bias

        entries.append(entry)

    if not entries:
        raise ValueError(f"No metrics.json found under {artifacts_dir}")

    entries.sort(key=lambda e: e["validation_metrics"].get(metric, float("inf")))

    for rank, entry in enumerate(entries, start=1):
        entry["rank_by_validation"] = rank
        entry["is_champion"] = rank == 1

    document = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "selection_metric": f"validation_{metric}",
        "artifacts_dir": str(artifacts_dir),
        "champion": entries[0]["name"],
        "model_count": len(entries),
        "models": entries,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    summary = pd.DataFrame([
        {
            "rank": e["rank_by_validation"],
            "model": e["name"],
            "family": e["family"],
            f"val_{metric}": e["validation_metrics"].get(metric),
            f"test_{metric}": e["test_metrics"].get(metric),
            "test_bias_pct": (e.get("test_totals") or {}).get("bias_pct"),
        }
        for e in entries
    ])

    logger.info("Registered %d models:\n%s", len(entries), summary.to_string(index=False))
    log_settings(logger, "Registry written", {"path": str(output_path), "champion": document["champion"]})


if __name__ == "__main__":
    main()
