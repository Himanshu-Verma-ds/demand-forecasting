from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import load_config
from .logging_utils import log_run_context, log_settings, setup_logging


EPS = 1e-6


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate Population Stability Index for a numeric feature."""
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()

    if len(reference_values) == 0 or len(current_values) == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_pct = np.histogram(reference_values, bins=edges)[0] / len(reference_values)
    current_pct = np.histogram(current_values, bins=edges)[0] / len(current_values)

    reference_pct = np.clip(reference_pct, EPS, None)
    current_pct = np.clip(current_pct, EPS, None)

    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """Calculate Population Stability Index for a categorical feature."""
    reference_values = reference.where(reference.notna(), "__MISSING__").astype(str)
    current_values = current.where(current.notna(), "__MISSING__").astype(str)

    categories = sorted(set(reference_values) | set(current_values))

    reference_pct = reference_values.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    current_pct = current_values.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()

    reference_pct = np.clip(reference_pct, EPS, None)
    current_pct = np.clip(current_pct, EPS, None)

    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


def psi_status(value: float, cfg: dict) -> str:
    """Convert PSI into an operational drift status using configured thresholds."""
    if not np.isfinite(value):
        return "unavailable"

    warning = float(cfg["monitoring"]["psi_warning"])
    alert = float(cfg["monitoring"]["psi_alert"])

    if value >= alert:
        return "alert"

    if value >= warning:
        return "warning"

    return "ok"


def missingness_report(reference: pd.Series, current: pd.Series) -> dict[str, float]:
    """Compare missing-value rates between reference and current data."""
    reference_rate = float(reference.isna().mean())
    current_rate = float(current.isna().mean())

    return {
        "reference_missing_rate": reference_rate,
        "current_missing_rate": current_rate,
        "absolute_change": current_rate - reference_rate,
    }


def numeric_drift(reference: pd.Series, current: pd.Series, cfg: dict) -> dict[str, float | str | dict]:
    """Calculate PSI, KS statistics, means, and missingness drift for a numeric feature."""
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()

    psi_value = psi(reference, current)

    if len(reference_values) > 0 and len(current_values) > 0:
        ks = ks_2samp(reference_values, current_values)
        ks_stat = float(ks.statistic)
        ks_pvalue = float(ks.pvalue)
    else:
        ks_stat = float("nan")
        ks_pvalue = float("nan")

    return {
        "psi": psi_value,
        "status": psi_status(psi_value, cfg),
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "reference_mean": float(np.mean(reference_values)) if len(reference_values) else float("nan"),
        "current_mean": float(np.mean(current_values)) if len(current_values) else float("nan"),
        "missingness": missingness_report(reference, current),
    }


def categorical_drift(reference: pd.Series, current: pd.Series, cfg: dict) -> dict[str, float | str | dict]:
    """Calculate categorical PSI and missingness drift."""
    psi_value = categorical_psi(reference, current)

    return {
        "psi": psi_value,
        "status": psi_status(psi_value, cfg),
        "missingness": missingness_report(reference, current),
    }


def promo_regime_report(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict) -> dict | None:
    """Measure whether the promotion regime changed materially."""
    if "promo_flag" not in reference.columns or "promo_flag" not in current.columns:
        return None

    reference_rate = float(pd.to_numeric(reference["promo_flag"], errors="coerce").mean())
    current_rate = float(pd.to_numeric(current["promo_flag"], errors="coerce").mean())

    relative_change = float((current_rate - reference_rate) / max(abs(reference_rate), EPS))
    threshold = float(cfg["monitoring"]["promo_rate_relative_change_alert"])

    return {
        "reference_rate": reference_rate,
        "current_rate": current_rate,
        "relative_change": relative_change,
        "absolute_relative_change": abs(relative_change),
        "alert_threshold": threshold,
        "status": "alert" if abs(relative_change) >= threshold else "ok",
    }


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict) -> dict:
    """Generate numeric, categorical, missingness, and regime drift diagnostics."""
    target_col = cfg["data"]["target_col"]

    numeric_candidates = [
        target_col,
        "list_price",
        "discount_pct",
        "temperature",
        "rain_mm",
        "stock_on_hand",
        "lead_time_days",
    ]

    categorical_candidates = [
        "channel",
        "category",
        "subcategory",
        "brand",
        "promo_flag",
        "stock_out_flag",
    ]

    report = {
        "numeric": {},
        "categorical": {},
        "promo_regime": None,
        "summary": {
            "warnings": [],
            "alerts": [],
        },
    }

    for column in numeric_candidates:
        if column not in reference.columns or column not in current.columns:
            continue

        result = numeric_drift(reference[column], current[column], cfg)
        report["numeric"][column] = result

        if result["status"] == "warning":
            report["summary"]["warnings"].append(f"numeric:{column}")

        if result["status"] == "alert":
            report["summary"]["alerts"].append(f"numeric:{column}")

    for column in categorical_candidates:
        if column not in reference.columns or column not in current.columns:
            continue

        result = categorical_drift(reference[column], current[column], cfg)
        report["categorical"][column] = result

        if result["status"] == "warning":
            report["summary"]["warnings"].append(f"categorical:{column}")

        if result["status"] == "alert":
            report["summary"]["alerts"].append(f"categorical:{column}")

    promo_report = promo_regime_report(reference, current, cfg)
    report["promo_regime"] = promo_report

    if promo_report and promo_report["status"] == "alert":
        report["summary"]["alerts"].append("promo_regime")

    report["summary"]["warning_count"] = len(report["summary"]["warnings"])
    report["summary"]["alert_count"] = len(report["summary"]["alerts"])

    return report


def main():
    ap = argparse.ArgumentParser(description="Compare reference and current datasets for feature and regime drift.")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output", default="reports/drift_report.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("drift", cfg)

    log_run_context(
        logger,
        "drift",
        config_path=args.config,
        reference=args.reference,
        current=args.current,
        output=args.output,
        psi_warning=cfg["monitoring"]["psi_warning"],
        psi_alert=cfg["monitoring"]["psi_alert"],
    )

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)

    logger.info("Comparing %s reference rows against %s current rows", f"{len(reference):,}", f"{len(current):,}")

    report = drift_report(reference, current, cfg)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log_settings(logger, "Drift report", report)
    logger.info(
        "Drift summary: %d warnings, %d alerts; report written to %s",
        report["summary"]["warning_count"],
        report["summary"]["alert_count"],
        output_path,
    )


if __name__ == "__main__":
    main()