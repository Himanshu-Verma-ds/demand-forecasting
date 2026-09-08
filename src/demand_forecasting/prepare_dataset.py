from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import read_raw, prepare_features
from .logging_utils import log_run_context, log_settings, setup_logging
from .splits import make_temporal_split, label_split


def main():
    ap = argparse.ArgumentParser(description="Prepare leakage-safe forecasting dataset.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output", default="data/processed/features.parquet")

    a = ap.parse_args()

    cfg = load_config(a.config)
    logger = setup_logging("prepare_dataset", cfg)
    log_run_context(logger, "prepare_dataset", cfg, config_path=a.config, output=a.output)

    raw = read_raw(cfg["data"]["raw_path"], cfg)
    logger.info("Loaded %s raw rows from %s", f"{len(raw):,}", cfg["data"]["raw_path"])

    split = make_temporal_split(raw, cfg)

    log_settings(logger, "Temporal split", {
        "train_end": str(split.train_end.date()),
        "val_start": str(split.val_start.date()),
        "val_end": str(split.val_end.date()),
        "test_start": str(split.test_start.date()),
        "test_end": str(split.test_end.date()),
    })

    feat = prepare_features(raw, cfg)
    feat["split"] = label_split(feat, split, cfg)

    log_settings(logger, "Split row counts", feat["split"].value_counts().to_dict())

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(a.output, index=False)

    logger.info("Wrote %s rows and %d columns to %s", f"{len(feat):,}", feat.shape[1], a.output)


if __name__ == "__main__":
    main()
