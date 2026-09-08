from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import read_raw
from .logging_utils import log_run_context, setup_logging


STRATEGIES = {
    "global_store_sku": ["store_id", "sku_id"],
    "model_per_subcategory": ["subcategory"],
    "model_per_sku_shared_stores": ["sku_id"],
    "local_store_sku": ["store_id", "sku_id"],
    "middle_level_store_subcategory": ["store_id", "subcategory"],
}


def strategy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Report how many models or series each modelling strategy would create."""
    rows = []

    for name, group_cols in STRATEGIES.items():
        sizes = df.groupby(group_cols, dropna=False).size()

        rows.append({
            "strategy": name,
            "grouping": "+".join(group_cols),
            "n_models_or_series": len(sizes),
            "min_rows": int(sizes.min()),
            "median_rows": float(sizes.median()),
            "max_rows": int(sizes.max()),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Compare the operational size of each modelling strategy.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output", default="reports/strategy_analysis.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("strategy_analysis", cfg)
    log_run_context(logger, "strategy_analysis", cfg, config_path=args.config, output=args.output)

    df = read_raw(cfg["data"]["raw_path"], cfg)
    table = strategy_table(df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    logger.info("Modelling strategy footprint:\n%s", table.to_string(index=False))
    logger.info("Wrote strategy comparison to %s", args.output)


if __name__ == "__main__":
    main()
