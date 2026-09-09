"""Standalone 14-day Store-SKU forecast from the bundled v2 champion models.

Everything needed to reproduce the out-of-sample forecast lives in this folder except the raw
history, which stays DVC-tracked (212 MB) rather than copied.

    # from the project root, with the venv active
    export PYTHONPATH=src
    python final_conclusion/inference_bundle/generate_forecast.py \
        --history data/raw/data.csv \
        --model catboost \
        --output my_forecast.csv

--model all runs every bundled champion (catboost / transformer / tide).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from demand_forecasting.config import load_config
from demand_forecasting.data import read_raw
from demand_forecasting.make_future_template import build_future_frame

BUNDLE = Path(__file__).resolve().parent

# family decides which inference adapter is used; each has a different leakage contract.
# All six trained models are bundled, not just the per-family winners, so any of them can be
# reloaded and compared without retraining.
MODELS = {
    "lightgbm": ("ml", BUNDLE / "models/lightgbm/model.joblib"),
    "xgboost": ("ml", BUNDLE / "models/xgboost/model.joblib"),
    "catboost": ("ml", BUNDLE / "models/catboost/model.joblib"),
    "lstm": ("dl", BUNDLE / "models/lstm/model.pt"),
    "transformer": ("dl", BUNDLE / "models/transformer/model.pt"),
    "tide": ("darts", BUNDLE / "models/tide/model.pt"),
}


def forecast_one(name: str, history: pd.DataFrame, future: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    family, artifact = MODELS[name]

    if family == "ml":
        from demand_forecasting.inference_ml import recursive_forecast
        from demand_forecasting.models.ml import MLBundle

        return recursive_forecast(MLBundle.load(str(artifact)), history, future.copy(), cfg)

    if family == "dl":
        from demand_forecasting.inference_dl import forecast as dl_forecast

        return dl_forecast(str(artifact), history, future.copy(), cfg)

    from demand_forecasting.inference_darts import forecast as darts_forecast

    return darts_forecast(
        model_type=name,
        model_path=str(artifact),
        metadata_path=str(artifact.parent / "metadata.joblib"),
        history=history,
        future=future.copy(),
        cfg=cfg,
    )


def main():
    ap = argparse.ArgumentParser(description="Generate a 14-day Store-SKU forecast from the bundled models.")
    ap.add_argument("--history", default="data/raw/data.csv", help="History CSV (DVC-tracked)")
    ap.add_argument("--config", default=str(BUNDLE / "config_v2.yaml"))
    ap.add_argument("--model", default="catboost", choices=[*MODELS, "all"])
    ap.add_argument("--future", default=None, help="Known-future covariates CSV (default: rebuild from history)")
    ap.add_argument("--output", default="forecast_store_sku.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    date_col = cfg["data"]["date_col"]
    series_cols = cfg["data"]["series_cols"]

    history = read_raw(args.history, cfg)

    # Rebuilding is preferred: it guarantees the horizon starts the day after history ends.
    if args.future:
        future = read_raw(args.future, cfg)
    else:
        future = build_future_frame(history, cfg)

    print(f"history ends {pd.to_datetime(history[date_col]).max().date()}")
    print(f"forecasting  {future[date_col].min().date()} .. {future[date_col].max().date()} "
          f"for {future.groupby(series_cols, dropna=False).ngroups} Store-SKU series")

    names = list(MODELS) if args.model == "all" else [args.model]
    frames = []

    for name in names:
        print(f"  running {name} ...", flush=True)
        out = forecast_one(name, history, future, cfg)
        frame = out[[date_col, *series_cols, "prediction"]].copy()
        frame["model"] = name
        frames.append(frame)
        print(f"    -> {len(frame):,} rows, {frame['prediction'].sum():,.0f} total units")

    result = pd.concat(frames, ignore_index=True).sort_values(["model", *series_cols, date_col])
    result.to_csv(args.output, index=False)
    print(f"wrote {len(result):,} rows to {args.output}")


if __name__ == "__main__":
    main()
