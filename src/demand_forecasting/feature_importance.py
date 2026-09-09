from __future__ import annotations

"""Feature importance for every trained model, for explainability.

Two methods, because the model families do not admit the same one:

  Tree models   - native gain importance. Gain is the total reduction in the training loss
                  contributed by every split on that feature, summed over all trees. It is
                  exact and free, but it is a statement about how the model was built, not
                  about how it behaves on unseen data.

  Sequence models - permutation importance on the validation window. Each input channel is
                  shuffled across series, the 14-day forecast is regenerated, and the rise in
                  validation WAPE is recorded. This measures what the model actually relies on
                  when predicting, which is the question explainability is usually asked.

Permutation numbers are therefore comparable within a model but not directly against gain.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import apply_dl_params, load_config
from .data import read_raw
from .logging_utils import log_run_context, log_settings, setup_logging
from .metrics import wape
from .splits import make_temporal_split


TREE_MODELS = ["lightgbm", "xgboost", "catboost"]
TORCH_MODELS = ["lstm", "transformer"]


def tree_importance(model_dir: Path, top_n: int) -> pd.DataFrame | None:
    """Native gain importance, mapped back to readable feature names."""
    from .models.ml import MLBundle

    path = model_dir / "model.joblib"

    if not path.exists():
        return None

    bundle = MLBundle.load(str(path))
    estimator = bundle.pipeline.named_steps["model"]

    # The ColumnTransformer emits numeric columns first, then categorical, in that order.
    names = list(bundle.numeric_features) + list(bundle.categorical_features)

    if bundle.model_name == "xgboost":
        # The sklearn wrapper defaults to weight; gain is the meaningful one.
        scores = estimator.get_booster().get_score(importance_type="gain")
        values = np.array([scores.get(f"f{i}", 0.0) for i in range(len(names))], dtype=float)
    elif bundle.model_name == "lightgbm":
        values = estimator.booster_.feature_importance(importance_type="gain").astype(float)
    else:
        values = np.asarray(estimator.get_feature_importance(), dtype=float)

    if len(values) != len(names):
        raise ValueError(f"{bundle.model_name}: {len(values)} importances vs {len(names)} names")

    total = values.sum()
    frame = pd.DataFrame({
        "feature": names,
        "importance": values,
        "importance_pct": 100.0 * values / total if total else 0.0,
    })

    return frame.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def torch_importance(model_name: str, model_dir: Path, cfg: dict, raw: pd.DataFrame,
                     split, top_n: int, seed: int) -> pd.DataFrame | None:
    """Permutation importance over the model's input channels, scored on validation WAPE."""
    import torch
    from torch.utils.data import DataLoader

    from .inference_dl import load_checkpoint
    from .models.dl_data import MultiSeriesWindowDataset
    from .train_dl import attach_prediction_metadata, enrich, origin_frame

    path = model_dir / "model.pt"

    if not path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, checkpoint = load_checkpoint(str(path), device)

    # The checkpoint's lookback is authoritative: it was tuned per model.
    trial_cfg = apply_dl_params(cfg, {"lookback": int(checkpoint["lookback"])})

    date_col = trial_cfg["data"]["date_col"]
    target_col = trial_cfg["data"]["target_col"]

    full = enrich(raw, trial_cfg)
    frame = origin_frame(full, split.val_start, split.val_end, trial_cfg)
    truth = raw[(raw[date_col] >= split.val_start) & (raw[date_col] <= split.val_end)].copy()

    dataset = MultiSeriesWindowDataset(frame, meta, trial_cfg)
    loader = DataLoader(dataset, batch_size=trial_cfg["dl"]["batch_size"], shuffle=False, num_workers=0)

    # Materialise once so each permutation is a cheap tensor shuffle rather than a rebuild.
    past, future, static, actual = [], [], [], []

    for batch in loader:
        past.append(batch["past_x"])
        future.append(batch["future_x"])
        static.append(batch["static_ids"])
        actual.append(batch["y"])

    past = torch.cat(past)
    future = torch.cat(future)
    static = torch.cat(static)

    def score(p, f, s) -> float:
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(p), 512):
                pb, fb, sb = p[i:i + 512].to(device), f[i:i + 512].to(device), s[i:i + 512].to(device)
                pred = model(pb, fb, sb, teacher_y=None, teacher_forcing=0.0) \
                    if model_name == "lstm" else model(pb, fb, sb)
                out.append(pred.cpu().numpy())

        flat = np.concatenate(out) * meta.target_std + meta.target_mean
        flat = np.clip(flat, 0.0, None)
        stacked = np.stack([np.zeros(flat.size), flat.reshape(-1)], axis=1)
        evaluation = attach_prediction_metadata(truth, stacked, trial_cfg)

        return wape(evaluation[target_col], evaluation["prediction"])

    baseline = score(past, future, static)
    rng = np.random.default_rng(seed)
    rows = []

    def permuted(tensor):
        order = rng.permutation(len(tensor))
        return tensor[torch.as_tensor(order)]

    for idx, name in enumerate(meta.past_cols):
        p = past.clone()
        p[:, :, idx] = permuted(past)[:, :, idx]
        rows.append({"feature": f"{name} (past window)", "wape_after": score(p, future, static)})

    for idx, name in enumerate(meta.future_cols):
        f = future.clone()
        f[:, :, idx] = permuted(future)[:, :, idx]
        rows.append({"feature": f"{name} (known future)", "wape_after": score(past, f, static)})

    for idx, name in enumerate(meta.static_cols):
        s = static.clone()
        s[:, idx] = permuted(static)[:, idx]
        rows.append({"feature": f"{name} (static)", "wape_after": score(past, future, s)})

    frame = pd.DataFrame(rows)
    frame["baseline_wape"] = baseline
    frame["wape_increase"] = frame["wape_after"] - baseline
    frame["importance_pct"] = 100.0 * frame["wape_increase"].clip(lower=0) / max(
        frame["wape_increase"].clip(lower=0).sum(), 1e-12)

    return frame.sort_values("wape_increase", ascending=False).head(top_n).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Compute top-N feature importance for every trained model.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--skip-torch", action="store_true", help="Tree models only (permutation is slow)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("feature_importance", cfg)

    artifacts_dir = Path(args.artifacts or cfg["training"]["save_dir"])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["project"]["random_seed"])

    log_run_context(logger, "feature_importance", cfg, artifacts_dir=str(artifacts_dir), top_n=args.top_n)

    collected = {}

    for name in TREE_MODELS:
        frame = tree_importance(artifacts_dir / name, args.top_n)

        if frame is None:
            logger.warning("Skipping %s - no model artifact", name)
            continue

        frame.insert(0, "model", name)
        frame.insert(1, "method", "native gain")
        collected[name] = frame
        logger.info("%s top features:\n%s", name, frame.to_string(index=False))

    if not args.skip_torch:
        raw = read_raw(cfg["data"]["raw_path"], cfg)
        split = make_temporal_split(raw, cfg)

        for name in TORCH_MODELS:
            frame = torch_importance(name, artifacts_dir / name, cfg, raw, split, args.top_n, seed)

            if frame is None:
                logger.warning("Skipping %s - no checkpoint", name)
                continue

            frame.insert(0, "model", name)
            frame.insert(1, "method", "permutation on validation WAPE")
            collected[name] = frame
            logger.info("%s top features:\n%s", name, frame.to_string(index=False))

    if not collected:
        raise ValueError(f"No models found under {artifacts_dir}")

    combined = pd.concat(collected.values(), ignore_index=True)
    path = outdir / "feature_importance.csv"
    combined.to_csv(path, index=False)

    log_settings(logger, "Feature importance written", {
        "path": str(path),
        "models": sorted(collected),
        "rows": len(combined),
    })


if __name__ == "__main__":
    main()
