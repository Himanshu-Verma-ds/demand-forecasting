from __future__ import annotations

from pathlib import Path

import yaml

from .config import load_config
from .data import read_raw
from .inference_darts import forecast as darts_forecast
from .inference_dl import forecast as dl_forecast
from .inference_ml import recursive_forecast
from .models.ml import MLBundle


def run_selected(
    history_path: str,
    future_path: str,
    registry_path: str = "configs/model_registry.yaml",
    config_path: str = "configs/config.yaml",
):
    """Load the registered champion model and route forecasting to the correct inference implementation."""
    registry_file = Path(registry_path)

    if not registry_file.exists():
        raise FileNotFoundError(f"Model registry not found: {registry_path}")

    registry = yaml.safe_load(registry_file.read_text(encoding="utf-8"))

    if not registry or "selected_model" not in registry:
        raise RuntimeError("No selected_model entry found in the model registry")

    selected = registry["selected_model"]

    if not selected.get("artifact_path"):
        raise RuntimeError("No artifact_path configured for the selected model")

    cfg = load_config(config_path)

    history = read_raw(history_path, cfg)
    future = read_raw(future_path, cfg)

    family = selected["family"].lower()
    model_name = selected["name"].lower()
    artifact_path = selected["artifact_path"]

    if family == "ml" or model_name in {"lightgbm", "xgboost", "catboost"}:
        bundle = MLBundle.load(artifact_path)
        return recursive_forecast(bundle, history, future, cfg)

    if family == "dl" or model_name in {"lstm", "transformer"}:
        model_type = selected.get("model_type", model_name)
        return dl_forecast(
            model_path=artifact_path,
            history=history,
            future=future,
            cfg=cfg,
        )

    if family == "darts" or model_name in {"tide", "tsmixer"}:
        metadata_path = selected.get("metadata_path")

        if not metadata_path:
            raise RuntimeError("Darts model registry entry requires metadata_path")

        model_type = selected.get("model_type", model_name)

        return darts_forecast(
            model_type=model_type,
            model_path=artifact_path,
            metadata_path=metadata_path,
            history=history,
            future=future,
            cfg=cfg,
        )

    raise NotImplementedError(f"No inference adapter configured for family={family}, model={model_name}")