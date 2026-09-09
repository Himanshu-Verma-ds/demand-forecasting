from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv


LOGGER = logging.getLogger("demand_forecasting.tracking")


def configure_mlflow(experiment: str) -> None:
    """Point MLflow at the configured backend and select the experiment.

    Credentials are read from .env (DagsHub token, tracking URI) without overriding anything
    already exported in the shell. With MLFLOW_TRACKING_URI unset, MLflow 3.x defaults to a
    local sqlite store (./mlflow.db); the legacy ./mlruns file store is in maintenance mode
    and now raises unless MLFLOW_ALLOW_FILE_STORE=true.
    """
    # override=False so an explicitly exported variable always wins over the file.
    load_dotenv(override=False)

    uri = os.getenv("MLFLOW_TRACKING_URI")

    if uri:
        mlflow.set_tracking_uri(uri)

    mlflow.set_experiment(experiment)
    LOGGER.info("MLflow tracking URI: %s | experiment: %s", mlflow.get_tracking_uri(), experiment)


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
        else:
            out[key] = json.dumps(v, default=str)
    return out


def log_json_artifact(obj: Any, name: str, artifact_dir: str = "evaluation", save_dir: str | Path = "artifacts") -> None:
    """Write a JSON blob under save_dir and log it to the active MLflow run."""
    tmp = Path(save_dir) / name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    mlflow.log_artifact(str(tmp), artifact_path=artifact_dir)
