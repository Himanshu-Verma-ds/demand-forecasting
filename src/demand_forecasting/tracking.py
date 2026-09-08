from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import mlflow


def configure_mlflow(experiment: str) -> None:
    """Use local ./mlruns by default, or DagsHub/remote if env vars are set."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


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


def log_json_artifact(obj: Any, name: str, artifact_dir: str = "evaluation") -> None:
    tmp = Path("artifacts") / name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    mlflow.log_artifact(str(tmp), artifact_path=artifact_dir)
