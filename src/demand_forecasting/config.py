from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_dl_params(cfg: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of cfg with tuned sequence-model hyperparameters applied.

    "lookback" is the input chunk length and lives under data; everything else is a
    dl.* setting. Returning a copy keeps each Optuna trial isolated from the next.
    """
    out = copy.deepcopy(cfg)

    for key, value in params.items():
        if key == "lookback":
            out["data"]["lookback"] = int(value)
        else:
            out["dl"][key] = value

    return out
