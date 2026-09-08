from __future__ import annotations

import json
import logging
import logging.handlers
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


DEFAULT_LOG_DIR = "logs"
DEFAULT_LEVEL = "INFO"
DEFAULT_BACKUP_COUNT = 14

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED_STEPS: set[str] = set()


def _logging_cfg(cfg: dict | None) -> dict[str, Any]:
    """Return the logging section with defaults applied for configs that omit it."""
    section = (cfg or {}).get("logging") or {}

    return {
        "dir": section.get("dir", DEFAULT_LOG_DIR),
        "level": str(section.get("level", DEFAULT_LEVEL)).upper(),
        "console": bool(section.get("console", True)),
        "backup_count": int(section.get("backup_count", DEFAULT_BACKUP_COUNT)),
        "utc": bool(section.get("utc", False)),
    }


def setup_logging(step: str, cfg: dict | None = None) -> logging.Logger:
    """Configure one rotating log file per pipeline step and return its logger.

    Each step writes to logs/<step>.log. The handler rolls over at midnight, so the
    previous day is preserved as logs/<step>.log.YYYY-MM-DD and retained for
    logging.backup_count days.
    """
    settings = _logging_cfg(cfg)
    logger = logging.getLogger(f"demand_forecasting.{step}")

    if step in _CONFIGURED_STEPS:
        return logger

    log_dir = Path(settings["dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / f"{step}.log",
        when="midnight",
        interval=1,
        backupCount=settings["backup_count"],
        encoding="utf-8",
        utc=settings["utc"],
    )

    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)

    logger.setLevel(settings["level"])
    logger.handlers.clear()
    logger.addHandler(file_handler)

    if settings["console"]:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    _CONFIGURED_STEPS.add(step)

    return logger


def log_settings(logger: logging.Logger, title: str, payload: dict[str, Any]) -> None:
    """Write a labelled settings block as indented JSON so runs stay auditable from the log file."""
    logger.info("%s:\n%s", title, json.dumps(payload, indent=2, default=str, sort_keys=True))


def log_run_context(logger: logging.Logger, step: str, cfg: dict | None = None, **extra: Any) -> None:
    """Record the environment and the configuration that produced this run."""
    context = {
        "step": step,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "working_dir": str(Path.cwd()),
        **extra,
    }

    log_settings(logger, "Run context", context)

    if cfg is not None:
        log_settings(logger, "Effective configuration", cfg)


def log_metrics(logger: logging.Logger, title: str, metrics: dict[str, Any]) -> None:
    """Write a metric block. MLflow remains the system of record; the log keeps a local copy."""
    log_settings(logger, title, metrics)
