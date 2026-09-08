from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_config
from .data import read_raw
from .inference_router import run_selected
from .logging_utils import log_run_context, setup_logging


LOGGER = setup_logging("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Record the paths and registry the service will serve from."""
    log_run_context(
        LOGGER,
        "api",
        history_csv=os.getenv("HISTORY_CSV", "data/raw/data.csv"),
        config_path=os.getenv("CONFIG_PATH", "configs/config.yaml"),
        registry_path=os.getenv("MODEL_REGISTRY_PATH", "configs/model_registry.yaml"),
    )

    yield


app = FastAPI(
    title="FMCG Demand Forecast API",
    version="1.0.0",
    lifespan=lifespan,
)


class ForecastRequest(BaseModel):
    """Future-known covariates for the requested Store-SKU forecast horizon."""

    future_covariates: list[dict[str, Any]] = Field(
        ...,
        description="Rows for each requested Store-SKU and each date in the configured forecast horizon.",
        min_length=1,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


def required_future_columns(cfg: dict) -> set[str]:
    """Return fields that must genuinely be known at forecast time."""
    data_cfg = cfg["data"]
    feature_cfg = cfg["features"]

    required = {
        data_cfg["date_col"],
        *data_cfg["series_cols"],
        "is_holiday",
    }

    if feature_cfg["promo_known_future"]:
        required.add("promo_flag")

    if feature_cfg["price_known_future"]:
        required.update({"list_price", "discount_pct"})

    if feature_cfg["weather_known_future"]:
        required.update({"temperature", "rain_mm"})

    return required


def validate_future_frame(future: pd.DataFrame, history: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Validate requested future rows and ensure a complete contiguous forecast horizon."""
    data_cfg = cfg["data"]

    date_col = data_cfg["date_col"]
    series_cols = data_cfg["series_cols"]
    horizon = int(data_cfg["horizon"])

    if future.empty:
        raise ValueError("future_covariates cannot be empty")

    missing_columns = sorted(required_future_columns(cfg).difference(future.columns))

    if missing_columns:
        raise ValueError(f"Missing required future fields: {missing_columns}")

    future = future.copy()
    future[date_col] = pd.to_datetime(future[date_col], errors="coerce")

    if future[date_col].isna().any():
        raise ValueError(f"Invalid or missing values found in {date_col}")

    if future.duplicated(subset=[*series_cols, date_col]).any():
        raise ValueError(f"Duplicate future rows found for {[*series_cols, date_col]}")

    history_end = pd.to_datetime(history[date_col]).max()
    expected_dates = pd.date_range(
        history_end + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    for key, group in future.groupby(series_cols, dropna=False):
        dates = pd.DatetimeIndex(sorted(group[date_col].unique()))

        if len(group) != horizon:
            raise ValueError(
                f"Series {key} must contain exactly {horizon} future rows; found {len(group)}"
            )

        if not dates.equals(expected_dates):
            raise ValueError(
                f"Series {key} must contain the contiguous forecast dates "
                f"{expected_dates.min().date()} to {expected_dates.max().date()}"
            )

    return future.sort_values([*series_cols, date_col]).reset_index(drop=True)


def attach_static_history(future: pd.DataFrame, history: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach the latest known static descriptors to future rows."""
    data_cfg = cfg["data"]

    date_col = data_cfg["date_col"]
    series_cols = data_cfg["series_cols"]

    static_candidates = [
        "country",
        "city",
        "channel",
        "latitude",
        "longitude",
        "sku_name",
        "category",
        "subcategory",
        "brand",
        "supplier_id",
        "purchase_cost",
        "margin_pct",
        "lead_time_days",
    ]

    static_cols = [column for column in static_candidates if column in history.columns]

    latest = (
        history
        .sort_values(date_col)
        .groupby(series_cols, as_index=False, dropna=False)
        .tail(1)
    )

    requested_series = future[series_cols].drop_duplicates()

    known_series = requested_series.merge(
        latest[series_cols].drop_duplicates(),
        on=series_cols,
        how="left",
        indicator=True,
    )

    unknown = known_series[known_series["_merge"].eq("left_only")]

    if not unknown.empty:
        raise ValueError(
            f"Unknown Store-SKU series requested: "
            f"{unknown[series_cols].to_dict('records')}"
        )

    if not static_cols:
        return future

    static_lookup = latest[[*series_cols, *static_cols]].copy()

    future = future.merge(
        static_lookup,
        on=series_cols,
        how="left",
        suffixes=("", "_history"),
        validate="many_to_one",
    )

    for column in static_cols:
        history_column = f"{column}_history"

        if history_column in future.columns:
            if column in future.columns:
                future[column] = future[column].fillna(future[history_column])
            else:
                future[column] = future[history_column]

            future = future.drop(columns=[history_column])

    return future


@app.post("/forecast")
def forecast(req: ForecastRequest):
    """Generate forecasts using the champion model registered in the model registry."""
    history_path = os.getenv("HISTORY_CSV", "data/raw/data.csv")
    config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
    registry_path = os.getenv("MODEL_REGISTRY_PATH", "configs/model_registry.yaml")

    if not Path(history_path).exists():
        raise HTTPException(status_code=500, detail=f"History file not found: {history_path}")

    if not Path(config_path).exists():
        raise HTTPException(status_code=500, detail=f"Config file not found: {config_path}")

    if not Path(registry_path).exists():
        raise HTTPException(status_code=500, detail=f"Model registry not found: {registry_path}")

    LOGGER.info("Forecast request received with %d future covariate rows", len(req.future_covariates))

    try:
        cfg = load_config(config_path)
        history = read_raw(history_path, cfg)

        future = pd.DataFrame(req.future_covariates)
        future = validate_future_frame(future, history, cfg)
        future = attach_static_history(future, history, cfg)

    except ValueError as exc:
        LOGGER.warning("Rejected forecast request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    future_path = tmp.name
    tmp.close()

    try:
        future.to_csv(future_path, index=False)

        output = run_selected(
            history_path=history_path,
            future_path=future_path,
            registry_path=registry_path,
            config_path=config_path,
        )

        date_col = cfg["data"]["date_col"]
        series_cols = cfg["data"]["series_cols"]
        output_columns = [date_col, *series_cols, "prediction"]

        missing_output = [column for column in output_columns if column not in output.columns]

        if missing_output:
            raise RuntimeError(f"Inference output is missing columns: {missing_output}")

        response = output[output_columns].copy()
        response[date_col] = pd.to_datetime(response[date_col]).dt.strftime("%Y-%m-%d")

        LOGGER.info(
            "Returned %d forecast rows for %d series",
            len(response),
            len(response[series_cols].drop_duplicates()),
        )

        return {
            "forecast_horizon": int(cfg["data"]["horizon"]),
            "forecast_count": len(response),
            "forecasts": response.to_dict("records"),
        }

    except ValueError as exc:
        LOGGER.warning("Rejected forecast request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        LOGGER.exception("Forecast generation failed")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {exc}") from exc

    finally:
        Path(future_path).unlink(missing_ok=True)