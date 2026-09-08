FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HISTORY_CSV=/app/data/raw/data.csv \
    CONFIG_PATH=/app/configs/config.yaml \
    MODEL_REGISTRY_PATH=/app/configs/model_registry.yaml

RUN python -m pip install --upgrade pip

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "demand_forecasting.api:app", "--host", "0.0.0.0", "--port", "8000"]