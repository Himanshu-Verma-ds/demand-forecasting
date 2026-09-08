#!/usr/bin/env bash
set -euo pipefail
: "${DAGSHUB_USER:?Set DAGSHUB_USER}"
: "${DAGSHUB_REPO:?Set DAGSHUB_REPO}"
: "${DAGSHUB_TOKEN:?Set DAGSHUB_TOKEN}"
export MLFLOW_TRACKING_URI="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.mlflow"
export MLFLOW_TRACKING_USERNAME="${DAGSHUB_USER}"
export MLFLOW_TRACKING_PASSWORD="${DAGSHUB_TOKEN}"
echo "MLflow configured for ${MLFLOW_TRACKING_URI}"
echo "For DVC storage, open the DagsHub repository Remote/Storage instructions and copy its generated DVC remote command; keep credentials local (never commit them)."
