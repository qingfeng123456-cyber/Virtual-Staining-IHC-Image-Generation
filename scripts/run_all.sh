#!/usr/bin/env sh
set -eu
ENV_NAME="${ENV_NAME:-MEDICAL}"
CONFIG="${CONFIG:-configs/smoke.yaml}"
RUN_ID="${RUN_ID:-smoke_pipeline_acceptance}"
cd "$(dirname "$0")/.."
conda run -n "$ENV_NAME" python -m compileall src tests
conda run -n "$ENV_NAME" ruff check .
conda run -n "$ENV_NAME" pytest -q
conda run -n "$ENV_NAME" python -m virtual_staining.cli run-pipeline --config "$CONFIG" --run-id "$RUN_ID"

