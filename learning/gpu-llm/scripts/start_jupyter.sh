#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cuda_env.sh"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
exec "${LENTICA_LEARNING_PYTHON:-python3}" -m jupyter lab --notebook-dir "$PROJECT_DIR" \
    --ip 127.0.0.1 --port "${LENTICA_JUPYTER_PORT:-8888}" --no-browser
