#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${LENTICA_VLLM_PYTHON:-python3}" "$SCRIPT_DIR/service.py" health "$@"
