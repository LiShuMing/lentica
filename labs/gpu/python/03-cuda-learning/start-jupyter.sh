#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./activate.sh
exec python -m jupyterlab --ip=127.0.0.1 --port="${1:-8891}" --no-browser --ServerApp.root_dir="${PWD}"
