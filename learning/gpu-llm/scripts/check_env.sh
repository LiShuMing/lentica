#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cuda_env.sh"
PYTHON="${LENTICA_LEARNING_PYTHON:-python3}"
"$PYTHON" - <<'PY'
import importlib
import sys
print("Python:", sys.executable, sys.version.split()[0])
for name in ("numpy", "torch", "matplotlib", "pandas", "sklearn", "jupyterlab", "ipykernel"):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "available"))
        if name == "torch":
            print("CUDA available:", module.cuda.is_available())
    except ImportError:
        print(name, "not installed")
PY
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc unavailable; native CUDA experiments require an installed toolkit"
fi
