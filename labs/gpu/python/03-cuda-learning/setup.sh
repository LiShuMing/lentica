#!/usr/bin/env bash
set -euo pipefail
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
UV="${HOME}/.local/bin/uv"
if [[ ! -x "${UV}" ]]; then
    UV="$(command -v uv)"
fi
if [[ ! -x .venv/bin/python ]]; then
    "${UV}" venv --python /usr/bin/python3 .venv
fi
"${UV}" pip install --python .venv/bin/python -r requirements.txt
"${UV}" pip install --python .venv/bin/python --torch-backend cu130 -r requirements-torch.txt
source ./activate.sh
python -m ipykernel install --sys-prefix --name cuda-learning --display-name "Python (CUDA Learning)"
"${UV}" pip freeze --python .venv/bin/python > requirements-lock.txt
python scripts/check_cuda.py
