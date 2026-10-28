#!/usr/bin/env bash
_CUDA_LEARNING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_CUDA_LEARNING_DIR}/.venv/bin/activate"
export CUDA_PATH="${CUDA_LEARNING_TOOLKIT:-/usr/local/cuda}"
export PATH="${CUDA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _CUDA_LEARNING_DIR
