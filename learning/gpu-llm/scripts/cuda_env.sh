#!/usr/bin/env bash
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="$CUDA_HOME"
if [[ -d "$CUDA_HOME/bin" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
