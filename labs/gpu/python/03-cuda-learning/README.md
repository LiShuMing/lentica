# Python CUDA Learning in WSL Ubuntu

Project: `~/work/lentica/labs/gpu/python/03-cuda-learning`.
All Python dependencies live in this project's Linux `.venv`.

## Start

```bash
cd ~/work/lentica/labs/gpu/python/03-cuda-learning
source ./activate.sh
bash start-jupyter.sh
```

Use the localhost URL printed by Jupyter, including its authentication token.
Start with `notebooks/00_numpy_pandas.ipynb`, then open
`notebooks/01_cuda_basics.ipynb`, using **Python (CUDA Learning)**.
The server listens on `127.0.0.1:8891`. Press Ctrl+C to stop a foreground server.

## Validate

```bash
source ./activate.sh
python scripts/check_cuda.py
python scripts/check_cuda.py --require-gpu --output diagnostics.json
```

The normal check reports missing hardware without treating it as a Python
installation failure. The strict check exits nonzero until PyTorch and CuPy
both execute GPU matrix multiplications and validate results against NumPy.

## Components

- Python 3.12, NumPy, pandas, Matplotlib, and JupyterLab.
- PyTorch 2.9.1, CUDA 13.0 build from the official PyTorch index.
- CuPy CUDA 13.x, using the existing Linux Toolkit at `/usr/local/cuda`.
- A notebook covering CPU data, GPU array transfers, matrix multiplication,
  a custom CUDA RawKernel, and synchronized GPU timing.

`activate.sh` selects the project Toolkit instead of the older user-local
`nvcc`. To select another compatible CUDA 13 Toolkit, set
`CUDA_LEARNING_TOOLKIT` before sourcing it. The system Toolkit, PyTorch
CUDA runtime, and Windows driver are separate components with different
version numbers. Current system Toolkit is CUDA 13.3; actual GPU execution
must still be validated once NVIDIA hardware is accessible.

Windows currently reports no accessible NVIDIA device. Windows Toolkit 13.0
is installed, but that alone does not enable the GPU. WSL uses the Windows
NVIDIA driver; do not install a Linux display driver inside WSL.

NumPy and pandas use the CPU. CUDA library operations explicitly move data
to the GPU. The notebook prints SKIPPED for GPU cells without CUDA hardware.

## Rebuild

```bash
bash setup.sh
```

The setup uses the existing `uv` executable and records exact versions in
`requirements-lock.txt`. To reproduce that lock in a fresh venv:

```bash
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python --torch-backend cu130 -r requirements-lock.txt
```

References:
[PyTorch CUDA builds](https://pytorch.org/get-started/previous-versions/),
[CuPy installation](https://docs.cupy.dev/en/stable/install.html),
[NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
