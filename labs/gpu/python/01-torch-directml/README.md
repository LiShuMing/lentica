# PyTorch DirectML Benchmark

This is the easiest Python path for AMD GPUs on Windows.

Install from Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torch-directml
python benchmark.py
```

The script compares CPU tensor multiplication with a DirectML device when available.
