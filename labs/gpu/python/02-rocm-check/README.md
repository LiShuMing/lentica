# ROCm PyTorch Check

Use this only after installing a ROCm-enabled PyTorch build.

```bash
python check_rocm.py
```

If `torch.cuda.is_available()` is true on a ROCm build, PyTorch is seeing the AMD GPU through the CUDA-compatible PyTorch API surface.
