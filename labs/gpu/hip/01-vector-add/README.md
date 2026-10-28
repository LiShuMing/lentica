# HIP Vector Add

This is the low-level AMD track. Use it after ROCm/HIP is installed and `hipcc --version` works.

Build:

```bash
hipcc vector_add.cpp -o vector_add
./vector_add
```

HIP uses a CUDA-like programming model:

- `blockIdx`
- `blockDim`
- `threadIdx`
- device memory allocation
- host-to-device and device-to-host copies
- kernel launch syntax
