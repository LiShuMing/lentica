# GPU Lab

This repository is a learning lab for AMD GPU programming on a Windows + WSL setup.

The main idea is to keep notes and source code in WSL, while choosing the most reliable runtime for each track:

- Rust + wgpu + WGSL for modern cross-platform GPU compute.
- Python + DirectML for approachable AMD GPU tensor experiments on Windows.
- ROCm/HIP as an advanced AMD-specific track when the GPU and driver stack support it.

## Tracks

| Track | Path | Goal |
| --- | --- | --- |
| Modern GPU compute | `rust/01-wgpu-vector-add` | Learn buffers, bind groups, compute pipelines, workgroups, and dispatch. |
| Python tensor compute | `python/01-torch-directml` | Compare CPU and AMD GPU tensor execution through DirectML. |
| ROCm capability check | `python/02-rocm-check` | Detect whether PyTorch can see a ROCm/HIP GPU. |
| AMD low-level compute | `hip/01-vector-add` | Practice CUDA-like AMD HIP kernels when ROCm is available. |

## Recommended Workflow

Use WSL for the repository, Git, shell scripts, notes, and most editing:

```bash
cd ~/work/lentica/labs/gpu
```

Run `wgpu` examples on Windows when you want the most reliable AMD path through DX12/Vulkan. Run ROCm/HIP experiments in WSL only after confirming your GPU and driver support it.

When running Rust from Windows against this WSL-hosted repository, keep Cargo build output on a Windows filesystem. The helper script does this for the first `wgpu` exercise:

```powershell
.\tools\run-wgpu-vector-add.ps1
```

## First Milestone

1. Record your GPU and driver information in `docs/01-amd-stack.md`.
2. Run the environment check script in `tools/check-env.ps1` from Windows PowerShell.
3. Run the Rust `wgpu` vector-add project.
4. Run the Python DirectML benchmark.
5. Compare CPU and GPU timings in `docs/02-lab-notes.md`.
