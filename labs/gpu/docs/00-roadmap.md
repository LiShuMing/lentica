# Roadmap

This lab is tuned for an AMD GPU environment.

## Phase 1: Modern GPU Compute With Rust

Focus on `wgpu` and WGSL first. This avoids CUDA-specific assumptions and works well with AMD GPUs through DirectX 12 or Vulkan.

Exercises:

1. Vector add.
2. Parallel sum.
3. Matrix multiply.
4. Grayscale image filter.
5. Blur image filter.
6. Particle update compute pass.

Concepts:

- Host memory vs GPU buffers.
- Storage buffers.
- Bind groups and layouts.
- Compute pipelines.
- Workgroups and invocation IDs.
- Dispatch size.
- Readback buffers.

## Phase 2: Python Tensor Experiments

Use Python to build intuition quickly.

Exercises:

1. CPU tensor benchmark.
2. DirectML GPU tensor benchmark.
3. Matrix multiplication benchmark.
4. Small MLP forward pass.
5. Batch image filter.

Preferred starting stack on Windows:

- Python.
- PyTorch.
- `torch-directml`.

## Phase 3: ROCm and HIP

Treat ROCm/HIP as the advanced track. Use it after checking whether your GPU, driver, and OS combination is supported.

Exercises:

1. HIP vector add.
2. HIP matrix multiply.
3. Shared memory tiled matrix multiply.
4. Memory coalescing benchmark.
5. Basic profiling.

## Phase 4: Integration Projects

Pick one:

- Image processing workbench.
- Tiny tensor library.
- GPU particle simulator.
- Small neural network inference path.
- LLM kernel reading notes.
