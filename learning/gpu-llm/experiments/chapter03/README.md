# Chapter 03 CUDA C++ Basics

本目录对应 `03-gpu-programming.md` 的 Week 4 可选扩展。

## 编译运行

如果当前 shell 找不到 `nvcc`，先加载项目里的 CUDA 环境：

```bash
cd ~/work/lentica/learning/gpu-llm
source ./scripts/cuda_env.sh
```

编译并运行：

```bash
cd experiments/chapter03
make run
```

也可以手动编译：

```bash
nvcc -O2 -std=c++17 cuda_basics.cu -o cuda_basics
./cuda_basics
```

## thread index 如何映射到数组元素

一维 vector add 使用下面的映射：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

- `threadIdx.x`: 当前 block 内的线程编号，范围是 `[0, blockDim.x)`。
- `blockIdx.x`: 当前 block 在 grid 里的编号。
- `blockDim.x`: 每个 block 有多少个线程。
- `idx`: 这个线程负责处理的全局数组元素下标。

例如 `blockDim.x = 256`：

- block 0 的 thread 0 处理 `idx = 0 * 256 + 0 = 0`。
- block 0 的 thread 255 处理 `idx = 0 * 256 + 255 = 255`。
- block 1 的 thread 0 处理 `idx = 1 * 256 + 0 = 256`。
- block 1 的 thread 17 处理 `idx = 1 * 256 + 17 = 273`。

因为数组长度不一定刚好等于 `gridDim.x * blockDim.x`，kernel 中需要检查：

```cpp
if (idx < n) {
    c[idx] = a[idx] + b[idx];
}
```

## global memory 与 shared memory

- vector add 直接从 global memory 读取 `a[idx]`、`b[idx]`，写回 `c[idx]`。global memory 容量大，但访问延迟高。
- tiled matrix multiplication 先把 A 和 B 的小块搬到 shared memory。shared memory 位于同一个 block 内可共享、访问更快，适合复用数据。

`cuda_basics.cu` 里包含：

- `vector_add_kernel`: 一维 grid/block 的 vector add。
- `matmul_tiled_kernel`: 使用 shared memory tile 的简单矩阵乘法。
