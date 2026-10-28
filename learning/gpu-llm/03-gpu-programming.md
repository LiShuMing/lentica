# Stage 3: GPU Programming and Performance Awareness

## 阶段目标

本阶段目标是建立 GPU 性能意识。主线不是马上写复杂 CUDA C++，而是理解 PyTorch 任务为什么快、为什么慢、为什么爆显存，以及如何做基本 profiling。

## 时间安排

默认 3-4 周，每周 8-10 小时。

## Week 1: GPU 执行模型

学习重点：

- CPU 与 GPU 的职责差异。
- kernel launch。
- 并行度、吞吐、延迟。
- host/device 数据传输。
- 为什么小 Tensor 操作可能不快。

练习：

- 比较 NumPy CPU、PyTorch CPU、PyTorch CUDA 的矩阵乘法耗时。
- 比较很多小矩阵乘法和一个大矩阵乘法的耗时。
- 观察 `nvidia-smi` 中显存和利用率变化。

验收标准：

- 能解释为什么 GPU 不一定让所有代码变快。
- 能说明数据频繁 CPU/GPU 传输的代价。

## Week 2: 显存、batch size 与混合精度

学习重点：

- 参数、梯度、optimizer state、activation 的显存来源。
- batch size 与 activation 显存。
- gradient accumulation。
- activation checkpointing。
- fp16/bf16。

练习：

- 在同一模型上逐步增大 batch size，记录显存变化。
- 用 gradient accumulation 模拟更大 batch。
- 对比 fp32、fp16/bf16 的显存差异。

验收标准：

- 能估算为什么训练比推理更吃显存。
- 遇到 CUDA OOM 时，能列出至少 5 个处理办法。

## Week 3: Profiling 与性能报告

学习重点：

- `torch.cuda.Event` 计时。
- `torch.profiler` 基础使用。
- warmup 的意义。
- 同步点：`torch.cuda.synchronize()`。
- 数据加载瓶颈。

练习：

- 对一个训练 step 做 profiling。
- 分析时间花在 data loading、forward、backward 还是 optimizer。
- 调整 `num_workers`、batch size、pin memory，观察变化。

实践项目：

- 写一份 CUDA 性能实验报告，包含：
  - 环境信息。
  - 测试代码。
  - CPU vs GPU 对比。
  - batch size 与显存曲线。
  - 一个明确的优化结论。

验收标准：

- 能用数据判断瓶颈，而不是猜。
- 能写出可复现实验。

## Week 4: CUDA C++ 可选扩展

前提：安装 CUDA Toolkit 并确认 `nvcc` 可用。

学习重点：

- CUDA C++ vector add。
- thread、block、grid。
- global memory。
- shared memory 概念。
- 简单 matrix multiplication。

验收标准：

- 能编译并运行一个 vector add kernel。
- 能解释 thread index 如何映射到数组元素。

配套代码：

- `experiments/chapter03/cuda_basics.cu`: CUDA C++ vector add、global memory 访问、shared memory tiled matrix multiplication。
- `experiments/chapter03/README.md`: 编译运行命令和 thread index 映射说明。

运行方式：

```bash
cd ~/work/lentica/learning/gpu-llm
source ./scripts/cuda_env.sh
cd experiments/chapter03
make run
```

## 常见坑

- 只看 GPU utilization，不看实际吞吐。
- 计时时忘记 GPU 异步执行。
- OOM 后只盲目减模型，不分析显存来源。
- 把 CUDA C++ 当成微调 LLM 的必要前置条件。

## 下一阶段前置条件

进入深度学习核心阶段前，应能用 PyTorch 训练模型、观察显存、计时，并解释 batch size、混合精度和 gradient accumulation 的作用。
