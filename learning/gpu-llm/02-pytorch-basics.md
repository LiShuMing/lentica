# Stage 2: PyTorch Basics

## 阶段目标

本阶段目标是掌握 PyTorch 的标准训练方式，并形成后续所有实验都能复用的训练模板。结束时你应该能独立写出模型、数据加载、训练、验证、保存、加载和 GPU 运行流程。

## 配套实现

代码位于 `experiments/chapter02/`：

- `tensor_autograd_basics.py`：Tensor 操作、autograd 梯度检查、线性回归、CPU/CUDA matmul 基准。
- `train_digits.py`：自定义 Dataset、DataLoader、MLP、训练/验证、checkpoint、恢复训练、独立评估和 AMP。
- `compare_metrics.py`：对比 fp32/AMP 的耗时、峰值显存和准确率。
- `README.md`：概念说明、逐周命令和验收方法。

完整运行：

```bash
cd ~/work/lentica/learning/gpu-llm/experiments/chapter02
make run
```

快速 smoke test：

```bash
make smoke
```

## 时间安排

默认 3-4 周，每周 8-10 小时。

## Week 1: Tensor、CUDA 与 autograd

学习重点：

- Tensor 创建、shape、dtype、device。
- CPU Tensor 与 CUDA Tensor 的区别。
- `requires_grad`、计算图、`backward()`。
- `torch.no_grad()`。
- 常见 Tensor 操作：view/reshape、permute、matmul、sum、mean。

练习：

- 把 NumPy 线性回归改写成 PyTorch。
- 在 CPU 和 CUDA 上分别跑矩阵乘法，比较耗时。
- 手动检查一个简单函数的梯度是否符合预期。

验收标准：

- 能解释 Tensor 的 `device` 和 `dtype`。
- 能说明为什么验证阶段要用 `torch.no_grad()`。
- 能定位 CPU/GPU device mismatch 错误。

## Week 2: nn.Module 与训练循环

学习重点：

- `nn.Module`、`forward()`、parameters。
- 常用层：Linear、ReLU、Dropout、LayerNorm。
- loss function 和 optimizer。
- `model.train()` 与 `model.eval()`。
- 标准训练循环：zero_grad、forward、loss、backward、step。

练习：

- 用 PyTorch 训练 MNIST 或 sklearn digits 分类器。
- 记录 train loss、val loss、val accuracy。
- 保存并加载 checkpoint。

验收标准：

- 能不用复制模板写出完整训练循环。
- 能解释 optimizer 为什么需要 `zero_grad()`。
- 能区分训练模式和评估模式。

## Week 3: Dataset、DataLoader 与工程化模板

学习重点：

- 自定义 Dataset。
- DataLoader、batch、shuffle。
- train/validation/test split。
- checkpoint 文件结构。
- 简单配置管理：学习率、batch size、epoch、seed。

实践项目：

- 写一个可复用训练模板，支持：
  - GPU 自动选择。
  - train/eval 两个入口。
  - checkpoint 保存和恢复。
  - 指标日志。
  - 随机种子设置。

验收标准：

- 更换模型或数据集时，不需要重写整个训练脚本。
- 训练中断后能从 checkpoint 恢复。
- 能输出清晰的最终评估结果。

## Week 4: 混合精度入门

学习重点：

- fp32、fp16、bf16 的区别。
- `torch.amp.autocast`。
- `GradScaler` 的用途。
- 显存占用和 batch size 的关系。

练习：

- 同一个模型分别用 fp32 和混合精度训练。
- 记录训练速度、显存占用和精度差异。

验收标准：

- 能说明混合精度为什么省显存。
- 能知道什么时候先不用混合精度排查问题。

## 常见坑

- 忘记把数据和模型放到同一个 device。
- 在验证阶段还在计算梯度。
- loss 下降但 validation accuracy 不涨，却不知道看数据或指标。
- checkpoint 只保存模型参数，不保存 optimizer 和 epoch。

## 下一阶段前置条件

进入 GPU 性能阶段前，应能稳定使用 PyTorch CUDA 完成一个小模型训练，并知道如何观察显存和运行时间。
