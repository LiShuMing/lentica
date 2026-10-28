# Chapter 02 PyTorch Basics

本目录覆盖 `02-pytorch-basics.md` 的四周内容，并提供一个可复用的小型分类训练模板。数据使用 scikit-learn 自带的 digits，不需要联网下载。

## 快速运行

```bash
cd ~/work/lentica/learning/gpu-llm/experiments/chapter02
make run
```

快速检查代码路径是否正常：

```bash
make smoke
```

## Week 1: Tensor、CUDA 与 autograd

```bash
python tensor_autograd_basics.py
```

脚本包含 shape/dtype/device、broadcast、reshape/permute、手动梯度校验、Tensor 版线性回归，以及 CPU/CUDA 矩阵乘法计时。

- `dtype` 决定元素的数据类型、数值范围、精度和单个元素占用的字节数。
- `device` 决定 Tensor 的存储和计算位置。参与同一次运算的 Tensor 通常必须位于同一设备。
- 验证时使用 `torch.no_grad()`，因为不需要反向传播；它会避免保存计算图中的中间结果，减少内存和额外计算。
- device mismatch 通常表现为模型在 CUDA、输入还在 CPU。检查 `next(model.parameters()).device` 和 `batch.device`，再用 `.to(device)` 统一。

## Week 2-3: 完整训练模板

训练：

```bash
python train_digits.py train \
  --epochs 30 \
  --batch-size 128 \
  --device auto \
  --output-dir ../../outputs/chapter02
```

评估保存的最佳模型：

```bash
python train_digits.py eval \
  --checkpoint ../../outputs/chapter02/best.pt
```

从最近一次 checkpoint 接着训练到第 40 个 epoch：

```bash
python train_digits.py train \
  --epochs 40 \
  --resume ../../outputs/chapter02/latest.pt \
  --output-dir ../../outputs/chapter02
```

训练模板包含：

- 自定义 `DigitsDataset` 和 train/validation/test split。
- 可替换的 `DigitsMLP(nn.Module)`。
- `model.train()`、`model.eval()` 与 `torch.no_grad()`。
- `zero_grad -> forward -> loss -> backward -> optimizer.step` 标准循环。
- 自动选择 CPU/CUDA、固定随机种子、DataLoader shuffle。
- 每轮 train/validation 指标和耗时日志。
- `latest.pt`、`best.pt` 与 `metrics.json`。
- checkpoint 中保存 epoch、模型、optimizer、GradScaler、配置和历史指标。

`optimizer.zero_grad()` 必不可少，因为 PyTorch 默认把新梯度累加到参数已有的 `.grad` 中。`model.train()` 启用 Dropout 等训练行为；`model.eval()` 切换到确定性的评估行为。

## Week 4: 混合精度

分别运行 fp32 和 AMP，然后比较日志里的耗时、精度以及 `metrics.json` 中的峰值 GPU 显存：

```bash
python train_digits.py train \
  --precision fp32 --output-dir ../../outputs/chapter02/fp32

python train_digits.py train \
  --precision amp --output-dir ../../outputs/chapter02/amp

python compare_metrics.py \
  ../../outputs/chapter02/fp32/metrics.json \
  ../../outputs/chapter02/amp/metrics.json
```

CUDA AMP 使用 fp16 autocast 和 `GradScaler`；CPU AMP 使用 bf16 autocast，并关闭 scaler。autocast 让适合低精度的算子使用较小的数据类型，通常能降低激活显存；`GradScaler` 放大 loss，减少 fp16 梯度下溢。排查 loss 为 NaN、精度异常或算子兼容问题时，先退回 fp32 建立基线。

digits MLP 很小，kernel 启动和数据搬运开销占比高，因此 AMP 不一定更快或明显节省显存。这个实验用于掌握 API 和测量方法，不应被当作大型模型的性能结论。对比工具使用每轮耗时中位数，降低首轮 CUDA 初始化开销对结果的影响。

## 验收

完整运行后应看到：

- Week 1 脚本输出 `all Chapter 02 Week 1 checks passed`。
- digits 模型能稳定达到约 95% 或更高的 test accuracy。
- `eval` 能独立加载 `best.pt` 并输出 test loss/accuracy。
- 使用 `--resume latest.pt` 后，epoch 从 checkpoint 的下一轮继续。
