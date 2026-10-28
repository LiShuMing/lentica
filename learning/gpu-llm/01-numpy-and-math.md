# Stage 1: NumPy and Math Foundations

## 阶段目标

本阶段目标是补齐后续深度学习必需的数值计算能力。结束时你应该能熟练使用 NumPy 做矩阵运算，并能不用 PyTorch 写出一个小型 MLP 的训练过程。

## 时间安排

默认 2-3 周，每周 8-10 小时。

## Week 1: NumPy 基础与向量化

学习重点：

- `ndarray`、shape、dtype、axis。
- slicing、reshape、transpose、concatenate。
- broadcasting 规则。
- 向量化计算替代 Python for-loop。
- 随机数、均值、方差、标准化。

练习：

- 创建不同 shape 的矩阵，手动推导每次运算后的 shape。
- 用 for-loop 和 NumPy 向量化分别实现欧氏距离，比较速度。
- 实现数据标准化：`(x - mean) / std`。

验收标准：

- 能解释 `(32, 10) @ (10, 3)` 的输出 shape。
- 能判断 broadcasting 是否成立。
- 能把简单 for-loop 改写成 NumPy 向量化版本。

## Week 2: 机器学习常用数学组件

学习重点：

- 线性模型：`y = xW + b`。
- sigmoid、ReLU、softmax。
- MSE、cross entropy。
- 梯度下降基本思想。
- 数值稳定性：softmax 前减最大值。

练习：

- 用 NumPy 实现线性回归训练。
- 用 NumPy 实现 softmax 和 cross entropy。
- 对比不同 learning rate 下 loss 的变化。

验收标准：

- 能解释 logits、probability、label、loss 的关系。
- 能画出 loss 随训练步数下降的曲线。
- 能发现 learning rate 太大导致 loss 发散。

## Week 3: 手写 MLP 与反向传播

学习重点：

- 两层 MLP 的 forward。
- ReLU 的 backward。
- 矩阵乘法的梯度 shape。
- 参数更新。
- train/validation split。

实践项目：

- 只用 NumPy 实现一个两层 MLP 分类器。
- 数据可以使用 sklearn 的 digits、make_moons，或手写二维 toy dataset。
- 保存训练日志：loss、accuracy、learning rate。

本目录已提供对应实现：

- 脚本：`experiments/chapter01/two_layer_mlp_numpy.py`
- Notebook：`notebooks/chapter01/02_two_layer_mlp_backprop.ipynb`

运行：

```bash
cd ~/work/lentica/learning/gpu-llm
python experiments/chapter01/two_layer_mlp_numpy.py
```

验收标准：

- 不依赖 autograd 完成训练。
- 能说明每个参数的 shape 和梯度 shape。
- 能解释 forward、loss、backward、update 的顺序。

## 常见坑

- shape 靠试，不靠推导。
- softmax 没做数值稳定处理。
- 只会调用库，不知道 loss 的输入输出含义。
- 反向传播只背公式，不检查每一步的矩阵维度。

## 下一阶段前置条件

进入 PyTorch 阶段前，应能从零写出一个 NumPy 训练循环，并理解为什么深度学习框架需要 autograd。
