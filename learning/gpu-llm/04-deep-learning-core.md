# Stage 4: Deep Learning Core

## 阶段目标

本阶段目标是掌握训练神经网络时最常见的问题：欠拟合、过拟合、学习率不合适、数据问题、指标不一致。结束时你应该能独立训练一个分类模型，并根据曲线诊断问题。

## 时间安排

默认 4-5 周，每周 8-10 小时。

## Week 1: MLP、初始化与优化

学习重点：

- MLP 表达能力。
- 激活函数：ReLU、GELU、sigmoid。
- 参数初始化。
- SGD、Momentum、AdamW。
- learning rate 的影响。

练习：

- 在同一任务上对比 SGD、AdamW。
- 对比不同 learning rate 的 loss 曲线。
- 观察初始化不合理时的训练问题。

验收标准：

- 能通过 loss 曲线识别 learning rate 太大或太小。
- 能解释 AdamW 和 weight decay 的基本作用。

## Week 2: CNN 与图像分类

学习重点：

- convolution、channel、kernel、stride、padding。
- pooling。
- batch norm。
- data augmentation。
- top-1 accuracy。

练习：

- 训练一个 CIFAR-10 或类似小图像分类模型。
- 加入简单数据增强。
- 对比无增强和有增强的验证集表现。

验收标准：

- 能解释卷积层输出 shape。
- 能说明 data augmentation 如何影响过拟合。

## Week 3: 正则化与训练诊断

学习重点：

- train loss 与 validation loss。
- dropout。
- weight decay。
- early stopping。
- class imbalance。
- confusion matrix。

练习：

- 人为减少训练数据，观察过拟合。
- 加入 dropout 或 weight decay 后对比。
- 输出 confusion matrix，分析错误类别。

验收标准：

- 能区分欠拟合和过拟合。
- 能根据指标提出合理下一步实验。

## Week 4: 文本分类入门

学习重点：

- 文本 tokenization 基础。
- vocabulary。
- embedding。
- sequence length。
- padding 和 mask。

练习：

- 用简单 embedding + pooling 模型做文本分类。
- 对比不同 sequence length 和 batch size。

验收标准：

- 能解释 token、id、embedding 的关系。
- 能处理 batch 内不同长度文本。

## Week 5: 阶段项目

实践项目：

- 任选图像分类或文本分类，完成一个端到端项目。
- 必须包含：
  - 数据加载。
  - 模型定义。
  - 训练和验证。
  - checkpoint。
  - 指标曲线。
  - 错误样本分析。

验收标准：

- 项目可以一条命令复现训练。
- README 写清楚实验结果和失败尝试。
- 能说明下一步如何提升模型。

## 常见坑

- 只看训练集指标。
- 模型越复杂越好，没有先做 baseline。
- 不固定随机种子，结果无法复现。
- 没有保留失败实验记录。

## 下一阶段前置条件

进入 Transformer 阶段前，应能完成一个标准深度学习项目，并理解 embedding、sequence、mask 等文本模型基础概念。
