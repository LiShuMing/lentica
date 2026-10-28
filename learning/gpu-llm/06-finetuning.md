# Stage 6: LLM Fine-Tuning

## 阶段目标

本阶段目标是完成一个可复现的 LoRA 或 QLoRA 微调项目。结束时你应该能准备数据、选择模型、配置训练、控制显存、保存 adapter，并评估微调前后的效果。

## 时间安排

默认 4-6 周，每周 8-10 小时。

## Week 1: 微调任务和数据集

学习重点：

- pretraining、supervised fine-tuning、preference tuning 的区别。
- instruction、input、output 格式。
- prompt template。
- train/validation split。
- 数据质量比数量更重要。

练习：

- 准备 500-5000 条 instruction 数据。
- 统一字段：instruction、input、output。
- 写脚本检查空值、过长样本、重复样本。

验收标准：

- 数据可以被稳定加载。
- 能随机抽样检查 prompt 拼接结果。
- 能说明哪些样本会伤害训练质量。

## Week 2: Hugging Face Datasets 与 Transformers 训练

学习重点：

- `datasets` 加载 json/jsonl。
- tokenize function。
- data collator。
- causal LM labels。
- Trainer 或自定义训练循环。

练习：

- 用小模型和小数据跑通一次完整 fine-tuning smoke test。
- 先训练几十 step，确认 loss 正常下降。

验收标准：

- 能定位 tokenize、padding、label 构造错误。
- 能区分训练脚本错误和数据质量问题。

## Week 3: LoRA

学习重点：

- LoRA 的低秩适配思想。
- target modules。
- rank、alpha、dropout。
- base model 与 adapter。
- merge 与不 merge 的区别。

实践项目：

- 使用 PEFT 对一个 1B-8B 级 causal LM 做 LoRA 微调。
- 保存 adapter。
- 加载 base model + adapter 推理。

验收标准：

- 能解释只训练 adapter 为什么省显存。
- 能列出实际训练参数量和总参数量。
- 能对比 base model 与 LoRA model 输出。

## Week 4: QLoRA 与 16GB 显存约束

学习重点：

- 4-bit quantization。
- NF4、double quantization 的基本概念。
- gradient accumulation。
- gradient checkpointing。
- batch size、sequence length 与显存。

练习：

- 用 QLoRA 跑通一个更大的模型或更长 sequence。
- 记录不同 batch size、sequence length 的显存占用。

验收标准：

- 能在 16GB 显存内完成稳定训练。
- 遇到 OOM 时能调整 batch、accumulation、sequence length、checkpointing。

## Week 5: 评估

学习重点：

- 固定评估 prompt。
- exact match、ROUGE、人工评分。
- 过拟合检查。
- hallucination 与格式错误。
- 训练集泄漏。

练习：

- 设计 20-50 条固定评估问题。
- 比较 base model、fine-tuned model 输出。
- 给每条输出打标签：正确、部分正确、格式错、幻觉。

验收标准：

- 不只凭主观感觉判断微调成功。
- 能指出微调改善和退化的样例。

## Week 6: 最终项目整理

最终项目必须包含：

- 数据集说明。
- 训练配置。
- 训练命令。
- 显存和耗时。
- loss 曲线。
- adapter 保存路径。
- 推理脚本。
- 微调前后对比。
- 已知问题和下一步计划。

验收标准：

- 从空环境按 README 可以复现。
- 能向别人讲清楚为什么这样选模型、数据、LoRA 参数和评估方法。

## 常见坑

- 数据格式混乱，训练实际学不到指令。
- 没有 validation set。
- 只看 loss，不看生成质量。
- 把 base model 覆盖掉，不清楚 adapter 如何加载。
- 盲目追大模型，忽略数据质量和评估。

## 完成路线后的下一步

- 学习 vLLM、llama.cpp 或 TensorRT-LLM 做推理部署。
- 学习 DPO、ORPO 等偏好优化方法。
- 学习多 GPU、DeepSpeed、FSDP。
- 学习 CUDA C++ 或 Triton，进入算子优化路线。
