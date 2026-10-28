# Stage 5: Transformer and LLM

## 阶段目标

本阶段目标是理解 decoder-only LLM 的核心结构，并亲手实现一个小型 Transformer 语言模型。结束时你应该能解释 tokenization、attention、causal mask、KV cache 和 next-token prediction。

## 时间安排

默认 4-5 周，每周 8-10 小时。

## Week 1: Tokenization 与语言模型任务

学习重点：

- character、word、subword tokenization。
- token id 与 embedding。
- sequence length。
- next-token prediction。
- train input 与 label 的 shift。

练习：

- 用字符级 tokenizer 处理一小段文本。
- 构造语言模型训练样本：输入序列和下一个 token label。
- 统计 vocabulary、sequence length、batch shape。

验收标准：

- 能解释为什么语言模型训练标签是右移一位。
- 能说明 tokenizer 会影响模型可学习内容。

## Week 2: Attention

学习重点：

- query、key、value。
- scaled dot-product attention。
- attention mask。
- causal mask。
- multi-head attention。

练习：

- 用 PyTorch 手写 single-head causal self-attention。
- 打印 attention score 和 mask 后的矩阵。
- 扩展到 multi-head attention。

验收标准：

- 能说明 causal mask 为什么防止看未来 token。
- 能推导 attention 输入输出 shape。

## Week 3: Decoder-only Transformer

学习重点：

- embedding + position embedding。
- attention block。
- MLP block。
- residual connection。
- layer norm。
- logits head。

实践项目：

- 实现一个小型 character-level decoder-only Transformer。
- 使用小文本数据训练生成模型。
- 支持 temperature sampling。

验收标准：

- 能训练出 loss 下降的小模型。
- 能生成文本样例。
- 能解释每个模块的作用。

## Week 4: Hugging Face Transformers 入门

学习重点：

- AutoTokenizer。
- AutoModelForCausalLM。
- `generate()`。
- `max_new_tokens`、temperature、top_p。
- model checkpoint 与 tokenizer 的关系。

练习：

- 加载一个小型开源 causal LM 做推理。
- 对比 greedy、temperature、top_p 的输出差异。
- 记录显存占用。

验收标准：

- 能正确加载模型和 tokenizer。
- 能解释 generation 参数对输出的影响。

## Week 5: LLM 推理性能与 KV cache

学习重点：

- prefill 与 decode。
- KV cache。
- context length。
- batch inference。
- 量化推理基础。

练习：

- 对比不同 prompt 长度下的推理速度。
- 对比不同 `max_new_tokens` 的耗时。
- 观察显存随 context length 的变化。

验收标准：

- 能解释为什么长上下文更吃显存。
- 能说明 KV cache 的用途。

## 常见坑

- 会用 `generate()`，但不知道 logits 和 sampling 的关系。
- attention shape 混乱。
- 忽略 tokenizer 与模型 checkpoint 必须匹配。
- 把微调和预训练混为一谈。

## 下一阶段前置条件

进入微调阶段前，应能跑通 Hugging Face causal LM 推理，并理解指令微调本质上仍是 next-token prediction。
