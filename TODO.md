# 双路召回本地复现：Prompt 工程指南 — 2026-10-13

> 当前状态（2026-10-31）：整体迭代保持暂停，视频暂缓；0009 产品检索主流程已统一。
> 最新运行/验证状态和未完成事项见 [项目状态](docs/project-status.md)。下文是独立
> 的开源实现与实验计划；发布复核后已移除外部系统映射。

目标：用开源组件实现独立的本地混合检索链路（双路索引构建 → 双路召回 → RRF 融合 → 效果验证），验证候选窗口、按排名融合、最终资源读取与过滤策略的正确性及效果。

## 一、本地实现与接口约束

- Parquet 数据集使用 int64 doc_id 作稳定标识；检索只返回 doc_id + rank，最终 Top-K 才读正文
- sentence-transformers 生成向量，faiss IndexHNSWFlat 提供可选近似检索；参数以实际库接口和实验为准
- tantivy-py 构建本地全文索引，使用 BM25 生成文本候选
- 文本与向量使用两个独立检索函数；BM25 分数与向量距离只作诊断输出，不进融合
- Weighted RRF 融合器：score = w_t/(c+t_rank) + w_v/(c+v_rank)，c 默认 60
- 两路各自先收敛 → 单进程内两路各自完整收敛 Top-K 再融合（进阶可分 shard 模拟）
- text_top_k / vector_top_k 分别限制两路候选窗口
- BM25 统计对照：切多个 tantivy index 模拟 shard，对比局部统计与全局统计
- 等价过滤下推 → 加 category 过滤维度，对比"两路下推"vs"融合后过滤"的召回差异
- ANN Benchmark / Recall@K → BEIR scifact 子集（5K 文档、1K query，本地分钟级跑完）；指标 Recall@10 / nDCG@10 / MRR

## 二、最小技术栈

python 3.10+；pip install tantivy faiss-cpu sentence-transformers ir-datasets pandas pyarrow。
embedding 模型：英文 all-MiniLM-L6-v2 或中文 BAAI/bge-small-zh-v1.5；数据集 BEIR scifact（英文）或自造中文语料（如产品说明书段落 + 拼写变体 query）。

## 三、分阶段 Prompt

每个 prompt 固定五要素：角色与目标、技术栈与版本、接口契约（函数签名先于实现）、可执行验收清单、反幻觉要求。建议逐阶段投喂，每阶段让 agent 先输出"文件树 + 函数签名"方案，确认后再写实现。

### P0 环境与数据基线

```text
你是检索系统工程师。帮我在本地搭建双路召回实验的 P0 阶段：环境与数据。
技术栈：python 3.10+，依赖 tantivy、faiss-cpu、sentence-transformers、ir-datasets、pandas、pyarrow，全部写入 requirements.txt 并 pin 版本。
任务：
1. 下载 BEIR scifact 数据集（ir_datasets），语料写入 data/corpus.parquet，字段：doc_id(int64，从 0 连续编号)、title、text；query 写入 data/queries.parquet（query_id, text），qrels 写入 data/qrels.parquet。
2. 输出 data_report.md：文档数、query 数、有标注 query 数、平均文档词数、top10 高频词。
3. 建工程骨架 hybrid/：config.py（全局参数：top_k、text_top_k、vector_top_k、rrf_constant=60、text_weight=1、vector_weight=1、模型名）、index_build.py（P1 占位）、retrieve.py（P2 占位）、evaluate.py（P3 占位）。
验收（必须真实运行并贴输出）：
- 打印 5 条语料样例（doc_id + 前 100 字符）
- 断言 doc_id 唯一且连续、qrels 中的 doc_id 全部存在于语料
反幻觉：命令执行结果如实贴出；库版本用 pip show 确认后写入报告；失败就报告失败原因，不要编造。
```

### P1 双路索引构建

```text
继续双路召回实验 P1 阶段：对 data/corpus.parquet 构建两条独立索引。全文使用 Tantivy，向量使用 FAISS。
1. 全文索引（index_build.py::build_text_index）：
   - tantivy schema：doc_id 作为 stored 字段，text 字段 TEXT 类型默认 BM25；title 与 text 拼接进同一可检索字段
   - bulk build，记录耗时与吞吐（docs/s）
2. 向量索引（build_vector_index）：
   - sentence-transformers 编码 title+text（batch 推理，记录吞吐），向量存 data/vectors.npy
   - faiss IndexHNSWFlat（L2），M=32、ef_construction=200，写入 index/vector.index；注释解释图连接数与构建搜索范围的作用
3. 注释说明参数含义、适用范围与实际库版本，不引入其他系统的参数映射。
验收：
- 两条单路各跑 3 个样例 query 的 top-10，能按 doc_id 从 parquet 回表打印正文前 100 字（体现"只回表一次"的回表语义）
- 构建耗时写入 report；二次加载索引可行（不重建）
反幻觉：faiss/tantivy 的 API 以实际 import 和运行结果为准，不确定的 API 先查官方文档再写。
```

### P2 双路召回 + RRF 融合（核心）

```text
继续 P2：实现双路召回与 RRF 融合，遵守以下独立接口契约。
接口契约（先确认签名再实现）：
- retrieve_text(query: str, k: int) -> list[tuple[int, int, float]]  # (doc_id, text_rank 从 1 开始, bm25_score 仅诊断)
- retrieve_vector(query_vec: np.ndarray, k: int) -> list[tuple[int, int, float]]  # (doc_id, vector_rank, distance 仅诊断)
- fuse_rrf(text_hits, vector_hits, top_k, rrf_constant=60, text_weight=1.0, vector_weight=1.0) -> list[tuple[int, float, dict]]
语义硬约束（逐条实现并在单测覆盖）：
1. rrf_score = w_t/(c+t_rank) + w_v/(c+v_rank)；rank 从 1 开始；某路未返回该 doc 时该项 contribution 为 0（全外连接语义：单路命中保留单路贡献，双路命中累加）
2. 同一路内一个 doc_id 只允许出现一次（去重后再计 rank）
3. 并列分数按 doc_id 升序 tie-break，保证重跑结果稳定
4. 融合器输入只有 (doc_id, rank)，不接触 bm25_score/distance——"融合不比较原始分数"
5. text_top_k / vector_top_k 独立于最终 top_k，窗口外不进融合器——bounded RRF
提供入口函数 hybrid_search(query: str, config) 组装两路（query 向量实时编码），显式参数为 top_k、text_top_k、vector_top_k、rrf_constant、text_weight、vector_weight。
验收（写 pytest）：
- 对拍测试：用 pandas 独立实现候选去重、排名、按 doc_id 合并与 RRF 参考计算，断言与 fuse_rrf 输出完全一致
- 构造单路命中样例、双路命中样例、并列 tie-break 样例各 1 个
- 同 query 跑 3 遍结果逐位一致
反幻觉：先给文件树和签名，我确认后再写实现。
```

### P3 评估与消融

```text
继续 P3：用 scifact qrels 做效果验证与消融，回答"双路何时优于单路"。
1. evaluate.py：对全部有标注 query 跑各配置，输出 Recall@10、nDCG@10、MRR（nDCG 可用 pytrec_eval 或手写并说明公式）。
2. 配置矩阵（结果写入 eval_report.md，每个配置注明消融变量）：
   - text-only（vector 权重 0）
   - vector-only（text 权重 0）
   - RRF 等权基线（c=60）
   - 权重扫描：text_weight ∈ {0.5, 1, 2}（vector_weight=1）
   - rrf_constant 敏感性：c ∈ {1, 10, 60, 1000}
   - bounded 窗口：text_top_k=vector_top_k ∈ {10, 50, 200} 对比窗口拉满（近似无限列表 RRF）
3. 报告必须回答三个问题：哪些 query 类型文本路赢/向量路赢（抽 3 个典型案例贴 query 与两路 top-5）；窗口从 10→200 指标变化多少（bounded RRF 的代价）；c 的敏感性是否如理论预期（c 只调节平滑，不改变排序趋势）。
验收：固定 seed 全流程可复现；报告数字与重跑结果一致。
反幻觉：所有指标来自真实运行输出，禁止估算或引用文献数字；跑不通的配置如实标注。
```

### P4 进阶实验（可选）

```text
在现有代码上做三个进阶实验，每个先写 5 行实验设计再实现：
1. Local vs Global BM25：把语料随机切 4 片建 4 个独立 tantivy index 模拟 shard。对比两种打分：每 shard 用自身统计（Local BM25）合并 Top-K vs 两轮法（第一轮汇总 4 片 N/df/avgdl，第二轮带全局统计重打分）。用 P3 的 qrels 评估两者 Recall@10 差异，验证"跨 shard 统计不一致会扰动 BM25 排序，而 RRF 用 rank 天然免疫部分此类问题"。
2. 过滤下推语义：给语料加 category 字段。对比 A（两路候选生成前各自过滤——tantivy filter + faiss IDSelector）与 B（融合后过滤）在"过滤后目标类目文档较少"场景下的 Recall 差异，测量各过滤位置对召回率与候选数量的影响，不预设实验结论。
3. Weighted RRF 调权：对 P3 中向量路明显占优的 query 子集单独调 text_weight，观察是否存在一致的子集级最优权重，讨论全局单权重与 per-query 权重的取舍。
验收：每个实验产出结论小节 + 支撑数据；实验 1 需展示同一 query 在 Local/Global 下的 top-5 排序差异实例。
```

## 四、执行技巧

- 逐阶段投喂：P0 跑通验收再进 P1；agent 输出的方案（文件树 + 签名）先人工确认，避免一次性生成跑偏。
- 反幻觉三件套：pin 版本、要求"先查文档再写不确定的 API"、强制"贴真实运行输出，失败如实说失败"。
- 对照学习：fuse_rrf 注释 rank_constant=60 的平滑作用；参考实现只用于正确性对拍，不替代实际检索实现。
- 验证效果的三层口径：功能正确性（P2 对拍 + 单测）、检索质量（P3 qrels 指标）、设计决策验证（P4 消融实验）。
- 中文场景替换：数据集换自造语料（如 100 篇产品说明书段落），query 造两类——精确词面（错误码/型号，文本路应赢）与同义改述（向量路应赢），即可直接观察双路互补性。
