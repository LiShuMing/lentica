# Context Plane Proposal

> Historical proposal, not implemented functionality. Lentica currently provides
> local photo/document retrieval. This directory preserves the PRD, UI spec, and
> architecture for a future project-level Context Pack workflow. Follow the root
> README and configuration docs for running the released implementation.

版本：v0.1 / 2026-10-25。状态：可开发规格，尚非已实现功能。

给 WSL Codex：先阅读本文件，再按下列顺序阅读规格，在现有项目上增量开发。先交付四周 P0；第五至六周扩 Notion。不要把整套历史照片路线、云 SaaS 和所有连接器同时纳入首发。

## 文档地图

1. [PRD.md](PRD.md)：用户、范围、逐项需求与验收。
2. [UI_SPEC.md](UI_SPEC.md)：页面、字段、交互、状态与响应式设计。
4. [ARCHITECTURE.md](ARCHITECTURE.md)：现有项目适配、数据模型、湖与索引、队列、授权、删除。

## 产品约定

一句话：把个人项目的资料、行为与决策，整理成有来源、可更新、可导出的 Context Pack，供不同 Agent 接续任务。

首个用例：GitHub 活动 + 博客仓 Markdown + Notion 技术笔记，生成“近 90 天数据库相关、用于写稿/答疑”的项目 Pack。P0 先打通 GitHub 和本地博客；Notion 为 P1。不能把 GitHub public Events API 当成完整 90 天历史。

首发三工具：`search_context` / `get_context_pack` / `list_sources`。Agent 只读；用户在网页上审核候选决策、更正、删除、导出与授权。

## 与现有文件的优先级

本目录定义新增 Context Plane MVP 的范围与契约。根目录 DESIGN_GUIDE.md 仍是既有照片/PDF/本地检索功能的参考；实现新模块时本目录更具体的规则优先，不能因此覆盖老功能或用户改动。现有 local_only 默认、旧 API 与数据库必须保留兼容。

当前 README/config 显示 local_only，不调用外部模型。此 MVP 默认延续；允许后续用户在设置中显式启用 remote 模式，并只从 WSL `~/.env` 加载配置。文档制作不代表授权现在启用或调用外部模型、账户、存储。

运行前检查现有工作树；2026-10-25 已有大量未提交实现，不 reset、不重建数据库、不运行全库重索引。采用迁移和小样本 fixture。默认不 commit、不 push、不发布。

## 四周完成的判据

- 两源导入、增量游标、幂等与队列状态可见。
- 项目级 Pack 在网页生成、证据可打开、Markdown/JSON 可导出。
- SQLite 控制面 + Parquet 快照 + DuckDB 查询；本地可运行，至少一个自带 S3 或 OSS 桶集成验证。
- 两个真实 Agent 客户端通过授权工具读同一项目，不越权；撤销后下次请求拒绝。
- 十条黄金任务、重启续跑、删除传播、版本更新与 token 预算验收通过。
- 截图、复现步骤、实际测量结果记录在实施日志；没有实测不得写“已完成”。

## 给 Codex 的启动提示

```text
在 ~/work/personal-knowledge-lab 增量实现 docs/mvp-context-plane/README.md 中的 P0。
先检查现有代码和未提交改动，再阅读 PRD、UI、架构、API、schema、实施计划和评测。
保留现有照片功能和 local_only 默认。按依赖逐个完成 M0-M4，每次通过验收再推进。
先使用合成 fixture，连接真实账户或桶时只使用已配置且项目明确选择的来源。
实现三项只读 MCP 工具与项目授权。不得把任意本地路径、凭证或未授权来源暴露给 Agent。
不 commit/push，不替换现有数据；最终运行所需服务并报告 URL、测试结果和剩余事项。
```
