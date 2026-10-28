from __future__ import annotations

from typing import Any


class LocalAnswerer:
    async def answer(self, question: str, evidence: list[dict[str, Any]]) -> str:
        return _extractive_answer(question, evidence)


def _extractive_answer(question: str, evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "本地索引中没有找到足够证据回答这个问题。请先在 Library 页面运行增量扫描。"

    confident = [item for item in evidence if item["score"] >= 0.08] or evidence[:3]
    lines = [
        "基于当前本地索引，最相关的证据如下。由于当前启用的是 extractive_local 模式，我只整理检索到的片段，不补充来源外的信息。",
        "",
    ]
    for idx, item in enumerate(confident[:5], start=1):
        excerpt = _compact(item["text"], limit=280)
        lines.append(f"[{idx}] {item['windows_path']} {item['page_or_segment']}")
        lines.append(excerpt)
        lines.append("")
    if confident[0]["score"] < 0.12:
        lines.append("不确定性：命中分数偏低，以上可能只是相近来源。")
    else:
        lines.append("结论：请以这些引用片段为依据继续判断；完整上下文可从来源文件打开查看。")
    del question
    return "\n".join(lines).strip()


def _compact(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."
