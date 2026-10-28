from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import File, Image, PhotoVector
from backend.app.ingestion.faces import matching_person_file_ids, people_for_file
from backend.app.models.local_clip import Embedder, HashEmbedder
from backend.app.retrieval.fusion import RetrievalHit, weighted_rrf
from backend.app.retrieval.query_plan import build_retrieval_plan


YEAR_RE = re.compile(r"(20\d{2}|19\d{2})")
TEXT_LOGICAL_NAMESPACES = [
    "text",
    "caption",
    "dense_caption",
    "ocr",
    "tags",
    "objects",
    "people",
    "activities",
    "location",
    "time",
    "event",
    "visual_attributes",
    "search_phrases",
    "metadata",
]


@dataclass(frozen=True)
class PhotoHit:
    file_id: int
    score: float
    confidence: float
    image_score: float
    text_score: float
    metadata_score: float
    namespace_scores: dict[str, float]
    matched_terms: list[str]
    match_reason: str
    path: str
    windows_path: str
    file_name: str
    thumb_path: str | None
    caption: str | None
    dense_caption: str | None
    ocr_text: str | None
    context: dict[str, object]
    tags: list[str]
    exif_taken_at: str | None


def normalize(vector: Iterable[float]) -> list[float]:
    values = [float(v) for v in vector]
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return values
    return [v / norm for v in values]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def upsert_photo_vector(db: Session, file_id: int, namespace: str, vector: list[float]) -> None:
    normalized = normalize(vector)
    existing = db.scalar(
        select(PhotoVector).where(PhotoVector.file_id == file_id, PhotoVector.namespace == namespace)
    )
    payload = json.dumps(normalized)
    if existing:
        existing.dims = len(normalized)
        existing.vector_json = payload
    else:
        db.add(PhotoVector(file_id=file_id, namespace=namespace, dims=len(normalized), vector_json=payload))


class PhotoSearch:
    def __init__(self, db: Session, visual_embedder: Embedder, text_embedder: Embedder | None = None):
        self.db = db
        self.visual_embedder = visual_embedder
        self.text_embedder = text_embedder or visual_embedder

    def search(self, query: str, limit: int = 12) -> tuple[str, list[PhotoHit]]:
        answer, hits, _debug = self.search_with_debug(query, limit=limit)
        return answer, hits

    def search_with_debug(self, query: str, limit: int = 12) -> tuple[str, list[PhotoHit], dict[str, object]]:
        plan = build_retrieval_plan(self.db, query, expand=expand_query)
        visual_query_vector = self.visual_embedder.embed_text(plan.visual_query)
        text_query_vector = self.text_embedder.embed_text(plan.semantic_query)
        hits, retrieval_debug = self._hybrid_search(
            query, plan.semantic_query, visual_query_vector, text_query_vector, limit=max(limit * 4, limit)
        )
        if plan.person_file_ids:
            scoped_hits = [hit for hit in hits if hit.file_id in plan.person_file_ids]
            if scoped_hits:
                hits = scoped_hits
        if plan.years:
            hits = [hit for hit in hits if hit.exif_taken_at and hit.exif_taken_at[:4] in plan.years]
        hits = hits[:limit]
        retrieval_debug["plan"] = plan.debug_payload()
        retrieval_debug["final_result_count"] = len(hits)
        return self._summarize(query, hits), hits, retrieval_debug

    def _hybrid_search(
        self, query: str, expanded_query: str, visual_query_vector: list[float], text_query_vector: list[float], limit: int
    ) -> tuple[list[PhotoHit], dict[str, object]]:
        visual_q = normalize(visual_query_vector)
        text_q = normalize(text_query_vector)
        namespaces = _physical_namespaces(_namespace_weights(query))
        smoke_mode = isinstance(self.visual_embedder, HashEmbedder)
        if smoke_mode:
            namespaces = [f"hash:{name}" for name in _namespace_weights(query)]
        person_file_ids, person_names = matching_person_file_ids(self.db, expanded_query)
        rows = self.db.execute(
            select(PhotoVector, File, Image)
            .join(File, File.id == PhotoVector.file_id)
            .join(Image, Image.file_id == File.id)
            .where(PhotoVector.namespace.in_(namespaces), File.ingestion_status == "indexed")
        ).all()
        candidates: dict[int, dict[str, object]] = {}
        for vector, file, image in rows:
            item = candidates.setdefault(
                file.id,
                {
                    "file": file,
                    "image": image,
                    "namespace_scores": {},
                    "physical_scores": {},
                },
            )
            query_vector = text_q if vector.namespace.startswith("bge:") else visual_q
            score = cosine(query_vector, json.loads(vector.vector_json))
            physical_scores = item["physical_scores"]
            physical_name = _logical_namespace(vector.namespace) if smoke_mode else vector.namespace
            physical_scores[physical_name] = max(float(physical_scores.get(physical_name, 0.0)), score)
            logical_namespace = _logical_namespace(vector.namespace)
            namespace_scores = item["namespace_scores"]
            namespace_scores[logical_namespace] = max(float(namespace_scores.get(logical_namespace, 0.0)), score)

        metadata_by_file: dict[int, tuple[float, list[str], list[str]]] = {}
        for file_id, item in candidates.items():
            file = item["file"]
            image = item["image"]
            tags = json.loads(image.tags or "[]")
            context = _parse_context(image.context_json)
            people_text = " ".join(_person_terms(people_for_file(self.db, file_id)))
            metadata_by_file[file_id] = _metadata_score(query, file, image, tags, context, people_text)

        result_lists = _independent_retrieval_lists(candidates, metadata_by_file, limit=max(limit, 50))
        fused = weighted_rrf(result_lists, weights=_retriever_weights(query))
        fused_scores = _normalize_rrf_scores(fused, _retriever_weights(query))
        fused_sources = {hit.asset_id: hit.sources for hit in fused}
        hits: list[PhotoHit] = []
        for file_id, item in candidates.items():
            if file_id not in fused_scores:
                continue
            file = item["file"]
            image = item["image"]
            tags = json.loads(image.tags or "[]")
            context = _parse_context(image.context_json)
            namespace_scores = {key: round(float(value), 6) for key, value in item["namespace_scores"].items()}
            metadata_score, metadata_hits, matched_terms = metadata_by_file[file_id]
            if file_id in person_file_ids:
                metadata_score = max(metadata_score, 1.0)
                metadata_hits.append("person_cluster")
                matched_terms.extend(person_names)
            image_score = float(namespace_scores.get("image", 0.0))
            text_score = max(
                float(namespace_scores.get(name, 0.0))
                for name in TEXT_LOGICAL_NAMESPACES
            )
            score = fused_scores[file_id]
            confidence = _confidence(score, image_score, text_score, metadata_score)
            hits.append(
                PhotoHit(
                    file_id=file_id,
                    score=round(score, 6),
                    confidence=confidence,
                    image_score=round(image_score, 6),
                    text_score=round(text_score, 6),
                    metadata_score=round(metadata_score, 6),
                    namespace_scores=namespace_scores,
                    matched_terms=matched_terms,
                    match_reason=_match_reason(
                        image_score,
                        text_score,
                        metadata_hits,
                        namespace_scores,
                        fused_sources.get(file_id, []),
                    ),
                    path=file.canonical_path,
                    windows_path=file.windows_path,
                    file_name=Path(file.canonical_path).name,
                    thumb_path=image.thumb_path,
                    caption=image.caption,
                    dense_caption=str(context.get("dense_caption") or image.caption or ""),
                    ocr_text=image.ocr_text,
                    context=context,
                    tags=tags,
                    exif_taken_at=image.exif_time.isoformat() if image.exif_time else None,
                )
            )
        hits.sort(key=lambda hit: (hit.score, hit.confidence), reverse=True)
        return hits[:limit], {
            "candidate_count": len(candidates),
            "retriever_counts": {source: len(items) for source, items in result_lists.items()},
            "rrf_candidate_count": len(fused),
            "retriever_weights": _retriever_weights(query),
        }

    def _summarize(self, query: str, hits: list[PhotoHit]) -> str:
        if not hits:
            return f"没有找到与「{query}」明显相关的图片。可以换一个描述，或先运行照片索引 worker。"
        lines = [f"找到 {len(hits)} 张候选图片，按本地混合相关度排序。"]
        for idx, hit in enumerate(hits[:5], start=1):
            when = f"{hit.exif_taken_at[:10]}，" if hit.exif_taken_at else ""
            caption = hit.caption or hit.file_name
            lines.append(f"{idx}. {when}confidence {hit.confidence:.2f}，{caption[:120]}")
        return "\n".join(lines)


def _metadata_score(
    query: str, file: File, image: Image, tags: list[str], context: dict[str, object], people_text: str = ""
) -> tuple[float, list[str], list[str]]:
    query_tokens = _tokens(query)
    haystacks = {
        "caption": image.caption or "",
        "dense_caption": str(context.get("dense_caption") or ""),
        "tags": " ".join(str(tag) for tag in tags),
        "ocr": image.ocr_text or "",
        "objects": " ".join(_as_text_list(context.get("objects"))),
        "people": " ".join([people_text, " ".join(_as_text_list(context.get("people")))]),
        "activities": " ".join(_as_text_list(context.get("activities"))),
        "search_phrases": " ".join(_as_text_list(context.get("search_phrases"))),
        "visual_attributes": " ".join(_as_text_list(context.get("visual_attributes"))),
        "file": Path(file.canonical_path).name,
        "date": image.exif_time.isoformat() if image.exif_time else "",
        "location": " ".join(
            [
                image.location or "",
                str(context.get("location") or ""),
                " ".join(_as_text_list(context.get("location_clues"))),
                _metadata_candidates_text(context, {"location"}),
            ]
        ),
        "time": " ".join(
            [
                str(context.get("time") or ""),
                " ".join(_as_text_list(context.get("time_clues"))),
                _metadata_candidates_text(context, {"time", "date", "year"}),
            ]
        ),
        "event": " ".join([str(context.get("event") or ""), " ".join(_as_text_list(context.get("events")))]),
        "metadata": " ".join([str(context.get("metadata_text") or ""), _metadata_candidates_text(context)]),
    }
    weights = {
        "caption": 0.18,
        "dense_caption": 0.14,
        "tags": 0.16,
        "ocr": 0.14,
        "objects": 0.10,
        "people": 0.08,
        "activities": 0.08,
        "search_phrases": 0.14,
        "visual_attributes": 0.06,
        "file": 0.07,
        "date": 0.08,
        "location": 0.08,
        "time": 0.08,
        "event": 0.07,
        "metadata": 0.10,
    }
    score = 0.0
    matched: list[str] = []
    matched_terms: list[str] = []
    for name, text in haystacks.items():
        text_tokens = _tokens(text)
        if not text_tokens:
            continue
        overlap = query_tokens & text_tokens
        if overlap:
            score += weights[name] * min(1.0, len(overlap) / max(1, len(query_tokens)))
            matched.append(name)
            matched_terms.extend(sorted(overlap))
    return min(1.0, score), matched, sorted(set(matched_terms))[:16]


def _tokens(text: str) -> set[str]:
    normalized = text.lower()
    ascii_tokens = set(re.findall(r"[a-z0-9]{2,}", normalized))
    cjk_tokens = set(re.findall(r"[\u4e00-\u9fff]{1,4}", normalized))
    return ascii_tokens | cjk_tokens


def _person_terms(people: list[dict[str, object]]) -> list[str]:
    terms: list[str] = []
    for person in people:
        for key in ["name", "label"]:
            text = str(person.get(key) or "").strip()
            if text:
                terms.append(text)
        aliases = person.get("aliases")
        if isinstance(aliases, list):
            terms.extend(str(alias).strip() for alias in aliases if str(alias).strip())
    return terms


def _metadata_candidates_text(context: dict[str, object], kinds: set[str] | None = None) -> str:
    candidates = context.get("metadata_candidates")
    if not isinstance(candidates, list):
        return ""
    values: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        kind = str(candidate.get("kind") or "")
        if kinds and kind not in kinds:
            continue
        values.append(
            " ".join(
                str(candidate.get(key) or "").strip()
                for key in ["kind", "value", "source", "confidence"]
                if str(candidate.get(key) or "").strip()
            )
        )
    return " ".join(values)


def _confidence(score: float, image_score: float, text_score: float, metadata_score: float) -> float:
    agreement = 0.0
    if image_score > 0.20 and text_score > 0.20:
        agreement += 0.10
    if metadata_score > 0:
        agreement += min(0.12, metadata_score * 0.12)
    return round(max(0.0, min(1.0, ((score + 1.0) / 2.0) + agreement)), 3)


def _match_reason(
    image_score: float,
    text_score: float,
    metadata_hits: list[str],
    namespace_scores: dict[str, float],
    retrieval_sources: list[str] | None = None,
) -> str:
    parts: list[str] = []
    top_namespaces = sorted(namespace_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    if top_namespaces:
        parts.append("vector:" + ",".join(name for name, _ in top_namespaces))
    elif image_score >= text_score:
        parts.append("visual")
    else:
        parts.append("caption")
    if metadata_hits:
        parts.append("metadata:" + ",".join(metadata_hits[:3]))
    if retrieval_sources:
        parts.append("rrf:" + ",".join(retrieval_sources))
    return " + ".join(parts)


def _independent_retrieval_lists(
    candidates: dict[int, dict[str, object]],
    metadata_by_file: dict[int, tuple[float, list[str], list[str]]],
    *,
    limit: int,
) -> dict[str, list[RetrievalHit]]:
    """Build rankings per embedding space; scores stay within their own retriever."""
    source_scores: dict[str, list[tuple[int, float]]] = {
        "visual": [],
        "caption": [],
        "context": [],
        "ocr_dense": [],
        "metadata": [],
    }
    for file_id, item in candidates.items():
        physical = item["physical_scores"]
        visual = _best_physical_score(physical, lambda name: name in {"image", "clip:image"})
        caption = _best_physical_score(
            physical,
            lambda name: name in {"bge:caption", "bge:dense_caption", "caption", "dense_caption"},
        )
        ocr = _best_physical_score(physical, lambda name: name in {"bge:ocr", "ocr", "clip:ocr"})
        context = _best_physical_score(
            physical,
            lambda name: name
            in {
                "bge:text",
                "bge:tags",
                "bge:objects",
                "bge:people",
                "bge:activities",
                "bge:visual_attributes",
                "bge:search_phrases",
                "text",
                "tags",
                "objects",
                "people",
                "activities",
                "visual_attributes",
                "search_phrases",
            },
        )
        if visual is not None:
            source_scores["visual"].append((file_id, visual))
        if caption is not None:
            source_scores["caption"].append((file_id, caption))
        if context is not None:
            source_scores["context"].append((file_id, context))
        if ocr is not None:
            source_scores["ocr_dense"].append((file_id, ocr))
        metadata_score = metadata_by_file[file_id][0]
        if metadata_score > 0:
            source_scores["metadata"].append((file_id, metadata_score))

    rankings: dict[str, list[RetrievalHit]] = {}
    for source, scores in source_scores.items():
        rankings[source] = [
            RetrievalHit(asset_id=file_id, rank=rank, score=score, source=source)
            for rank, (file_id, score) in enumerate(
                sorted(scores, key=lambda item: item[1], reverse=True)[:limit], start=1
            )
        ]
    return rankings


def _best_physical_score(physical: object, predicate) -> float | None:
    if not isinstance(physical, dict):
        return None
    scores = [float(score) for namespace, score in physical.items() if predicate(str(namespace))]
    return max(scores) if scores else None


def _retriever_weights(query: str) -> dict[str, float]:
    intent = classify_query_intent(query)
    weights = {"visual": 1.2, "caption": 1.0, "context": 0.9, "ocr_dense": 0.35, "metadata": 0.7}
    if "ocr" in intent:
        weights.update({"visual": 0.4, "caption": 0.6, "context": 0.7, "ocr_dense": 1.2, "metadata": 1.1})
    if "people" in intent or "time" in intent or "location" in intent:
        weights["metadata"] = 1.4
    if "event" in intent:
        weights["context"] = 1.2
    return weights


def _normalize_rrf_scores(fused, weights: dict[str, float], k: int = 60) -> dict[int, float]:
    maximum = sum(max(0.0, value) for value in weights.values()) / (k + 1)
    if maximum <= 0:
        return {hit.asset_id: 0.0 for hit in fused}
    return {hit.asset_id: round(min(1.0, hit.score / maximum), 6) for hit in fused}


def _namespace_weights(query: str) -> dict[str, float]:
    intent = classify_query_intent(query)
    weights = {
        "image": 0.38,
        "text": 0.10,
        "caption": 0.12,
        "dense_caption": 0.15,
        "ocr": 0.06,
        "tags": 0.08,
        "objects": 0.04,
        "people": 0.03,
        "activities": 0.03,
        "location": 0.01,
        "time": 0.01,
        "event": 0.01,
        "visual_attributes": 0.01,
        "search_phrases": 0.05,
        "metadata": 0.01,
    }
    if "ocr" in intent:
        weights.update({"image": 0.18, "ocr": 0.24, "text": 0.12, "dense_caption": 0.12, "metadata": 0.08})
    if "time" in intent:
        weights["time"] += 0.16
        weights["metadata"] += 0.12
        weights["dense_caption"] += 0.05
        weights["image"] -= 0.08
    if "location" in intent:
        weights["location"] += 0.16
        weights["metadata"] += 0.10
        weights["tags"] += 0.04
        weights["image"] -= 0.06
    if "people" in intent:
        weights["people"] += 0.08
        weights["activities"] += 0.04
        weights["image"] -= 0.05
    if "event" in intent:
        weights["event"] += 0.12
        weights["activities"] += 0.06
        weights["search_phrases"] += 0.05
    if "object" in intent:
        weights["objects"] += 0.10
        weights["tags"] += 0.04
    total = sum(max(0.0, value) for value in weights.values())
    return {key: max(0.0, value) / total for key, value in weights.items()}


def classify_query_intent(query: str) -> set[str]:
    lowered = query.lower()
    intent: set[str] = set()
    if re.search(r"\d{2,}|编号|文字|写着|截图|票|证|单|码|ocr|发票|收据|证件|文档", lowered):
        intent.add("ocr")
    if YEAR_RE.search(query) or re.search(r"昨天|今天|去年|今年|月份|春节|生日|假期|周末|晚上|夜晚|白天|早上|下午|夏天|冬天", query):
        intent.add("time")
    if re.search(r"哪里|地点|城市|上海|北京|成都|杭州|海边|山|公园|餐厅|酒店|学校|公司|机场|车站|家里", query):
        intent.add("location")
    if re.search(r"人|合影|朋友|家人|孩子|同事|自拍|聚会|父母|妈妈|爸爸|同学", query):
        intent.add("people")
    if re.search(r"聚餐|旅行|婚礼|生日|会议|工作|学习|徒步|露营|运动|吃饭|火锅", query):
        intent.add("event")
    if re.search(r"猫|狗|车|电脑|手机|书|花|食物|饭菜|衣服|包|杯子|屏幕|白板", query):
        intent.add("object")
    if not intent:
        intent.add("scene")
    return intent


QUERY_EXPANSIONS = {
    "聚餐": "餐厅 饭菜 合影 朋友 聚会 吃饭 火锅",
    "吃饭": "餐厅 饭菜 聚餐 餐桌",
    "旅行": "景点 机场 酒店 山 海边 城市 拍照 出游",
    "证件": "身份证 护照 证书 文字 编号 卡片",
    "发票": "票据 收据 金额 编号 日期 抬头",
    "工作": "电脑 屏幕 文档 白板 会议 办公室",
    "学习": "书 笔记 电脑 教室 文档 白板",
    "海边": "沙滩 大海 海岸 日落 旅行",
    "合影": "多人 朋友 家人 同事 聚会 自拍",
    "截图": "屏幕 文字 应用 页面 聊天 票据",
}


def expand_query(query: str) -> str:
    additions: list[str] = []
    for key, value in QUERY_EXPANSIONS.items():
        if key in query:
            additions.append(value)
    return " ".join([query, *additions])


def _physical_namespaces(weights: dict[str, float]) -> list[str]:
    namespaces: set[str] = {"clip:image", "image"}
    for logical in weights:
        if logical == "image":
            continue
        namespaces.add(f"bge:{logical}")
        namespaces.add(f"clip:{logical}")
        namespaces.add(logical)
    return sorted(namespaces)


def _logical_namespace(namespace: str) -> str:
    if ":" in namespace:
        return namespace.split(":", 1)[1]
    return namespace


def _apply_negative_clues(score: float, query: str, context: dict[str, object]) -> float:
    negatives = " ".join(_as_text_list(context.get("negative_clues")))
    if not negatives:
        return score
    if _tokens(query) & _tokens(negatives):
        return score * 0.82
    return score


def _parse_context(value: str | None) -> dict[str, object]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _as_text_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []
