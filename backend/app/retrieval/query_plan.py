from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date

from sqlalchemy.orm import Session

from backend.app.ingestion.faces import matching_person_file_ids


YEAR_RE = re.compile(r"(?:19|20)\d{2}")
OCR_RE = re.compile(r"写着|包含|文字|代码|报错|错误|函数|堆栈|截图|白板|编号|ocr", re.IGNORECASE)
DOCUMENT_RE = re.compile(r"文档|pdf|页面|论文|表格|图表|公式|财报", re.IGNORECASE)
LOCATION_RE = re.compile(r"上海|北京|成都|杭州|广州|深圳|苏州|南京|西湖|机场|车站|地铁|餐厅|酒店|海边|公园")
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]{2,}|\b\d{3,}\b")


@dataclass(frozen=True)
class RetrievalPlan:
    original_query: str
    semantic_query: str
    visual_query: str
    lexical_terms: list[str] = field(default_factory=list)
    person_file_ids: set[int] = field(default_factory=set)
    person_names: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    years: set[str] = field(default_factory=set)
    asset_types: list[str] = field(default_factory=lambda: ["photo"])
    use_visual: bool = True
    use_caption_dense: bool = True
    use_context_dense: bool = True
    use_ocr_dense: bool = False
    use_ocr_lexical: bool = False
    use_document_retrieval: bool = False

    def debug_payload(self) -> dict[str, object]:
        result = asdict(self)
        result["person_file_ids"] = sorted(self.person_file_ids)
        result["years"] = sorted(self.years)
        return result


def build_retrieval_plan(db: Session, query: str, *, expand) -> RetrievalPlan:
    normalized = query.strip()
    expanded = expand(normalized)
    person_file_ids, person_names = matching_person_file_ids(db, expanded)
    ocr_intent = bool(OCR_RE.search(normalized))
    document_intent = bool(DOCUMENT_RE.search(normalized))
    locations = list(dict.fromkeys(LOCATION_RE.findall(normalized)))
    years = set(YEAR_RE.findall(normalized))
    if "去年" in normalized:
        years.add(str(date.today().year - 1))
    if "今年" in normalized:
        years.add(str(date.today().year))
    return RetrievalPlan(
        original_query=normalized,
        semantic_query=expanded,
        visual_query=expanded,
        lexical_terms=list(dict.fromkeys(IDENTIFIER_RE.findall(normalized)))[:16],
        person_file_ids=person_file_ids,
        person_names=person_names,
        locations=locations,
        years=years,
        asset_types=["document"] if document_intent else ["photo", "screenshot"] if ocr_intent else ["photo"],
        use_visual=not document_intent,
        use_caption_dense=not document_intent,
        use_context_dense=not document_intent,
        use_ocr_dense=ocr_intent,
        use_ocr_lexical=ocr_intent,
        use_document_retrieval=document_intent,
    )
