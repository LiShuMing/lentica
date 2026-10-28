from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.models.asset_context import AssetContext, asset_context_from_payload


@dataclass(frozen=True)
class ContextVectorTexts:
    caption_text: str
    context_text: str
    ocr_text: str
    lexical_terms: list[str]


def build_context_vector_texts(
    context: AssetContext | dict[str, Any], *, inference_threshold: float = 0.65
) -> ContextVectorTexts:
    parsed = context if isinstance(context, AssetContext) else asset_context_from_payload(context)
    observation = parsed.observations
    caption_text = _join([observation.short_caption, observation.caption])
    sections = [
        _label("Objects", observation.objects),
        _label("Actions", observation.actions),
        _label("Scenes", observation.scenes),
        _label("Relationships", observation.relationships),
        _label("Attributes", observation.visual_attributes),
    ]
    inferred = [
        item.value
        for values in (
            parsed.inferences.possible_events,
            parsed.inferences.possible_locations,
            parsed.inferences.possible_time_periods,
            parsed.inferences.possible_relationships,
        )
        for item in values
        if item.confidence >= inference_threshold and item.value.strip()
    ]
    if inferred:
        sections.append(_label("Possible context", inferred))
    lexical_terms = _unique(
        observation.objects
        + observation.actions
        + observation.scenes
        + observation.logos
        + observation.visible_text
    )
    return ContextVectorTexts(
        caption_text=caption_text,
        context_text="\n".join(section for section in sections if section),
        ocr_text=parsed.normalized_ocr_text.strip(),
        lexical_terms=lexical_terms,
    )


def _label(name: str, values: list[str]) -> str:
    values = _unique(values)
    return f"{name}: {', '.join(values)}" if values else ""


def _join(values: list[str]) -> str:
    return "\n".join(_unique([value.strip() for value in values if value.strip()]))


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))
