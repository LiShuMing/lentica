from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


PHOTO_CONTEXT_V2 = "photo_context_v2"


class ConfidenceValue(BaseModel):
    value: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VisualObservation(BaseModel):
    caption: str = ""
    short_caption: str = ""
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    scenes: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    visual_attributes: list[str] = Field(default_factory=list)
    visible_people_count: int | None = None
    visible_text: list[str] = Field(default_factory=list)
    logos: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    notable_regions: list[str] = Field(default_factory=list)


class VisualInference(BaseModel):
    possible_events: list[ConfidenceValue] = Field(default_factory=list)
    possible_locations: list[ConfidenceValue] = Field(default_factory=list)
    possible_time_periods: list[ConfidenceValue] = Field(default_factory=list)
    possible_relationships: list[ConfidenceValue] = Field(default_factory=list)


class DeterministicMetadata(BaseModel):
    captured_at: str | None = None
    modified_at: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    country: str | None = None
    city: str | None = None
    location_name: str | None = None
    device_make: str | None = None
    device_model: str | None = None
    width: int | None = None
    height: int | None = None
    file_type: str | None = None
    person_ids: list[str] = Field(default_factory=list)
    event_id: str | None = None


class AssetContext(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    asset_id: str = ""
    asset_type: str = "photo"
    observations: VisualObservation = Field(default_factory=VisualObservation)
    inferences: VisualInference = Field(default_factory=VisualInference)
    metadata: DeterministicMetadata = Field(default_factory=DeterministicMetadata)
    raw_ocr_text: str = ""
    normalized_ocr_text: str = ""
    model_name: str = ""
    model_revision: str | None = None
    prompt_version: str = PHOTO_CONTEXT_V2
    pipeline_version: str = PHOTO_CONTEXT_V2
    raw_model_output: str | None = None


def asset_context_from_payload(payload: dict[str, Any]) -> AssetContext:
    """Read v2 records and project legacy records into the v2 contract."""
    if isinstance(payload.get("observations"), dict):
        return AssetContext.model_validate(payload)

    visible_text = _strings(payload.get("ocr_text") or payload.get("text_in_image"))
    location_clues = _strings(payload.get("location_clues"))
    time_clues = _strings(payload.get("time_clues"))
    events = _strings(payload.get("events"))
    relationships = _strings(payload.get("people"))
    return AssetContext(
        asset_id=str(payload.get("asset_id") or ""),
        asset_type=str(payload.get("asset_type") or "photo"),
        observations=VisualObservation(
            caption=str(payload.get("dense_caption") or payload.get("caption") or ""),
            short_caption=str(payload.get("caption") or ""),
            objects=_strings(payload.get("objects")),
            actions=_strings(payload.get("activities")),
            scenes=_strings(payload.get("scene")),
            colors=_strings(payload.get("colors")),
            visual_attributes=_strings(payload.get("visual_attributes")),
            visible_text=visible_text,
            relationships=relationships,
        ),
        inferences=VisualInference(
            possible_events=_legacy_inferences(events + _strings(payload.get("event"))),
            possible_locations=_legacy_inferences(location_clues + _strings(payload.get("location"))),
            possible_time_periods=_legacy_inferences(time_clues + _strings(payload.get("time"))),
        ),
        raw_ocr_text="\n".join(visible_text),
        normalized_ocr_text="\n".join(visible_text),
        model_name=str(payload.get("model_name") or "legacy_photo_context"),
        prompt_version=str(payload.get("schema_version") or "photo_context"),
        pipeline_version=str(payload.get("schema_version") or "photo_context"),
    )


def asset_context_to_legacy_projection(context: AssetContext) -> dict[str, object]:
    """Keep existing readers working while v2 sections remain the source of truth."""
    observations = context.observations
    return {
        "schema_version": PHOTO_CONTEXT_V2,
        "asset_id": context.asset_id,
        "asset_type": context.asset_type,
        "observations": context.observations.model_dump(),
        "inferences": context.inferences.model_dump(),
        "metadata": context.metadata.model_dump(),
        "raw_ocr_text": context.raw_ocr_text,
        "normalized_ocr_text": context.normalized_ocr_text,
        "model_name": context.model_name,
        "model_revision": context.model_revision,
        "prompt_version": context.prompt_version,
        "pipeline_version": context.pipeline_version,
        "raw_model_output": context.raw_model_output,
        "caption": observations.short_caption or observations.caption,
        "dense_caption": observations.caption,
        "objects": observations.objects,
        "people": observations.relationships,
        "scene": " ".join(observations.scenes),
        "activities": observations.actions,
        "text_in_image": context.raw_ocr_text,
        "ocr_text": context.normalized_ocr_text,
        "colors": observations.colors,
        "visual_attributes": observations.visual_attributes,
        "tags": _unique(observations.objects + observations.actions + observations.scenes + observations.logos),
    }


def _legacy_inferences(values: list[str]) -> list[ConfidenceValue]:
    # Legacy context did not expose confidence. Preserve it as low-confidence inference.
    return [ConfidenceValue(value=value, confidence=0.4) for value in _unique(values)]


def _strings(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value and str(value).strip():
        return [str(value).strip()]
    return []


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
