from __future__ import annotations

import json
import re
from pathlib import Path

from backend.app.models.asset_context import AssetContext, asset_context_to_legacy_projection
from backend.app.models.photo_context_prompt import PHOTO_CONTEXT_PROMPT


class ModelUnavailable(RuntimeError):
    pass


class Captioner:
    def caption(self, image_path: Path) -> dict[str, object]:
        raise NotImplementedError


class BasicCaptioner(Captioner):
    def caption(self, image_path: Path) -> dict[str, object]:
        tags = [image_path.suffix.lower().lstrip("."), image_path.stem]
        return {
            "caption": f"Local image file named {image_path.name}. Vision captioning model is not loaded.",
            "dense_caption": f"Local image file named {image_path.name}.",
            "objects": [],
            "people": [],
            "scene": "",
            "activities": [],
            "text_in_image": "",
            "time_clues": [],
            "location_clues": [],
            "location": "",
            "time": "",
            "events": [],
            "event": "",
            "colors": [],
            "style": "",
            "quality": "",
            "visual_attributes": [],
            "search_phrases": [image_path.stem],
            "negative_clues": [],
            "tags": tags,
            "ocr_text": "",
        }


class QwenVlCaptioner(Captioner):
    """Local Qwen-VL captioner adapted from apps/photo_rag.

    Qwen-VL is used only to collect local photo context. Search remains CLIP/vector based.
    """

    def __init__(self, model_name: str, device: str = "cuda", max_pixels: int = 768 * 768):
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise ModelUnavailable(
                "Install torch, transformers and qwen-vl-utils in the worker environment to use Qwen-VL."
            ) from exc

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, max_pixels=max_pixels)
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_available() else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

    def caption(self, image_path: Path) -> dict[str, object]:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": PHOTO_CONTEXT_PROMPT}],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=256)
        generated = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], generated)]
        caption = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return normalize_vision_context(caption)


def normalize_vision_context(raw: str) -> dict[str, object]:
    parsed = _extract_json(raw)
    if parsed is None:
        tags = [part.strip(" #，,。") for part in raw.replace("\n", "，").split("，") if 1 < len(part.strip()) < 16]
        return {
            "caption": raw[:240],
            "dense_caption": raw,
            "objects": [],
            "people": [],
            "scene": "",
            "activities": [],
            "text_in_image": "",
            "time_clues": [],
            "location_clues": [],
            "location": "",
            "events": [],
            "event": "",
            "time": "",
            "colors": [],
            "style": "",
            "quality": "",
            "visual_attributes": [],
            "search_phrases": [],
            "negative_clues": [],
            "tags": tags[:24],
            "ocr_text": "",
        }
    if isinstance(parsed.get("observations"), dict):
        try:
            context = _normalize_v2_context(parsed, raw)
            return asset_context_to_legacy_projection(context)
        except Exception:
            # Preserve the raw response in the legacy fallback instead of dropping the audit evidence.
            pass

    context = _default_context()
    for key in context:
        if key in parsed:
            context[key] = parsed[key]
    context["caption"] = str(context.get("caption") or context.get("dense_caption") or "")[:500]
    context["dense_caption"] = str(context.get("dense_caption") or context["caption"])
    context["text_in_image"] = str(context.get("text_in_image") or "")
    context["ocr_text"] = context["text_in_image"]
    for key in [
        "objects",
        "people",
        "activities",
        "time_clues",
        "location_clues",
        "events",
        "colors",
        "visual_attributes",
        "search_phrases",
        "negative_clues",
        "tags",
    ]:
        context[key] = _list_of_strings(context.get(key))
    for key in ["scene", "location", "event", "time", "style", "quality"]:
        context[key] = str(context.get(key) or "")
    if not context["tags"]:
        context["tags"] = _derive_tags(context)
    if not context["search_phrases"]:
        context["search_phrases"] = _derive_search_phrases(context)
    return context


def _normalize_v2_context(parsed: dict[str, object], raw: str) -> AssetContext:
    observations = parsed.get("observations") if isinstance(parsed.get("observations"), dict) else {}
    inferences = parsed.get("inferences") if isinstance(parsed.get("inferences"), dict) else {}
    normalized = {
        "asset_id": str(parsed.get("asset_id") or ""),
        "asset_type": str(parsed.get("asset_type") or "photo"),
        "observations": {
            "caption": str(observations.get("caption") or "")[:1200],
            "short_caption": str(observations.get("short_caption") or "")[:500],
            "objects": _list_of_strings(observations.get("objects")),
            "actions": _list_of_strings(observations.get("actions")),
            "scenes": _list_of_strings(observations.get("scenes")),
            "colors": _list_of_strings(observations.get("colors")),
            "visual_attributes": _list_of_strings(observations.get("visual_attributes")),
            "visible_people_count": _optional_int(observations.get("visible_people_count")),
            "visible_text": _list_of_strings(observations.get("visible_text")),
            "logos": _list_of_strings(observations.get("logos")),
            "relationships": _list_of_strings(observations.get("relationships")),
            "notable_regions": _list_of_strings(observations.get("notable_regions")),
        },
        "inferences": {
            "possible_events": _confidence_values(inferences.get("possible_events")),
            "possible_locations": _confidence_values(inferences.get("possible_locations")),
            "possible_time_periods": _confidence_values(inferences.get("possible_time_periods")),
            "possible_relationships": _confidence_values(inferences.get("possible_relationships")),
        },
        "raw_ocr_text": str(parsed.get("raw_ocr_text") or ""),
        "normalized_ocr_text": str(parsed.get("normalized_ocr_text") or parsed.get("raw_ocr_text") or ""),
        "model_name": str(parsed.get("model_name") or "qwen-vl"),
        "model_revision": str(parsed.get("model_revision") or "") or None,
        "prompt_version": str(parsed.get("schema_version") or "photo_context_v2"),
        "pipeline_version": "photo_context_v2",
        "raw_model_output": raw,
    }
    return AssetContext.model_validate(normalized)


def _confidence_values(value: object) -> list[dict[str, object]]:
    items = value if isinstance(value, list) else [value]
    normalized: list[dict[str, object]] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("value") or "").strip()
            confidence = item.get("confidence", 0.0)
        else:
            text = str(item or "").strip()
            confidence = 0.0
        if not text:
            continue
        try:
            score = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            score = 0.0
        normalized.append({"value": text, "confidence": score})
    return normalized[:16]


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _default_context() -> dict[str, object]:
    return {
        "caption": "",
        "dense_caption": "",
        "objects": [],
        "people": [],
        "scene": "",
        "activities": [],
        "text_in_image": "",
        "time_clues": [],
        "location_clues": [],
        "location": "",
        "events": [],
        "event": "",
        "time": "",
        "colors": [],
        "style": "",
        "quality": "",
        "visual_attributes": [],
        "search_phrases": [],
        "negative_clues": [],
        "tags": [],
        "ocr_text": "",
    }


def _extract_json(raw: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _list_of_strings(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:32]
    if isinstance(value, str) and value.strip():
        return [part.strip(" #，,。") for part in value.split("，") if part.strip()][:32]
    return []


def _derive_tags(context: dict[str, object]) -> list[str]:
    tags: list[str] = []
    for key in ["scene", "objects", "activities", "events", "location_clues", "time_clues", "colors"]:
        value = context.get(key)
        if isinstance(value, list):
            tags.extend(str(item) for item in value)
        elif value:
            tags.append(str(value))
    return [tag for tag in tags if 1 < len(tag) < 24][:24]


def _derive_search_phrases(context: dict[str, object]) -> list[str]:
    phrases: list[str] = []
    caption = str(context.get("caption") or "")
    if caption:
        phrases.append(caption[:48])
    for key in ["event", "location", "scene", "tags", "activities", "objects", "people"]:
        value = context.get(key)
        if isinstance(value, list):
            phrases.extend(str(item) for item in value)
        elif value:
            phrases.append(str(value))
    return [phrase for phrase in phrases if 1 < len(phrase) < 48][:12]


def build_captioner(kind: str, model_name: str, device: str, max_pixels: int) -> Captioner:
    if kind == "qwen-vl":
        return QwenVlCaptioner(model_name=model_name, device=device, max_pixels=max_pixels)
    if kind == "basic":
        return BasicCaptioner()
    raise ValueError(f"Unknown captioner: {kind}")
