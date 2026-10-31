"""Fixed original/translated/English controls with failed queries kept in metrics."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation.hybrid_eval import image_quality, language_quality, mirror_images, reporting
from backend.app.retrieval.hybrid import marian_onnx, vision_reranking

if TYPE_CHECKING:
    from numpy import typing as npt


class QueryTranslator(Protocol):
    """Explicit query adapter; no qrels or expected translation enter this call."""

    version: str

    def translate(self, text: str) -> marian_onnx.Translation:
        """Return generated text or raise; failures remain scored empty rankings."""
        ...


def _translations(translator: QueryTranslator) -> tuple[list[str | None], list[reporting.JsonObject]]:
    prompts, rows = [], []
    version = translator.version
    for original in language_quality.CHINESE:
        started = time.perf_counter()
        try:
            result = translator.translate(original)
            if result.original != original or result.producer != version or not result.text.strip():
                raise ValueError("Translation provenance/text differs from the explicit request")
            prompts.append(result.text)
            row = {"status": "ok", **dataclasses.asdict(result)}
        except Exception as error:
            # Extension boundary: report arbitrary provider failures explicitly;
            # never replace a failed translation with a reference or drop it.
            prompts.append(None)
            row = {
                "status": "failed",
                "original": original,
                "producer": version,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        rows.append({**row, "seconds": time.perf_counter() - started})
    if translator.version != version:
        raise ValueError("Translator identity changed during evaluation")
    return prompts, rows


def _group(
    items: Sequence[mirror_images.ImageItem],
    vectors: npt.NDArray,
    prompts: Sequence[str | None],
    model: vision_reranking.ImageTextEncoder,
) -> tuple[reporting.JsonObject, dict[str, tuple[int, ...]]]:
    """Encode only successful prompts; retain every labeled query in the denominator."""
    started = time.perf_counter()
    positions = [index for index, prompt in enumerate(prompts) if prompt is not None]
    scores = vectors @ model.encode_texts([prompts[index] for index in positions]).T if positions else None
    if scores is not None:
        import numpy as np

        if (
            scores.shape != (len(items), len(positions))
            or scores.dtype.kind not in "fiu"
            or not np.isfinite(scores).all()
        ):
            raise ValueError("Image/text encoder returned invalid aligned scores")
    text_seconds = time.perf_counter() - started
    columns = {position: column for column, position in enumerate(positions)}
    cases, rankings, rows = [], {}, []
    for index, (label, prompt) in enumerate(zip(mirror_images.CLASSES, prompts)):
        case = evaluation_dataset.EvaluationCase(
            label, prompt or "", {item.identifier for item in items if item.label == label}, label
        )
        ranked = (
            ()
            if prompt is None
            else tuple(
                items[row].identifier
                for row in sorted(
                    range(len(items)), key=lambda row: (-float(scores[row, columns[index]]), items[row].identifier)
                )[:10]
            )
        )
        cases.append(case)
        rankings[label] = ranked
        rows.append({"label": label, "prompt": prompt, "metrics": image_quality._metrics([case], {label: ranked})})
    return {
        "metrics": image_quality._metrics(cases, rankings),
        "text_seconds": text_seconds,
        "per_prompt": rows,
    }, rankings


def evaluate(
    source: mirror_images.ImageDataset, model: vision_reranking.ImageTextEncoder, translator: QueryTranslator
) -> reporting.JsonObject:
    """Run all ten predeclared prompts on test images, with no selection or tuning.

    Image features are computed once and shared across all three query groups.
    Only original Chinese strings reach the translator; class labels/reference
    English reach metric calculation and their separately reported control.
    Costs are offline encoding times, not single-request serving latency.
    """
    model_id, version = model.encoder_id, translator.version
    translated, translation_rows = _translations(translator)
    items = source.partition("test")
    started = time.perf_counter()
    vectors = model.encode_images([source.directory / item.filename for item in items])
    image_seconds = time.perf_counter() - started
    groups, rankings = {}, {}
    for name, prompts in (
        ("chinese_original", language_quality.CHINESE),
        ("machine_translation", translated),
        ("english_reference", image_quality.PROMPTS),
    ):
        groups[name], rankings[name] = _group(items, vectors, prompts, model)
    if model.encoder_id != model_id or translator.version != version:
        raise ValueError("Model identity changed during evaluation")
    return {
        "schema": "lentica-query-translation-v1",
        "dataset_identity": source.identity,
        "encoder_id": model_id,
        "translator_id": version,
        "images": len(items),
        "protocol_sha256": reporting.digest(
            {
                "chinese": language_quality.CHINESE,
                "english": image_quality.PROMPTS,
                "translator": version,
                "encoder": model_id,
            }
        ),
        "translations": translation_rows,
        "failed_queries": sum(text is None for text in translated),
        "translation_seconds": sum(row["seconds"] for row in translation_rows),
        "image_encoding_seconds": image_seconds,
        "groups": groups,
        "rankings": rankings,
        "ranking_sha256": reporting.digest(rankings),
        "scope": "Ten fixed known-class queries, source-isolated images; no test tuning, open-world, general translation or Chinese-document retrieval claim. Pretraining overlap unknown.",
    }
