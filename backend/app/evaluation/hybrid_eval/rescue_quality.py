"""Confidence rescue with actual slow-space retrieval and separately built galleries."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation.hybrid_eval import cascade_quality, image_quality, mirror_images, reporting
from backend.app.retrieval.hybrid import category_gating, vision_reranking

if TYPE_CHECKING:
    from numpy import typing as npt

FAST_NEGATIVE_MARGIN = 0.02
TIGHT_NEGATIVE_MARGIN = 0.005
CALIBRATION_MARGINS = (0.005, 0.01, 0.015, 0.02)


@dataclass(frozen=True)
class PreparedGallery:
    """One actual gallery/reference run, bound to a source partition and space.

    Gallery vectors are read-only. The full reference report contains rankings,
    not query vectors: it cannot substitute for selective query inference.
    """

    dataset_identity: str
    partition: str
    encoder_id: str
    item_ids: tuple[int, ...]
    gallery_ids: tuple[int, ...]
    vectors: npt.NDArray
    build_seconds: float
    reference: reporting.JsonObject


def prepare_gallery(
    source: mirror_images.ImageDataset, slow: vision_reranking.ImageTextEncoder, partition: str
) -> PreparedGallery:
    """Actually build one reusable gallery and a separate full reference for scoring."""
    encoder_id = slow.encoder_id
    items = source.partition(partition)
    gallery = [item for item in items if item.label in cascade_quality.GALLERY]
    started = time.perf_counter()
    vectors = category_gating._features(
        slow.encode_images([source.directory / item.filename for item in gallery]), len(gallery)
    ).copy()
    vectors.setflags(write=False)
    elapsed = time.perf_counter() - started
    reference = cascade_quality._reference(source, items, slow)
    if slow.encoder_id != encoder_id:
        raise ValueError("Rescue gallery model identity changed during preparation")
    return PreparedGallery(
        source.identity,
        partition,
        encoder_id,
        tuple(item.identifier for item in items),
        tuple(item.identifier for item in gallery),
        vectors,
        elapsed,
        reference,
    )


def _acceptable(quality: reporting.JsonObject, reference: reporting.JsonObject, fpr: float) -> bool:
    return (
        quality["no_answer_false_positive_rate"] <= fpr
        and quality["positive"]["hit_at_10"] >= reference["positive"]["hit_at_10"] - 0.1 - 1e-12
        and quality["positive"]["ndcg_at_10"] >= reference["positive"]["ndcg_at_10"] - 0.05 - 1e-12
    )


def measure(
    source: mirror_images.ImageDataset,
    gate: category_gating.CascadedCategoryGate,
    slow: vision_reranking.ImageTextEncoder,
    partition: str,
    *,
    margins: tuple[float, ...],
    fast_negative_margin: float = FAST_NEGATIVE_MARGIN,
    prepared: PreparedGallery | None = None,
) -> reporting.JsonObject:
    """Build an explicit gallery, then perform fresh bounded query inference.

    Zero placeholders for fast-rejected queries never become retrieval evidence.
    The full slow-model reference is another actual run, not a query cache.
    """
    if (
        fast_negative_margin not in CALIBRATION_MARGINS
        or gate.fast_negative_margin != fast_negative_margin
        or gate.batch_size != 8
        or dict(zip(gate.labels, gate.prompts)) != cascade_quality.PROMPTS
        or gate.gallery_classes != cascade_quality.GALLERY
        or gate.slow_encoder_id != slow.encoder_id
    ):
        raise ValueError("Rescue gate differs from the frozen protocol")
    items = source.partition(partition)
    gallery = [item for item in items if item.label in cascade_quality.GALLERY]
    if prepared is not None:
        if (
            prepared.dataset_identity != source.identity
            or prepared.partition != partition
            or prepared.encoder_id != slow.encoder_id
            or prepared.item_ids != tuple(item.identifier for item in items)
            or prepared.gallery_ids != tuple(item.identifier for item in gallery)
        ):
            raise ValueError("Prepared rescue gallery source/partition/model/ID identity differs")
        gallery_vectors, gallery_seconds = prepared.vectors, prepared.build_seconds
    else:
        started = time.perf_counter()
        gallery_vectors = category_gating._features(
            slow.encode_images([source.directory / item.filename for item in gallery]), len(gallery)
        )
        gallery_seconds = time.perf_counter() - started
    batches, details = cascade_quality._batches(source, items, gate)
    costs = cascade_quality._cost(batches, len(items))
    started = time.perf_counter()
    raw, cases = {}, []
    decisions = {str(margin): [] for margin in margins}
    offset = 0
    for batch in batches:
        scores = batch.slow_vectors @ gallery_vectors.T
        for margin in margins:
            decisions[str(margin)].extend(batch.decisions(margin))
        for index, sent in enumerate(batch.fast_accepts):
            item = items[offset + index]
            key = str(item.identifier)
            eligible = [row for row, other in enumerate(gallery) if other.identifier != item.identifier]
            ordered = sorted(eligible, key=lambda row: (-float(scores[index, row]), gallery[row].identifier))
            raw[key] = tuple(gallery[row].identifier for row in ordered[:10]) if sent else ()
            cases.append(
                evaluation_dataset.EvaluationCase(
                    key,
                    "same dominant class within declared gallery",
                    {
                        other.identifier
                        for other in gallery
                        if other.label == item.label and other.identifier != item.identifier
                    },
                    item.label,
                )
            )
        offset += len(batch.fast_accepts)
    ranking_seconds = time.perf_counter() - started
    costs["ranking_seconds"] = ranking_seconds
    costs["seconds_per_query_image"] += ranking_seconds / len(items)
    costs["scope"] += " Ranking time additionally included in average; gallery build reported separately."
    rankings = {
        name: {
            str(item.identifier): raw[str(item.identifier)] if accepted else () for item, accepted in zip(items, chosen)
        }
        for name, chosen in decisions.items()
    }
    reference = prepared.reference if prepared is not None else cascade_quality._reference(source, items, slow)
    return {
        "partition": partition,
        "images": len(items),
        "gallery_images": len(gallery),
        "gallery_build_seconds": gallery_seconds,
        "cost": costs,
        "batches": details,
        "thresholds": {name: image_quality._metrics(cases, ranked) for name, ranked in rankings.items()},
        "rankings": rankings,
        "full_slow_reference": reference,
        "shared_preparation": prepared is not None,
    }


def select(
    source: mirror_images.ImageDataset,
    gate: category_gating.CascadedCategoryGate,
    slow: vision_reranking.ImageTextEncoder,
    *,
    fast_negative_margin: float = FAST_NEGATIVE_MARGIN,
) -> reporting.JsonObject:
    """Choose on development only; validation checks one margin without replacement."""
    dev = measure(source, gate, slow, "dev", margins=image_quality.MARGINS, fast_negative_margin=fast_negative_margin)
    viable = [
        margin
        for margin in image_quality.MARGINS
        if _acceptable(dev["thresholds"][str(margin)], dev["full_slow_reference"]["raw_gallery"], 0.05)
        and dev["cost"]["seconds_per_query_image"] <= 0.2
    ]
    proposed = min(viable) if viable else -1.0
    validation = measure(
        source, gate, slow, "validation", margins=(proposed,), fast_negative_margin=fast_negative_margin
    )
    accepted = (
        bool(viable)
        and _acceptable(validation["thresholds"][str(proposed)], validation["full_slow_reference"]["raw_gallery"], 0.05)
        and (validation["cost"]["seconds_per_query_image"] <= 0.2)
    )
    result = {
        "schema": "lentica-rescue-selection-v1",
        "dataset_identity": source.identity,
        "gate_id": gate.gate_id,
        "fast_negative_margin": fast_negative_margin,
        "margin_grid": image_quality.MARGINS,
        "accepted": accepted,
        "proposed_margin": proposed,
        "selected_margin": proposed if accepted else -1.0,
        "development": dev,
        "validation": validation,
        "policy": "dev/validation FPR<=.05, hit loss<=.1, nDCG loss<=.05 vs full slow raw; query cost<=.2; test FPR<=.1",
    }
    result["sha256"] = reporting.digest(result)
    return result


def evaluate(
    source: mirror_images.ImageDataset,
    gate: category_gating.CascadedCategoryGate,
    slow: vision_reranking.ImageTextEncoder,
    selection: reporting.JsonObject,
    *,
    fast_negative_margin: float = FAST_NEGATIVE_MARGIN,
) -> reporting.JsonObject:
    """Check source/model/policy before held-out inference; keep failed selection failed."""
    frozen = dict(selection)
    digest = frozen.pop("sha256")
    if (
        reporting.digest(frozen) != digest
        or frozen["schema"] != "lentica-rescue-selection-v1"
        or frozen["dataset_identity"] != source.identity
        or frozen["gate_id"] != gate.gate_id
        or frozen["fast_negative_margin"] != fast_negative_margin
        or list(frozen["margin_grid"]) != list(image_quality.MARGINS)
        or type(frozen["accepted"]) is not bool
        or type(frozen["selected_margin"]) not in (float, int)
        or frozen["selected_margin"] not in image_quality.MARGINS
    ):
        raise ValueError("Frozen rescue source/model/config identity differs")
    result = measure(
        source, gate, slow, "test", margins=(frozen["selected_margin"],), fast_negative_margin=fast_negative_margin
    )
    selected = result["thresholds"][str(frozen["selected_margin"])]
    result.update(
        selection_sha256=digest,
        selection_accepted=frozen["accepted"],
        selected_margin=frozen["selected_margin"],
        selected_no_answer=selected,
        quality_gate_passed=_acceptable(selected, result["full_slow_reference"]["raw_gallery"], 0.1),
        cost_gate_passed=result["cost"]["seconds_per_query_image"] <= 0.2,
        prompt_encoding_seconds=gate.prompt_encoding_seconds,
    )
    result["heldout_gate_passed"] = result["quality_gate_passed"] and result["cost_gate_passed"]
    result["eligible_for_promotion"] = result["selection_accepted"] and result["heldout_gate_passed"]
    result["ranking_sha256"] = reporting.digest(
        {"rescue": result["rankings"], "reference": result["full_slow_reference"]["rankings"]}
    )
    return result
