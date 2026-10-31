"""Source-isolated cascade selection, refusal/cost gates and full-model controls."""

from __future__ import annotations

import time
from collections.abc import Sequence

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation.hybrid_eval import image_quality, mirror_images, reporting
from backend.app.retrieval.hybrid import category_gating, vision_reranking

MARGINS = image_quality.MARGINS
PROMPTS = dict(zip(mirror_images.CLASSES, image_quality.PROMPTS))
GALLERY = image_quality.GALLERY_CLASSES
COST_LIMIT = 0.2


def _queries(items, scores):
    """Create class qrels independently of inference, with self exclusion."""
    ids = [item.identifier for item in items]
    positive, open_cases, full, gallery = [], [], {}, {}
    for index, item in enumerate(items):
        key = str(item.identifier)
        relevant = {
            other.identifier for other in items if other.label == item.label and other.identifier != item.identifier
        }
        positive.append(evaluation_dataset.EvaluationCase(key, "same dominant class", relevant, item.label))
        open_cases.append(
            evaluation_dataset.EvaluationCase(
                key, "same class in declared gallery", relevant if item.label in GALLERY else set(), item.label
            )
        )
        ordered = sorted(
            (row for row in range(len(items)) if row != index), key=lambda row: (-float(scores[index, row]), ids[row])
        )
        full[key] = tuple(ids[row] for row in ordered[:10])
        gallery[key] = tuple(ids[row] for row in ordered if items[row].label in GALLERY)[:10]
    return positive, open_cases, full, gallery


def _reference(source, items, model):
    """Fresh full SigLIP inference, timed separately from cascade execution."""
    started = time.perf_counter()
    vectors = model.encode_images([source.directory / item.filename for item in items])
    image_seconds = time.perf_counter() - started
    started = time.perf_counter()
    texts = model.encode_texts(tuple(PROMPTS.values()))
    text_seconds = time.perf_counter() - started
    positive, open_cases, full, raw = _queries(items, vectors @ vectors.T)
    classes = vectors @ texts.T
    category = {
        str(item.identifier): raw[str(item.identifier)] if mirror_images.CLASSES[int(row.argmax())] in GALLERY else ()
        for item, row in zip(items, classes)
    }
    return {
        "encoder_id": model.encoder_id,
        "images": len(items),
        "image_seconds": image_seconds,
        "text_seconds": text_seconds,
        "seconds_per_image": image_seconds / len(items),
        "full_image_quality": image_quality._metrics(positive, full),
        "raw_gallery": image_quality._metrics(open_cases, raw),
        "category_gate": image_quality._metrics(open_cases, category),
        "rankings": {"full_image": full, "raw_gallery": raw, "category_gate": category},
    }


def _batches(source, items, gate):
    """Keep actual batch timing and the exact slow-model call domain together."""
    batches, details = [], []
    for start in range(0, len(items), gate.batch_size):
        chosen = items[start : start + gate.batch_size]
        result = gate.score_images([source.directory / item.filename for item in chosen])
        batches.append(result)
        details.append(
            {
                "ids": [item.identifier for item in chosen],
                "slow_ids": [item.identifier for item, accepted in zip(chosen, result.fast_accepts) if accepted],
                "seconds": result.total_seconds,
            }
        )
    return batches, details


def _cost(batches, images):
    total = sum(batch.total_seconds for batch in batches)
    return {
        "fast_images": images,
        "slow_images": sum(sum(batch.fast_accepts) for batch in batches),
        "fast_image_seconds": sum(batch.fast_seconds for batch in batches),
        "slow_image_seconds": sum(batch.slow_seconds for batch in batches),
        "total_cascade_seconds": total,
        "seconds_per_query_image": total / images,
        "batch_p50_ms": reporting.percentile([batch.total_seconds for batch in batches], 0.5),
        "batch_p95_ms": reporting.percentile([batch.total_seconds for batch in batches], 0.95),
        "scope": "Actual bounded image batches including gate math; prompt/model construction and full reference comparison excluded. Not single-request serving latency.",
    }


def measure(
    source: mirror_images.ImageDataset,
    gate: category_gating.CascadedCategoryGate,
    reference: vision_reranking.ImageTextEncoder,
    partition: str,
    *,
    margins: Sequence[float],
) -> reporting.JsonObject:
    """Measure actual batches and a fresh independent full-model comparator.

    Fast image features are reused for self-excluding query/gallery ranks within
    this partition. Slow features only classify eligible queries. Full SigLIP
    comparison executes separately and is excluded from cascade cost, not hidden
    as a cache hit or free preprocessing. Source labels define qrels only.
    """
    if (
        gate.slow_encoder_id != reference.encoder_id
        or gate.gallery_classes != GALLERY
        or gate.labels != tuple(PROMPTS)
        or gate.prompts != tuple(PROMPTS.values())
        or gate.batch_size != 8
    ):
        raise ValueError("Cascade/reference configuration differs from declared protocol")
    items = source.partition(partition)
    if len(items) < 20:
        raise ValueError("Cascade quality requires enough source-isolated images")
    batches, details = _batches(source, items, gate)
    import numpy as np

    vectors = np.concatenate([batch.fast_vectors for batch in batches])
    positive, open_cases, full, raw = _queries(items, vectors @ vectors.T)
    fast_pass = tuple(value for batch in batches for value in batch.fast_accepts)
    fast_ranks = {
        str(item.identifier): raw[str(item.identifier)] if accepted else () for item, accepted in zip(items, fast_pass)
    }
    ranks = {}
    for margin in margins:
        accepted = tuple(value for batch in batches for value in batch.decisions(margin))
        ranks[str(margin)] = {
            str(item.identifier): raw[str(item.identifier)] if keep else () for item, keep in zip(items, accepted)
        }
    return {
        "partition": partition,
        "images": len(items),
        "gate_id": gate.gate_id,
        "fast_full_quality": image_quality._metrics(positive, full),
        "raw_gallery": image_quality._metrics(open_cases, raw),
        "fast_category_gate": image_quality._metrics(open_cases, fast_ranks),
        "thresholds": {key: image_quality._metrics(open_cases, values) for key, values in ranks.items()},
        "cost": _cost(batches, len(items)),
        "batches": details,
        "rankings": {"fast_full": full, "fast_gallery": raw, "fast_gate": fast_ranks, "cascade": ranks},
        "full_slow_reference": _reference(source, items, reference),
    }


def _acceptable(candidate, raw, *, fpr_limit):
    return (
        candidate["no_answer_false_positive_rate"] <= fpr_limit
        and candidate["positive"]["hit_at_10"] >= raw["positive"]["hit_at_10"] - 0.1
    )


def select(
    source: mirror_images.ImageDataset,
    gate: category_gating.CascadedCategoryGate,
    reference: vision_reranking.ImageTextEncoder,
) -> reporting.JsonObject:
    """Select on development only, validate once, and freeze before test."""
    development = measure(source, gate, reference, "dev", margins=MARGINS)
    candidates = [
        margin
        for margin in MARGINS
        if _acceptable(development["thresholds"][str(margin)], development["raw_gallery"], fpr_limit=0.05)
        and development["cost"]["seconds_per_query_image"] <= COST_LIMIT
    ]
    proposed = min(candidates) if candidates else -1.0
    validation = measure(source, gate, reference, "validation", margins=(proposed,))
    accepted = (
        bool(candidates)
        and _acceptable(validation["thresholds"][str(proposed)], validation["raw_gallery"], fpr_limit=0.05)
        and validation["cost"]["seconds_per_query_image"] <= COST_LIMIT
    )
    artifact = {
        "schema": "lentica-cascade-selection-v1",
        "dataset_identity": source.identity,
        "gate_id": gate.gate_id,
        "prompts": PROMPTS,
        "gallery_classes": sorted(GALLERY),
        "margin_grid": MARGINS,
        "cost_limit": COST_LIMIT,
        "proposed_margin": proposed,
        "selected_margin": proposed if accepted else -1.0,
        "accepted": accepted,
        "development": development,
        "validation": validation,
        "decision": "Fixed two-category-gate fallback if validation fails; no default switch or test-based reversal.",
    }
    artifact["sha256"] = reporting.digest(artifact)
    return artifact


def evaluate(
    source: mirror_images.ImageDataset,
    gate: category_gating.CascadedCategoryGate,
    reference: vision_reranking.ImageTextEncoder,
    selection: reporting.JsonObject,
) -> reporting.JsonObject:
    """Check source/model/policy lock before any held-out image inference."""
    artifact = dict(selection)
    digest = artifact.pop("sha256")
    if (
        reporting.digest(artifact) != digest
        or artifact["schema"] != "lentica-cascade-selection-v1"
        or artifact["dataset_identity"] != source.identity
        or artifact["gate_id"] != gate.gate_id
        or artifact["prompts"] != PROMPTS
        or artifact["gallery_classes"] != sorted(GALLERY)
        or list(artifact["margin_grid"]) != list(MARGINS)
        or artifact["cost_limit"] != COST_LIMIT
        or type(artifact["accepted"]) is not bool
        or type(artifact["selected_margin"]) not in (int, float)
        or artifact["selected_margin"] not in MARGINS
    ):
        raise ValueError("Frozen cascade source/model/config identity differs")
    result = measure(source, gate, reference, "test", margins=(artifact["selected_margin"],))
    selected = result["thresholds"][str(artifact["selected_margin"])]
    result.update(
        selection_sha256=digest,
        selection_accepted=artifact["accepted"],
        selected_margin=artifact["selected_margin"],
        selected_no_answer=selected,
        quality_gate_passed=_acceptable(selected, result["raw_gallery"], fpr_limit=0.1),
        cost_gate_passed=result["cost"]["seconds_per_query_image"] <= COST_LIMIT,
        prompt_encoding_seconds=gate.prompt_encoding_seconds,
    )
    result["heldout_gate_passed"] = result["quality_gate_passed"] and result["cost_gate_passed"]
    result["eligible_for_promotion"] = artifact["accepted"] and result["heldout_gate_passed"]
    result["ranking_sha256"] = reporting.digest(
        {"cascade": result["rankings"], "reference": result["full_slow_reference"]["rankings"]}
    )
    result["scope"] = (
        "Fifth isolated-source known-class task; fast retrieval/slow eligibility trades ranking quality for refusal/cost. Full slow reference shown; not open-world, arbitrary relations or production SLO."
    )
    return result
