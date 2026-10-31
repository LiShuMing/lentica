"""Joint fast/slow cascade calibration using actual development query inference."""

from __future__ import annotations

from backend.app.evaluation.hybrid_eval import cascade_quality, image_quality, mirror_images, reporting, rescue_quality
from backend.app.retrieval.hybrid import category_gating, vision_reranking

FAST_GRID = rescue_quality.CALIBRATION_MARGINS
CONSTRAINTS = {
    "development_fpr": 0.05,
    "test_fpr": 0.1,
    "hit_loss": 0.1,
    "ndcg_loss": 0.05,
    "seconds_per_query_image": 0.2,
}


def _gate(
    fast: vision_reranking.ImageTextEncoder, slow: vision_reranking.ImageTextEncoder, tolerance: float
) -> category_gating.CascadedCategoryGate:
    """Construct the explicitly calibrated policy without changing strict defaults."""
    return category_gating.CascadedCategoryGate(
        fast, slow, cascade_quality.PROMPTS, cascade_quality.GALLERY, fast_negative_margin=tolerance
    )


def _passes(report: reporting.JsonObject, margin: float, *, fpr: float) -> bool:
    return (
        rescue_quality._acceptable(report["thresholds"][str(margin)], report["full_slow_reference"]["raw_gallery"], fpr)
        and report["cost"]["seconds_per_query_image"] <= CONSTRAINTS["seconds_per_query_image"]
    )


def select(
    source: mirror_images.ImageDataset,
    fast: vision_reranking.ImageTextEncoder,
    slow: vision_reranking.ImageTextEncoder,
) -> reporting.JsonObject:
    """Run every fixed candidate on development, then validate only one pair.

    Only gallery/reference preparation is shared across candidates. Every fast
    query and eligible slow query is actually encoded for each candidate; its
    per-batch invocation IDs/costs are retained. No test partition is accessed.
    """
    fast_id, slow_id = fast.encoder_id, slow.encoder_id
    prepared = rescue_quality.prepare_gallery(source, slow, "dev")
    development, gates, viable = {}, {}, []
    for tolerance in FAST_GRID:
        gate = _gate(fast, slow, tolerance)
        report = rescue_quality.measure(
            source, gate, slow, "dev", margins=image_quality.MARGINS, fast_negative_margin=tolerance, prepared=prepared
        )
        gates[tolerance], development[str(tolerance)] = gate, report
        viable.extend(
            (tolerance, margin)
            for margin in image_quality.MARGINS
            if _passes(report, margin, fpr=CONSTRAINTS["development_fpr"])
        )
    proposed_fast, proposed_slow = min(viable) if viable else (FAST_GRID[0], -1.0)
    gate = gates[proposed_fast]
    validation = rescue_quality.measure(
        source, gate, slow, "validation", margins=(proposed_slow,), fast_negative_margin=proposed_fast
    )
    accepted = bool(viable) and _passes(validation, proposed_slow, fpr=CONSTRAINTS["development_fpr"])
    if fast.encoder_id != fast_id or slow.encoder_id != slow_id:
        raise ValueError("Cascade model identity changed during calibration")
    result = {
        "schema": "lentica-calibrated-rescue-selection-v1",
        "dataset_identity": source.identity,
        "fast_encoder_id": fast_id,
        "slow_encoder_id": slow_id,
        "gate_id": gate.gate_id,
        "fast_grid": FAST_GRID,
        "slow_grid": image_quality.MARGINS,
        "constraints": dict(CONSTRAINTS),
        "proposed_fast_margin": proposed_fast,
        "proposed_slow_margin": proposed_slow,
        "selected_fast_margin": proposed_fast,
        "selected_margin": proposed_slow if accepted else -1.0,
        "accepted": accepted,
        "development": development,
        "validation": validation,
        "prompt_encoding_seconds": sum(item.prompt_encoding_seconds for item in gates.values()),
        "preparation_scope": "One actual dev gallery and one full slow dev reference shared; every candidate query is fresh. Validation preparation/query inference is separate.",
        "policy": "Smallest feasible fast margin then smallest slow margin; only one validation candidate; no test reversal or gate relaxation.",
    }
    result["sha256"] = reporting.digest(result)
    return result


def evaluate(
    source: mirror_images.ImageDataset,
    fast: vision_reranking.ImageTextEncoder,
    slow: vision_reranking.ImageTextEncoder,
    selection: reporting.JsonObject,
) -> reporting.JsonObject:
    """Reject wrong locks before inference; report test-only and selection gates separately."""
    frozen = dict(selection)
    digest = frozen.pop("sha256")
    if (
        reporting.digest(frozen) != digest
        or frozen["schema"] != "lentica-calibrated-rescue-selection-v1"
        or frozen["dataset_identity"] != source.identity
        or frozen["fast_encoder_id"] != fast.encoder_id
        or frozen["slow_encoder_id"] != slow.encoder_id
        or list(frozen["fast_grid"]) != list(FAST_GRID)
        or list(frozen["slow_grid"]) != list(image_quality.MARGINS)
        or frozen["constraints"] != CONSTRAINTS
        or type(frozen["accepted"]) is not bool
        or type(frozen["selected_fast_margin"]) not in (int, float)
        or frozen["selected_fast_margin"] not in FAST_GRID
        or type(frozen["selected_margin"]) not in (int, float)
        or frozen["selected_margin"] not in image_quality.MARGINS
    ):
        raise ValueError("Frozen calibrated cascade source/model/config identity differs")
    gate = _gate(fast, slow, frozen["selected_fast_margin"])
    if gate.gate_id != frozen["gate_id"]:
        raise ValueError("Frozen calibrated cascade gate identity differs")
    result = rescue_quality.measure(
        source,
        gate,
        slow,
        "test",
        margins=(frozen["selected_margin"],),
        fast_negative_margin=frozen["selected_fast_margin"],
    )
    selected = result["thresholds"][str(frozen["selected_margin"])]
    result.update(
        selection_sha256=digest,
        selection_accepted=frozen["accepted"],
        selected_fast_margin=frozen["selected_fast_margin"],
        selected_margin=frozen["selected_margin"],
        selected_no_answer=selected,
        prompt_encoding_seconds=gate.prompt_encoding_seconds,
        quality_gate_passed=rescue_quality._acceptable(
            selected, result["full_slow_reference"]["raw_gallery"], CONSTRAINTS["test_fpr"]
        ),
        cost_gate_passed=result["cost"]["seconds_per_query_image"] <= CONSTRAINTS["seconds_per_query_image"],
    )
    result["heldout_gate_passed"] = result["quality_gate_passed"] and result["cost_gate_passed"]
    result["eligible_for_promotion"] = result["selection_accepted"] and result["heldout_gate_passed"]
    result["ranking_sha256"] = reporting.digest(
        {"rescue": result["rankings"], "reference": result["full_slow_reference"]["rankings"]}
    )
    return result
