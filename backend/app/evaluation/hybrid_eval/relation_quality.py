"""Fresh image/caption relation preferences, fixed gates and paired diagnostics."""

from __future__ import annotations

import math
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from backend.app.evaluation.hybrid_eval import relation_source, reporting
from backend.app.retrieval.hybrid import category_gating, vision_reranking

Progress = Callable[[int, int], None]


def _summary(rows: Sequence[reporting.JsonObject]) -> reporting.JsonObject:
    same_bag = [row for row in rows if row["same_word_bag"]]
    return {
        "cases": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "ties": sum(row["tie"] for row in rows),
        "failures": sum(row["failed"] for row in rows),
        "seconds_per_case": sum(row["seconds"] for row in rows) / len(rows),
        "same_word_bag_cases": len(same_bag),
        "same_word_bag_accuracy": sum(row["correct"] for row in same_bag) / len(same_bag) if same_bag else None,
    }


def _measure(source, model_id, identity, score, progress):
    """Count every case, including explicitly reported arbitrary provider failures."""
    if not source.cases or not isinstance(model_id, str) or not model_id:
        raise ValueError("Relation evaluation needs explicit cases and model identity")
    rows = []
    for index, case in enumerate(source.cases):
        if identity() != model_id:
            raise ValueError("Relation model identity changed before scoring")
        started = time.perf_counter()
        values, error, unique = (), None, None
        try:
            values, unique = score(case)
            values = tuple(float(value) for value in values)
            if len(values) != 2 or any(not math.isfinite(value) for value in values):
                raise ValueError("Relation provider requires two finite aligned scores")
        except Exception as failure:
            # This is the explicit external model boundary: retain the failed
            # case in all denominators and forbid passing the quality gate.
            error = {"type": type(failure).__name__, "message": str(failure)[:240]}
        seconds = time.perf_counter() - started
        if identity() != model_id:
            raise ValueError("Relation model identity changed during scoring")
        correct = error is None and values[case.positive_index] > values[1 - case.positive_index]
        same_bag = Counter(re.findall(r"\w+", case.positive.casefold())) == Counter(
            re.findall(r"\w+", case.negative.casefold())
        )
        rows.append(
            {
                "case_id": case.case_id,
                "kind": case.kind,
                "partition": case.partition,
                "positive_index": case.positive_index,
                "captions": case.captions,
                "scores": values if error is None else None,
                "correct": bool(correct),
                "tie": error is None and values[0] == values[1],
                "failed": error is not None,
                "error": error,
                "seconds": seconds,
                "reported_unique_vision_images": unique,
                "same_word_bag": same_bag,
            }
        )
        if progress is not None:
            progress(index + 1, len(source.cases))
    report = {
        "schema": "lentica-relation-quality-v1",
        "dataset_identity": source.identity,
        "model_id": model_id,
        "overall": _summary(rows),
        "partitions": {
            part: _summary([row for row in rows if row["partition"] == part]) for part in relation_source.PARTITIONS
        },
        "kinds": {kind: _summary([row for row in rows if row["kind"] == kind]) for kind in relation_source.KINDS},
        "per_case": rows,
        "scope": "Actual per-case image and competing-caption inference; strict positive>negative, ties fail. Not cross-image Recall, open-world confidence, training or default selection. Data/model preparation excluded from query costs.",
    }
    report["ranking_sha256"] = reporting.digest(
        [{key: row[key] for key in ("case_id", "positive_index", "correct", "tie", "failed")} for row in rows]
    )
    return report


def measure_dual(
    source: relation_source.RelationDataset,
    encoder: vision_reranking.ImageTextEncoder,
    *,
    progress: Progress | None = None,
) -> reporting.JsonObject:
    """Encode every image and both texts afresh, with no cross-case query cache."""

    def score(case):
        image = category_gating._features(encoder.encode_images([source.directory / case.filename]), 1)
        texts = category_gating._features(encoder.encode_texts(case.captions), 2, image.shape[1])
        return (image @ texts.T)[0], 1

    return _measure(source, encoder.encoder_id, lambda: encoder.encoder_id, score, progress)


def measure_joint(
    source: relation_source.RelationDataset,
    scorer: vision_reranking.ImageTextPairScorer,
    *,
    progress: Progress | None = None,
) -> reporting.JsonObject:
    """Actually run pair-conditioned image/text matching for every frozen case."""

    def score(case):
        scores = scorer.score_pairs(case.captions, [source.directory / case.filename] * 2)
        return scores, getattr(scorer, "last_unique_images", None)

    return _measure(source, scorer.scorer_id, lambda: scorer.scorer_id, score, progress)


def _paired(first, second):
    differences = [int(a["correct"]) - int(b["correct"]) for a, b in zip(first, second)]
    rng = random.Random(42)
    means = sorted(sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(1000))
    return {
        "accuracy_delta": sum(differences) / len(differences),
        "improved": sum(value > 0 for value in differences),
        "regressed": sum(value < 0 for value in differences),
        "tied": sum(value == 0 for value in differences),
        "paired_bootstrap_95_percentile_interval": [means[24], means[974]],
        "bootstrap_samples": 1000,
        "seed": 42,
        "scope": "Descriptive paired resampling for fixed source images; no pretraining/generalization or multiple-search correction.",
    }


def _checked_partitions(source, name, report):
    """Recompute summaries from checked case scores; aggregate fields are not proof."""
    rows = report["per_case"]
    if (
        report["schema"] != "lentica-relation-quality-v1"
        or report["dataset_identity"] != source.identity
        or not isinstance(report["model_id"], str)
        or not report["model_id"]
        or report.get("model_kind", name) != name
        or len(rows) != len(source.cases)
    ):
        raise ValueError("Relation comparison source/model/case identity differs")
    for row, case in zip(rows, source.cases):
        if (
            tuple(row[key] for key in ("case_id", "positive_index", "kind", "partition"))
            != (case.case_id, case.positive_index, case.kind, case.partition)
            or tuple(row["captions"]) != case.captions
            or row["same_word_bag"]
            != (
                Counter(re.findall(r"\w+", case.positive.casefold()))
                == Counter(re.findall(r"\w+", case.negative.casefold()))
            )
            or any(type(row[key]) is not bool for key in ("correct", "tie", "failed", "same_word_bag"))
            or type(row["seconds"]) not in (int, float)
            or not math.isfinite(row["seconds"])
            or row["seconds"] < 0
        ):
            raise ValueError("Relation comparison case labels/cost differ")
        scores = row["scores"]
        if row["failed"]:
            if scores is not None or row["correct"] or row["tie"]:
                raise ValueError("Failed relation case cannot count as correct")
        elif (
            len(scores) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in scores)
            or row["correct"] != (scores[case.positive_index] > scores[1 - case.positive_index])
            or row["tie"] != (scores[0] == scores[1])
        ):
            raise ValueError("Relation comparison scores and decisions differ")
    parts = {part: _summary([row for row in rows if row["partition"] == part]) for part in relation_source.PARTITIONS}
    kinds = {kind: _summary([row for row in rows if row["kind"] == kind]) for kind in relation_source.KINDS}
    ranking = reporting.digest(
        [{key: row[key] for key in ("case_id", "positive_index", "correct", "tie", "failed")} for row in rows]
    )
    if (
        parts != report["partitions"]
        or _summary(rows) != report["overall"]
        or kinds != report["kinds"]
        or ranking != report["ranking_sha256"]
    ):
        raise ValueError("Relation comparison summaries differ from actual cases")
    return parts


def compare(
    source: relation_source.RelationDataset, reports: Mapping[str, reporting.JsonObject]
) -> reporting.JsonObject:
    """Check matching cases and apply predeclared quality/cost gates, without tuning."""
    if set(reports) != {"clip", "siglip", "blip"}:
        raise ValueError("Relation comparison requires all three fixed models")
    checked = {name: _checked_partitions(source, name, report) for name, report in reports.items()}
    partitions = {}
    for part in relation_source.PARTITIONS:
        quality = {name: parts[part] for name, parts in checked.items()}
        joint = quality["blip"]
        strongest = max(quality[name]["accuracy"] for name in ("clip", "siglip"))
        passed = (
            joint["accuracy"] >= 0.6
            and joint["accuracy"] >= strongest + 0.05 - 1e-12
            and joint["seconds_per_case"] <= 0.5
            and all(summary["failures"] == 0 for summary in quality.values())
        )
        pairs = {
            name: _paired(
                [row for row in reports["blip"]["per_case"] if row["partition"] == part],
                [row for row in reports[name]["per_case"] if row["partition"] == part],
            )
            for name in ("clip", "siglip")
        }
        partitions[part] = {"models": quality, "gate_passed": passed, "paired_joint_vs_dual": pairs}
    return {
        "schema": "lentica-relation-comparison-v1",
        "dataset_identity": source.identity,
        "model_ids": {name: report["model_id"] for name, report in reports.items()},
        "partitions": partitions,
        "eligible_for_promotion": all(result["gate_passed"] for result in partitions.values()),
        "policy": "Fixed native profiles, no fitting/tuning: every split joint accuracy>=.6, gain>=.05 vs stronger dual, <=.5 actual seconds/case and no failed cases. Test cannot reverse dev/validation failure.",
        "scope": "Frozen SugarCrepe subset; official COCO-finetuned BLIP/pretraining overlap not certified absent. Strict caption preference is not full retrieval quality or product serving latency.",
    }
