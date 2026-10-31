"""Fixed real sound/text retrieval and separately frozen known-class rejection."""

from __future__ import annotations

import time
from collections.abc import Sequence

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation.hybrid_eval import image_quality, reporting, sound_source
from backend.app.retrieval.hybrid import clap_onnx

PROMPTS = tuple(f"the sound of {label.replace('_', ' ')}" for label in sound_source.CLASSES)
MARGINS = (-1.0, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 1.0)
GALLERY_CLASSES = frozenset(sound_source.CLASSES[:3])


def measure(
    source: sound_source.SoundDataset, model: clap_onnx.ClapOnnxEncoder, partition: str, *, margins: Sequence[float]
) -> reporting.JsonObject:
    """Actually encode one partition; labels define qrels, not sound features.

    Every sound query excludes itself. Same-class matching is a coarse5s-clip
    task; fixed class prompts and a gallery missing two known classes do not
    represent arbitrary queries, exact event boundaries or open-world rejection.
    """
    items = source.partition(partition)
    ids = [item.identifier for item in items]
    started = time.perf_counter()
    vectors = model.encode_audios([source.directory / item.filename for item in items])
    audio_seconds = time.perf_counter() - started
    started = time.perf_counter()
    texts = model.encode_texts(PROMPTS)
    text_seconds = time.perf_counter() - started
    audio_scores, classes = vectors @ vectors.T, vectors @ texts.T
    full, raw, gated = {}, {}, {}
    positive_cases, open_cases = [], []
    margin_values = {}
    predictions = [sound_source.CLASSES[int(row.argmax())] for row in classes]
    gallery = [row for row, item in enumerate(items) if item.label in GALLERY_CLASSES]
    for index, item in enumerate(items):
        key = str(item.identifier)
        relevant = {
            other.identifier for other in items if other.identifier != item.identifier and other.label == item.label
        }
        positive_cases.append(evaluation_dataset.EvaluationCase(key, "same sound class", relevant, item.label))
        open_cases.append(
            evaluation_dataset.EvaluationCase(
                key,
                "same sound class in selected gallery",
                relevant if item.label in GALLERY_CLASSES else set(),
                item.label,
            )
        )
        ordered = sorted(
            (row for row in range(len(items)) if row != index),
            key=lambda row: (-float(audio_scores[index, row]), ids[row]),
        )
        full[key] = tuple(ids[row] for row in ordered[:50])
        eligible = sorted(
            (row for row in gallery if row != index), key=lambda row: (-float(audio_scores[index, row]), ids[row])
        )
        raw[key] = tuple(ids[row] for row in eligible[:10])
        gated[key] = raw[key] if predictions[index] in GALLERY_CLASSES else ()
        margin_values[key] = float(classes[index, :3].max() - classes[index, 3:].max())
    threshold_ranks = {
        str(margin): {key: values if margin_values[key] >= margin else () for key, values in gated.items()}
        for margin in margins
    }
    text_ranks, text_cases = {}, []
    for index, label in enumerate(sound_source.CLASSES):
        text_cases.append(
            evaluation_dataset.EvaluationCase(
                label, PROMPTS[index], {item.identifier for item in items if item.label == label}, label
            )
        )
        text_ranks[label] = tuple(
            ids[row] for row in sorted(range(len(items)), key=lambda row: (-float(classes[row, index]), ids[row]))[:10]
        )
    return {
        "partition": partition,
        "clips": len(items),
        "encoder_id": model.encoder_id,
        "audio_encoding_seconds": audio_seconds,
        "text_encoding_seconds": text_seconds,
        "class_accuracy": sum(predicted == item.label for predicted, item in zip(predictions, items)) / len(items),
        "quality": {
            "clap_sound": image_quality._metrics(positive_cases, full),
            "clap_text": image_quality._metrics(text_cases, text_ranks),
        },
        "raw_gallery": image_quality._metrics(open_cases, raw),
        "thresholds": {key: image_quality._metrics(open_cases, rank) for key, rank in threshold_ranks.items()},
        "rankings": {"sound": full, "text": text_ranks, "thresholds": threshold_ranks},
    }


def _acceptable(candidate, raw):
    return (
        candidate["no_answer_false_positive_rate"] <= 0.1
        and candidate["positive"]["hit_at_10"] >= raw["positive"]["hit_at_10"] - 0.1
    )


def select(source: sound_source.SoundDataset, model: clap_onnx.ClapOnnxEncoder) -> reporting.JsonObject:
    """Select at most one development margin, validate it, and freeze the lock."""
    development = measure(source, model, "dev", margins=MARGINS)
    candidates = [
        value for value in MARGINS if _acceptable(development["thresholds"][str(value)], development["raw_gallery"])
    ]
    proposed = min(candidates) if candidates else -1.0
    validation = measure(source, model, "validation", margins=(proposed,))
    accepted = bool(candidates) and _acceptable(validation["thresholds"][str(proposed)], validation["raw_gallery"])
    artifact = {
        "schema": "lentica-sound-selection-v1",
        "dataset_identity": source.identity,
        "encoder_id": model.encoder_id,
        "prompts": list(PROMPTS),
        "gallery_classes": sorted(GALLERY_CLASSES),
        "margin_grid": MARGINS,
        "proposed_margin": proposed,
        "selected_margin": proposed if accepted else -1.0,
        "accepted": accepted,
        "development": development,
        "validation": validation,
        "decision": "Fixed existing class gate fallback if margin validation fails; no default switch, no test selection.",
    }
    artifact["sha256"] = reporting.digest(artifact)
    return artifact


def evaluate(
    source: sound_source.SoundDataset, model: clap_onnx.ClapOnnxEncoder, selection: reporting.JsonObject
) -> reporting.JsonObject:
    """Verify frozen identities before test inference; report all predeclared gates."""
    artifact = dict(selection)
    digest = artifact.pop("sha256")
    if (
        reporting.digest(artifact) != digest
        or artifact["schema"] != "lentica-sound-selection-v1"
        or artifact["dataset_identity"] != source.identity
        or artifact["encoder_id"] != model.encoder_id
        or artifact["prompts"] != list(PROMPTS)
        or artifact["gallery_classes"] != sorted(GALLERY_CLASSES)
        or list(artifact["margin_grid"]) != list(MARGINS)
        or type(artifact["selected_margin"]) not in (int, float)
        or artifact["selected_margin"] not in MARGINS
    ):
        raise ValueError("Frozen sound source/model/config identity differs")
    result = measure(source, model, "test", margins=(artifact["selected_margin"],))
    selected = result["thresholds"][str(artifact["selected_margin"])]
    cost = result["audio_encoding_seconds"] / result["clips"]
    result.update(
        selection_sha256=digest,
        selected_no_answer=selected,
        selection_accepted=artifact["accepted"],
        quality_gate_passed=_acceptable(selected, result["raw_gallery"]),
        cost_gate={"seconds_per_5s_clip": cost, "limit": 0.5, "passed": cost <= 0.5},
    )
    result["heldout_gate_passed"] = result["quality_gate_passed"] and result["cost_gate"]["passed"]
    result["eligible_for_promotion"] = bool(artifact["accepted"]) and result["heldout_gate_passed"]
    result["ranking_sha256"] = reporting.digest(result["rankings"])
    result["scope"] = (
        "Five fixed English class prompts, source-separated ESC50 clips, same-class5s qrels; batched gallery/query features, unknown pretraining overlap; no open-world/word timing/action or production SLO claim."
    )
    return result
