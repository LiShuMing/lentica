"""Source-isolated native image/text retrieval and bounded no-answer controls."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation import metrics
from backend.app.evaluation.hybrid_eval import mirror_images, reporting
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import adapters, media, units, vision_reranking

THRESHOLDS = (-1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MARGINS = (-1.0, 0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 1.0)
GALLERY_CLASSES = frozenset(mirror_images.CLASSES[:6])
PROMPTS = tuple(f"a photo of a {label}." for label in mirror_images.CLASSES)


def _metrics(
    cases: Sequence[evaluation_dataset.EvaluationCase], ranks: Mapping[str, Sequence[int]]
) -> reporting.JsonObject:
    positive = [case for case in cases if case.relevant_asset_ids]
    negative = [case for case in cases if not case.relevant_asset_ids]
    rows = [metrics.metrics_for_case(case, list(ranks[case.query_id])[:10]) for case in positive]
    return {
        "positive_queries": len(positive),
        "no_answer_queries": len(negative),
        "positive": {
            key: sum(row[key] for row in rows) / len(rows) if rows else 0.0
            for key in ("recall_at_10", "hit_at_10", "ndcg_at_10", "mrr")
        },
        "precision_at_10": sum(len(set(ranks[case.query_id][:10]) & case.relevant_asset_ids) / 10 for case in positive)
        / len(positive)
        if positive
        else 0.0,
        "no_answer_false_positive_rate": sum(bool(ranks[case.query_id]) for case in negative) / len(negative)
        if negative
        else None,
    }


def _image_queries(items, image_scores, predictions):
    """Form same-source-partition qrels and self-excluding fixed gallery ranks."""
    ids = [item.identifier for item in items]
    gallery = [index for index, item in enumerate(items) if item.label in GALLERY_CLASSES]
    positive_cases, open_cases = [], []
    full_ranks, raw_open, gated_open, top_scores = {}, {}, {}, {}
    for index, item in enumerate(items):
        key = str(item.identifier)
        relevant = {
            other.identifier for other in items if other.label == item.label and other.identifier != item.identifier
        }
        positive_cases.append(
            evaluation_dataset.EvaluationCase(key, "same dominant object class", relevant, item.label)
        )
        open_cases.append(
            evaluation_dataset.EvaluationCase(
                key,
                "same dominant object class within the fixed gallery",
                relevant if item.label in GALLERY_CLASSES else set(),
                item.label,
            )
        )
        eligible = [row for row in gallery if row != index]
        ranked = sorted(eligible, key=lambda row: (-float(image_scores[index, row]), ids[row]))
        raw_open[key] = tuple(ids[row] for row in ranked[:10])
        gated_open[key] = raw_open[key] if predictions[index] in GALLERY_CLASSES else ()
        top_scores[key] = float(image_scores[index, ranked[0]]) if ranked else -1.0
        full = sorted(
            (row for row in range(len(items)) if row != index),
            key=lambda row: (-float(image_scores[index, row]), ids[row]),
        )
        full_ranks[key] = tuple(ids[row] for row in full[:50])
    return positive_cases, open_cases, full_ranks, raw_open, gated_open, top_scores


def _controls(source, items, positive_cases, full_ranks, class_scores, *, encoder_name="clip"):
    """Compare actual HSV and fixed RRF without selecting a test winner."""
    ids = [item.identifier for item in items]
    paths = [source.directory / item.filename for item in items]
    result = {}
    catalog = units.UnitCatalog([units.RetrievalUnit(item.identifier, item.identifier, "image") for item in items])
    pixel = media.ColorHistogramRetriever(catalog, dict(zip(ids, paths)))
    color_ranks, hybrid_ranks = {}, {}
    config = hybrid.SearchConfig((hybrid.Branch(encoder_name, 50), hybrid.Branch("color", 50)))
    for item, path in zip(items, paths):
        key = str(item.identifier)
        color_ranks[key] = pixel.retrieve_image(
            path, hybrid.SearchFilter(allowed_ids=frozenset(ids) - {item.identifier}), 50
        )
        fused = hybrid.HybridSearch(
            {
                encoder_name: adapters.RankedListRetriever(full_ranks[key]),
                "color": adapters.RankedListRetriever(color_ranks[key]),
            }
        ).search(hybrid.SearchQuery("same dominant object class"), config)
        hybrid_ranks[key] = tuple(hit.asset_id for hit in fused.hits)
    text_ranks, text_cases = {}, []
    for index, label in enumerate(mirror_images.CLASSES):
        text_cases.append(
            evaluation_dataset.EvaluationCase(
                label, PROMPTS[index], {item.identifier for item in items if item.label == label}, label
            )
        )
        text_ranks[label] = tuple(
            ids[row]
            for row in sorted(range(len(items)), key=lambda row: (-float(class_scores[row, index]), ids[row]))[:10]
        )
    result["quality"] = {
        f"{encoder_name}_image": _metrics(positive_cases, full_ranks),
        "pixel_image": _metrics(positive_cases, color_ranks),
        "rrf_image": _metrics(positive_cases, hybrid_ranks),
        f"{encoder_name}_text": _metrics(text_cases, text_ranks),
    }
    result["control_rankings"] = {
        f"{encoder_name}_image": full_ranks,
        "pixel_image": color_ranks,
        "rrf_image": hybrid_ranks,
        f"{encoder_name}_text": text_ranks,
    }
    return result


def measure(
    source: mirror_images.ImageDataset,
    model: vision_reranking.ImageTextEncoder,
    partition: str,
    *,
    include_controls: bool,
    thresholds: Sequence[float] = THRESHOLDS,
    threshold_kind: str = "image_similarity",
) -> reporting.JsonObject:
    """Run actual pixels/text; labels are used only for qrels and candidate classes.

    Gallery/query image sources are local to one partition. Self IDs are excluded
    from every image-query route. This object-class task is not caption, region,
    relation or temporal understanding; a category gate is not general rejection.
    """
    items = source.partition(partition)
    if len(items) < 20:
        raise ValueError("Image quality requires enough independent partition images")
    paths = [source.directory / item.filename for item in items]
    started = time.perf_counter()
    images = model.encode_images(paths)
    image_seconds = time.perf_counter() - started
    started = time.perf_counter()
    texts = model.encode_texts(PROMPTS)
    text_seconds = time.perf_counter() - started
    image_scores, class_scores = images @ images.T, images @ texts.T
    predictions = [mirror_images.CLASSES[int(row.argmax())] for row in class_scores]
    positive_cases, open_cases, full_ranks, raw_open, gated_open, top_scores = _image_queries(
        items, image_scores, predictions
    )
    if threshold_kind == "class_margin":
        top_scores = {
            str(item.identifier): float(class_scores[index, :6].max() - class_scores[index, 6:].max())
            for index, item in enumerate(items)
        }
    elif threshold_kind != "image_similarity":
        raise ValueError("Unknown frozen image threshold semantics")
    threshold_metrics = {}
    threshold_ranks = {}
    for threshold in thresholds:
        ranks = {key: values if top_scores[key] >= threshold else () for key, values in gated_open.items()}
        threshold_metrics[str(threshold)] = _metrics(open_cases, ranks)
        threshold_ranks[str(threshold)] = ranks
    result = {
        "partition": partition,
        "threshold_kind": threshold_kind,
        "images": len(items),
        "encoder_id": model.encoder_id,
        "image_encoding_seconds": image_seconds,
        "text_encoding_seconds": text_seconds,
        "image_to_label_accuracy": sum(predicted == item.label for predicted, item in zip(predictions, items))
        / len(items),
        "open_gallery": {
            "classes": sorted(GALLERY_CLASSES),
            "raw": _metrics(open_cases, raw_open),
            "category_gate": _metrics(open_cases, gated_open),
            "thresholds": threshold_metrics,
        },
        "threshold_rankings": threshold_ranks,
        "raw_open_rankings": raw_open,
        "category_predictions": {str(item.identifier): prediction for item, prediction in zip(items, predictions)},
    }
    if include_controls:
        result.update(
            _controls(
                source, items, positive_cases, full_ranks, class_scores, encoder_name=getattr(model, "kind", "clip")
            )
        )
    return result


def _acceptable(candidate: reporting.JsonObject, raw: reporting.JsonObject, *, fpr_limit: float = 0.1) -> bool:
    return (
        candidate["no_answer_false_positive_rate"] <= fpr_limit
        and candidate["positive"]["hit_at_10"] >= raw["positive"]["hit_at_10"] - 0.1
    )


def select(
    source: mirror_images.ImageDataset, model: vision_reranking.ImageTextEncoder, *, protocol: str = "similarity-v1"
) -> reporting.JsonObject:
    """Use only development/validation images; never measure the test partition."""
    if protocol not in {"similarity-v1", "margin-v2"}:
        raise ValueError("Unknown image selection protocol")
    grid = MARGINS if protocol == "margin-v2" else THRESHOLDS
    kind = "class_margin" if protocol == "margin-v2" else "image_similarity"
    fpr_limit = 0.05 if protocol == "margin-v2" else 0.1
    dev = measure(source, model, "dev", include_controls=False, thresholds=grid, threshold_kind=kind)
    candidates = [
        threshold
        for threshold in grid
        if _acceptable(
            dev["open_gallery"]["thresholds"][str(threshold)], dev["open_gallery"]["raw"], fpr_limit=fpr_limit
        )
    ]
    proposed = min(candidates) if candidates else -1.0
    validation = measure(
        source, model, "validation", include_controls=False, thresholds=(proposed,), threshold_kind=kind
    )
    accepted = bool(candidates) and _acceptable(
        validation["open_gallery"]["thresholds"][str(proposed)], validation["open_gallery"]["raw"], fpr_limit=fpr_limit
    )
    artifact = {
        "schema": "lentica-image-selection-v2" if protocol == "margin-v2" else "lentica-image-selection-v1",
        "protocol": protocol,
        "dataset_identity": source.identity,
        "encoder_id": model.encoder_id,
        "prompts": list(PROMPTS),
        "gallery_classes": sorted(GALLERY_CLASSES),
        "threshold_grid": grid,
        "development_validation_fpr_limit": fpr_limit,
        "proposed_threshold": proposed,
        "accepted": accepted,
        "selected_threshold": proposed if accepted else -1.0,
        "development": dev,
        "validation": validation,
        "decision": "Keep the fixed category gate when threshold validation fails; no product default switch. Same class prompts across partitions, no unseen-query claim.",
    }
    artifact["sha256"] = reporting.digest(artifact)
    return artifact


def evaluate(
    source: mirror_images.ImageDataset, model: vision_reranking.ImageTextEncoder, selection: reporting.JsonObject
) -> reporting.JsonObject:
    """Check a frozen selection first, then run one held-out source partition."""
    artifact = dict(selection)
    fingerprint = artifact.pop("sha256")
    protocol = artifact.get("protocol", "similarity-v1")
    grid = MARGINS if protocol == "margin-v2" else THRESHOLDS
    kind = "class_margin" if protocol == "margin-v2" else "image_similarity"
    if (
        reporting.digest(artifact) != fingerprint
        or protocol not in {"similarity-v1", "margin-v2"}
        or artifact["schema"]
        != ("lentica-image-selection-v2" if protocol == "margin-v2" else "lentica-image-selection-v1")
        or artifact["dataset_identity"] != source.identity
        or artifact["encoder_id"] != model.encoder_id
        or artifact["prompts"] != list(PROMPTS)
        or artifact["gallery_classes"] != sorted(GALLERY_CLASSES)
        or list(artifact["threshold_grid"]) != list(grid)
        or type(artifact["selected_threshold"]) not in (int, float)
        or artifact["selected_threshold"] not in grid
    ):
        raise ValueError("Frozen public image selection identity/config differs")
    result = measure(
        source, model, "test", include_controls=True, thresholds=(artifact["selected_threshold"],), threshold_kind=kind
    )
    result["selected_no_answer"] = result["open_gallery"]["thresholds"][str(artifact["selected_threshold"])]
    result["heldout_gate_passed"] = _acceptable(result["selected_no_answer"], result["open_gallery"]["raw"])
    if protocol == "margin-v2":
        seconds_per_image = result["image_encoding_seconds"] / result["images"]
        result["batch_encoding_cost_gate"] = {
            "seconds_per_image": seconds_per_image,
            "limit": 0.2,
            "passed": seconds_per_image <= 0.2,
        }
        result["heldout_gate_passed"] = result["heldout_gate_passed"] and result["batch_encoding_cost_gate"]["passed"]
    result["additional_similarity_cutoff_enabled"] = artifact["selected_threshold"] != -1.0
    result["selection_accepted"] = bool(artifact["accepted"])
    result["eligible_for_promotion"] = result["selection_accepted"] and result["heldout_gate_passed"]
    result["selection_sha256"] = fingerprint
    result["ranking_sha256"] = reporting.digest(
        {
            "controls": result["control_rankings"],
            "selected_open": result["threshold_rankings"][str(artifact["selected_threshold"])],
        }
    )
    result["scope"] = (
        f"Fixed pretrained English {getattr(model, 'kind', 'clip')} on new image sources and dominant-class qrels, 32x32 inputs; pretraining/near-duplicate overlap unknown; no captions, spatial relations, Chinese, or temporal claim."
    )
    return result
