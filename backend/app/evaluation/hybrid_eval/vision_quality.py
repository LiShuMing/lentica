"""Fixed same-source CLIP/SigLIP controls and actual bounded image reranking."""

from __future__ import annotations

import time

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation.hybrid_eval import image_quality, mirror_images, reporting
from backend.app.retrieval.hybrid import contracts, units, vision_reranking


def compare(
    source: mirror_images.ImageDataset,
    baseline: vision_reranking.ImageTextEncoder,
    candidate: vision_reranking.ImageTextEncoder,
) -> reporting.JsonObject:
    """Compare predeclared test controls; never select weights or thresholds here.

    Both full-gallery encoders actually run. Each text-query reranking separately
    re-encodes its twenty candidate pixels and query without a feature cache.
    Cached whole-gallery features are only a separate diagnostic, since dynamic
    quantization and batching can change features and rankings slightly.
    """
    items = source.partition("test")
    paths = {item.identifier: source.directory / item.filename for item in items}
    ids = [item.identifier for item in items]
    observed, image_matrices, text_matrices = _encode_gallery(paths, baseline, candidate)
    image_cases, text_cases, image_ranks, text_ranks = _rank_gallery(items, ids, image_matrices, text_matrices)
    reranking = _actual_reranking(
        candidate, paths, text_cases, text_ranks["clip"], image_matrices["siglip"], ids, text_matrices["siglip"]
    )
    all_text = {**text_ranks, "clip_siglip_top20": reranking["rankings"]}
    return {
        "schema": "lentica-vision-comparison-v1",
        "dataset_identity": source.identity,
        "partition": "test",
        "prompts": list(image_quality.PROMPTS),
        "full_gallery_cost": observed,
        "quality_image": {name: image_quality._metrics(image_cases, ranks) for name, ranks in image_ranks.items()},
        "quality_text": {name: image_quality._metrics(text_cases, ranks) for name, ranks in all_text.items()},
        "candidate_coverage_text": {
            str(limit): sum(
                len(set(ranks[:limit]) & case.relevant_asset_ids) / len(case.relevant_asset_ids)
                for case in text_cases
                for ranks in [text_ranks["clip"][case.query_id]]
            )
            / len(text_cases)
            for limit in (20, 50)
        },
        "paired_text": {
            name: reporting.paired_comparison(text_cases, text_ranks["clip"], ranks)
            for name, ranks in all_text.items()
            if name != "clip"
        },
        "reranking": reranking,
        "quality_text_reused_diagnostic": image_quality._metrics(text_cases, reranking["reused_feature_rankings"]),
        "rankings": {"image": image_ranks, "text": all_text},
        "ranking_sha256": reporting.digest({"image": image_ranks, "text": all_text}),
        "scope": "Fixed ten class prompts and dominant-class qrels on a third disjoint source set; small sample, unknown pretraining/near duplicates; no model selection, Chinese, relation, region, or open-world refusal claim.",
    }


def _encode_gallery(paths, baseline, candidate):
    """Actually encode the selected complete gallery for each fixed model."""
    observed, image_matrices, text_matrices = {}, {}, {}
    for name, encoder in (("clip", baseline), ("siglip", candidate)):
        start = time.perf_counter()
        image_matrices[name] = encoder.encode_images(list(paths.values()))
        image_seconds = time.perf_counter() - start
        start = time.perf_counter()
        text_matrices[name] = encoder.encode_texts(image_quality.PROMPTS)
        observed[name] = {
            "encoder_id": encoder.encoder_id,
            "actual_image_encodings": len(paths),
            "actual_text_encodings": len(image_quality.PROMPTS),
            "image_seconds": image_seconds,
            "text_seconds": time.perf_counter() - start,
        }
    return observed, image_matrices, text_matrices


def _rank_gallery(items, ids, image_matrices, text_matrices):
    """Build same-class qrels after inference and exclude image self matches."""
    image_ranks, text_ranks = {"clip": {}, "siglip": {}}, {"clip": {}, "siglip": {}}
    image_cases, text_cases = [], []
    for index, item in enumerate(items):
        case = evaluation_dataset.EvaluationCase(
            str(item.identifier),
            "same dominant object class",
            {other.identifier for other in items if other.label == item.label and other.identifier != item.identifier},
            item.label,
        )
        image_cases.append(case)
        for name, images in image_matrices.items():
            scores = images @ images[index]
            ordered = sorted(
                (row for row in range(len(ids)) if row != index), key=lambda row: (-float(scores[row]), ids[row])
            )
            image_ranks[name][case.query_id] = tuple(ids[row] for row in ordered[:50])
    for index, label in enumerate(mirror_images.CLASSES):
        text_cases.append(
            evaluation_dataset.EvaluationCase(
                label, image_quality.PROMPTS[index], {item.identifier for item in items if item.label == label}, label
            )
        )
        for name, images in image_matrices.items():
            scores = images @ text_matrices[name][index]
            text_ranks[name][label] = tuple(
                ids[row] for row in sorted(range(len(ids)), key=lambda row: (-float(scores[row]), ids[row]))[:50]
            )
    return image_cases, text_cases, image_ranks, text_ranks


def _actual_reranking(encoder, paths, cases, baseline, features, ids, texts) -> reporting.JsonObject:
    """Measure independent inference and compare it with reused batch features."""
    positions = {identifier: index for index, identifier in enumerate(ids)}
    reranker = vision_reranking.TextImageReranker(encoder, paths, max_candidates=20)
    ranks, reused_ranks, rows = {}, {}, []
    for index, case in enumerate(cases):
        prefix = baseline[case.query_id][:20]
        start = time.perf_counter()
        actual = reranker.rerank(
            contracts.SearchQuery(case.query, contracts.SearchFilter(allowed_ids=frozenset(ids))),
            [units.RetrievalUnit(identifier, identifier, "image") for identifier in prefix],
        )
        seconds = time.perf_counter() - start
        reused = vision_reranking.order_candidates(
            prefix, features[[positions[identifier] for identifier in prefix]] @ texts[index]
        )
        ranks[case.query_id] = actual + tuple(baseline[case.query_id][20:])
        reused_ranks[case.query_id] = reused + tuple(baseline[case.query_id][20:])
        rows.append(
            {
                "query_id": case.query_id,
                "seconds": seconds,
                "candidate_count": len(prefix),
                "reused_feature_ranking_matches": actual == reused,
                "reused_feature_top10_matches": actual[:10] == reused[:10],
            }
        )
    return {
        "rankings": ranks,
        "reused_feature_rankings": reused_ranks,
        "window": 20,
        "candidate_window": 50,
        "method": "independent native text/image cosine; no cross-attention",
        "actual_image_encodings": sum(row["candidate_count"] for row in rows),
        "actual_text_encodings": len(cases),
        "reused_candidate_encodings": 0,
        "seconds_total": sum(row["seconds"] for row in rows),
        "query_p50_ms": reporting.percentile([row["seconds"] for row in rows], 0.5),
        "query_p95_ms": reporting.percentile([row["seconds"] for row in rows], 0.95),
        "rows": rows,
    }
