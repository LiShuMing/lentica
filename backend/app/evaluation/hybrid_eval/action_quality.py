"""Fixed temporal-model controls and existing unit/resource pipeline composition."""

from __future__ import annotations

import time
from collections.abc import Sequence

from backend.app.evaluation.hybrid_eval import reporting
from backend.app.retrieval.hybrid import (
    action_model,
    action_retrieval,
    contracts,
    documents,
    stages,
    units,
    video_frames,
)


def _retrieval_control(
    model: action_model.VideoMaeClassifier, logits: tuple[float, ...], frames: Sequence[video_frames.FrameSample]
) -> reporting.JsonObject:
    """Reuse actual original class features; hydrate only the selected parent."""
    unit = units.RetrievalUnit(
        1000,
        1,
        "video",
        units.Locator(
            start_seconds=frames[0].unit.locator.start_seconds, end_seconds=frames[-1].unit.locator.end_seconds
        ),
        metadata={"source": "public-demo"},
    )
    catalog = units.UnitCatalog([unit])
    features = action_model.ActionScores(model.labels, logits, model.version, "original")
    retriever = action_retrieval.ActionLabelRetriever(catalog, {1000: features})
    store = reporting.CountingStore([documents.CorpusDocument(1, "Public action demonstration")])
    searcher = stages.StagedSearch({"action": retriever}, catalog, store=store)
    config = stages.StageConfig(
        contracts.SearchConfig((contracts.Branch("action", 4),), top_k=1), candidate_window=4, rerank_window=4
    )
    result = searcher.search(
        contracts.SearchQuery("eating spaghetti", contracts.SearchFilter(equals={"source": "public-demo"})), config
    )
    wrong_filter = searcher.search(
        contracts.SearchQuery("eating spaghetti", contracts.SearchFilter(equals={"source": "other"})), config
    )
    return {
        "asset_ids": [hit.asset_id for hit in result.hits],
        "unit_id": unit.unit_id,
        "start_seconds": unit.locator.start_seconds,
        "end_seconds": unit.locator.end_seconds,
        "final_fetch_calls": store.calls,
        "final_fetch_rows": store.rows,
        "wrong_filter_assets": [hit.asset_id for hit in wrong_filter.hits],
        "scope": "Reuse just-computed original class features; exact native class query, whole-clip locator, one real candidate/resource, no free-text or broad quality proof.",
    }


def evaluate(sampled: video_frames.VideoFrames, model: action_model.VideoMaeClassifier) -> reporting.JsonObject:
    """Uniformly choose16 already decoded frames and run fixed model controls.

    The expected whole-clip class was declared before inference. This one source
    name is weak annotation, not independent temporal/event-boundary labeling.
    """
    if len(sampled.samples) < 16:
        raise ValueError("Action demonstration requires at least16 sampled frames")
    indices = [index * (len(sampled.samples) - 1) // 15 for index in range(16)]
    frames = tuple(sampled.samples[index] for index in indices)
    expected = model.labels.index("eating spaghetti")
    rows, scores = {}, {}
    for kind, values in (
        ("original", frames),
        ("reverse_control", frames[::-1]),
        ("static_control", (frames[len(frames) // 2],) * 16),
    ):
        started = time.perf_counter()
        predicted = model.score(values, input_kind=kind)
        elapsed = time.perf_counter() - started
        ordered = predicted.ranking()
        rows[kind] = {
            "top5": [{"label": predicted.labels[index], "logit": predicted.logits[index]} for index in ordered[:5]],
            "expected_rank": ordered.index(expected) + 1,
            "top1_hit": ordered[0] == expected,
            "top5_hit": expected in ordered[:5],
            "inference_seconds": elapsed,
            "source_frame_indices": [frame.frame_index for frame in values],
            "logit_sha256": reporting.digest(predicted.logits),
        }
        scores[kind] = predicted.logits
    return {
        "schema": "lentica-action-control-v1",
        "source_sha256": sampled.source_sha256,
        "sampling_version": sampled.version,
        "model_version": model.version,
        "provided_clip_label": "eating spaghetti",
        "sample_start_seconds": frames[0].unit.locator.start_seconds,
        "sample_end_seconds": frames[-1].unit.locator.end_seconds,
        "results": rows,
        "control_max_abs_logit_delta": {
            kind: max(abs(left - right) for left, right in zip(scores["original"], values))
            for kind, values in scores.items()
            if kind != "original"
        },
        "retrieval_control": _retrieval_control(model, scores["original"], frames),
        "scope": "Single public source-named whole-clip class, not independent annotated quality/generalization/event boundaries; reverse/static are diagnostics only; no default switch.",
    }
