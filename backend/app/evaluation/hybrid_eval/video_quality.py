"""Compose native feature routes, temporal constraints and independent scoring."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from backend.app.retrieval.hybrid import (
    action_model,
    action_retrieval,
    contracts,
    snapshots,
    temporal,
    units,
    vector,
    video_frames,
)

from . import reporting, video_metrics, video_source, video_window_io


class ActionModel(Protocol):
    """Native original/control inference, explicitly constructed by the caller."""

    version: str
    labels: tuple[str, ...]

    def score(
        self, frames: Sequence[video_frames.FrameSample], *, input_kind: str = "original"
    ) -> action_model.ActionScores:
        """Run fresh inference over the supplied real or marked control frames."""
        ...


class ImageEncoder(Protocol):
    """Caller-owned aligned text/image encoder; numerical imports remain lazy."""

    encoder_id: str

    def encode_image(self, path: Path) -> object:
        """Encode one actual middle frame."""
        ...

    def encode_texts(self, texts: Sequence[str]) -> object:
        """Encode the six frozen prompts once; later queries explicitly reuse them."""
        ...


def _actions(
    videos: Sequence[video_window_io.PreparedVideo], model: ActionModel, progress: Callable[[int], None]
) -> tuple[contracts.Retriever | None, reporting.JsonObject]:
    features, rows, controls, errors = {}, [], [], []
    for video in videos:
        for window in video.proposals.windows:
            started = time.perf_counter()
            try:
                # Provider boundary: arbitrary runtime failures stay explicit;
                # the missing window remains in all reference denominators.
                scores = model.score(window.frames)
                if scores.input_kind != "original" or scores.version != model.version or scores.labels != model.labels:
                    raise ValueError("Action provider returned a different original/model/vocabulary identity")
                features[window.unit.unit_id] = scores
                rows.append(
                    {
                        "unit_id": window.unit.unit_id,
                        "logits": list(scores.logits),
                        "seconds": time.perf_counter() - started,
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "unit_id": window.unit.unit_id,
                        "input_kind": "original",
                        "error": type(exc).__name__,
                        "seconds": time.perf_counter() - started,
                    }
                )
        first = video.proposals.windows[0]
        for kind, frames in (
            ("reverse_control", first.frames[::-1]),
            ("static_control", (first.middle_frame,) * 16),
        ):
            started = time.perf_counter()
            try:
                scores = model.score(frames, input_kind=kind)
                if scores.input_kind != kind or scores.version != model.version or scores.labels != model.labels:
                    raise ValueError("Action control provider identity differs")
                controls.append(
                    {
                        "unit_id": first.unit.unit_id,
                        "input_kind": kind,
                        "logits": list(scores.logits),
                        "seconds": time.perf_counter() - started,
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "unit_id": first.unit.unit_id,
                        "input_kind": kind,
                        "error": type(exc).__name__,
                        "seconds": time.perf_counter() - started,
                    }
                )
        progress(video.source.asset_id)
    catalog = units.UnitCatalog([w.unit for v in videos for w in v.proposals.windows])
    retriever = action_retrieval.ActionLabelRetriever(catalog, features) if features else None
    return retriever, {
        "model_version": model.version,
        "labels": list(model.labels),
        "windows": rows,
        "controls": controls,
        "errors": errors,
        "feature_seconds": sum(row["seconds"] for row in rows)
        + sum(row["seconds"] for row in errors if row["input_kind"] == "original"),
        "control_seconds": sum(row["seconds"] for row in controls)
        + sum(row["seconds"] for row in errors if row["input_kind"] != "original"),
    }


def _images(
    videos: Sequence[video_window_io.PreparedVideo], encoder: ImageEncoder, progress: Callable[[int], None]
) -> tuple[contracts.Retriever | None, reporting.JsonObject]:
    import numpy as np

    features, rows, errors = {}, [], []
    for video in videos:
        for window in video.proposals.windows:
            started = time.perf_counter()
            frame = window.middle_frame
            try:
                # Pixel inference is a provider boundary, checked on both sides
                # of the actual read. An error cannot replace the selected frame.
                if snapshots.sha256_file(frame.path) != frame.sha256:
                    raise ValueError("Middle frame changed before inference")
                encoded = np.asarray(encoder.encode_image(frame.path), dtype=np.float32)
                if encoded.shape != (768,) or not np.isfinite(encoded).all() or not np.linalg.norm(encoded):
                    raise ValueError("SigLIP returned invalid middle-frame features")
                if snapshots.sha256_file(frame.path) != frame.sha256:
                    raise ValueError("Middle frame changed during inference")
                features[window.unit.unit_id] = encoded
                rows.append(
                    {
                        "unit_id": window.unit.unit_id,
                        "vector": encoded.tolist(),
                        "seconds": time.perf_counter() - started,
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "unit_id": window.unit.unit_id,
                        "error": type(exc).__name__,
                        "seconds": time.perf_counter() - started,
                    }
                )
        progress(video.source.asset_id)
    prompts = ["a person " + name for name in video_source.LABELS.values()]
    started = time.perf_counter()
    query_features = {}
    try:
        encoded = np.asarray(encoder.encode_texts(prompts), dtype=np.float32)
        if (
            encoded.shape != (6, 768)
            or not np.isfinite(encoded).all()
            or not (np.linalg.norm(encoded, axis=1) > 0).all()
        ):
            raise ValueError("SigLIP returned invalid fixed-prompt features")
        query_features = dict(zip(video_source.LABELS.values(), encoded))
    except Exception as exc:
        errors.append({"query_prompts": prompts, "error": type(exc).__name__})
    query_seconds = time.perf_counter() - started
    metadata = {w.unit.unit_id: w.unit.metadata for v in videos for w in v.proposals.windows}
    retriever = (
        vector.ExactVectorRetriever(
            tuple(features),
            list(features.values()),
            query_features.__getitem__,
            encoder_id=encoder.encoder_id,
            query_encoder_id=encoder.encoder_id,
            metadata=metadata,
        )
        if features and query_features
        else None
    )
    return retriever, {
        "model_version": encoder.encoder_id,
        "windows": rows,
        "query_vectors": {name: value.tolist() for name, value in query_features.items()},
        "query_prompts": prompts,
        "query_encoding_seconds": query_seconds,
        "errors": errors,
        "feature_seconds": sum(row["seconds"] for row in rows)
        + sum(row["seconds"] for row in errors if "unit_id" in row),
    }


def _fetch(source: video_source.VideoDataset, ids: Sequence[int]) -> reporting.JsonObject:
    """Read a single final batch of original containers, never candidate pixels."""
    ids = tuple(dict.fromkeys(ids))
    lookup = {case.asset_id: case for case in source.cases}
    started, total = time.perf_counter(), 0
    for identifier in ids:
        case = lookup[identifier]
        path = source.directory / case.filename
        if path.is_symlink() or path.stat().st_size != case.reference.bytes:
            raise ValueError("Final original video source byte/path identity differs")
        payload = path.read_bytes()
        if len(payload) > video_source.MAX_VIDEO_BYTES or hashlib.sha256(payload).hexdigest() != case.sha256:
            raise ValueError("Final original video source checksum differs")
        total += len(payload)
    return {"calls": int(bool(ids)), "assets": list(ids), "bytes": total, "seconds": time.perf_counter() - started}


def _rank(retriever: contracts.Retriever | None, text: str, filters: contracts.SearchFilter) -> tuple[int, ...]:
    ranked = tuple(retriever.retrieve(contracts.SearchQuery(text, filters), 40)) if retriever is not None else ()
    if len(ranked) > 40 or len(set(ranked)) != len(ranked):
        raise ValueError("Video provider returned duplicate/excess candidates")
    return ranked


def _partition(
    source: video_source.VideoDataset,
    videos: Sequence[video_window_io.PreparedVideo],
    retriever: contracts.Retriever | None,
) -> reporting.JsonObject:
    filters = contracts.SearchFilter(equals={"partition": videos[0].source.reference.partition})
    catalog = {w.unit.unit_id: w.unit for v in videos for w in v.proposals.windows}
    rankings, singles, sequences = {}, [], []
    for code, name in video_source.LABELS.items():
        started, error = time.perf_counter(), None
        try:
            # Extension boundary. Wrong domains/identity fail this fixed query
            # to an empty rank rather than removing it from the denominator.
            ranked = _rank(retriever, name, filters)
            if any(i not in catalog or not filters.matches(i, catalog[i].metadata) for i in ranked):
                raise ValueError("Video provider violated pre-window filters")
        except Exception as exc:
            ranked, error = (), type(exc).__name__
        query_seconds = time.perf_counter() - started
        rankings[code] = ranked
        row = video_metrics.single(videos, code, ranked)
        row.update(
            {
                "code": code,
                "native_query": name,
                "candidate_ids": list(ranked),
                "query_seconds": query_seconds,
                "error": error,
                "final_fetch": _fetch(source, row["resource"]["ranking"]),
            }
        )
        singles.append(row)
    for first in video_source.LABELS:
        for second in video_source.LABELS:
            if first == second:
                continue
            started = time.perf_counter()
            groups = [(code, [catalog[i] for i in rankings[code]]) for code in (first, second)]
            matches = temporal.sequence(groups, filters, min_gap_seconds=0.5)
            ranks = [
                {identifier: rank for rank, identifier in enumerate(rankings[code], 1)} for code in (first, second)
            ]
            ordered = sorted(
                matches,
                key=lambda match: (
                    sum(r[u.unit_id] for r, u in zip(ranks, match.evidence)),
                    match.asset_id,
                    tuple(u.unit_id for u in match.evidence),
                ),
            )[:5]
            elapsed = time.perf_counter() - started
            row = video_metrics.ordered(videos, first, second, ordered)
            row.update(
                {
                    "first": first,
                    "second": second,
                    "combination_seconds": elapsed,
                    "query_error": any(r["error"] is not None for r in singles if r["code"] in (first, second)),
                    "reference_oracle": video_metrics.oracle(videos, first, second),
                    "final_fetch": _fetch(source, [match.asset_id for match in ordered]),
                }
            )
            sequences.append(row)
    wrong = _rank(
        retriever, next(iter(video_source.LABELS.values())), contracts.SearchFilter(equals={"partition": "absent"})
    )
    if wrong:
        raise ValueError("Wrong-filter video control returned candidates")
    return {"single": singles, "sequence": sequences, "summary": video_metrics.summarize(singles, sequences)}


def evaluate(
    source: video_source.VideoDataset,
    videos: Sequence[video_window_io.PreparedVideo],
    *,
    route: str,
    action: ActionModel | None = None,
    image: ImageEncoder | None = None,
    progress: Callable[[int], None] = lambda identifier: None,
) -> reporting.JsonObject:
    """Run fresh feature inference and all108 fixed queries; reuse features explicitly.

    Neither source references nor descriptions enter an encoder/retriever. Each
    partition is filtered before Top40, sequence evidence preserves native times,
    and final original-video reads occur once per nonempty final query batch.
    Provider failures retain missing windows/queries in reference denominators.
    """
    if route not in {"videomae", "siglip"} or (action is None) == (image is None):
        raise ValueError("Choose exactly one explicit native video or single-frame provider")
    if tuple(v.source for v in videos) != source.cases:
        raise ValueError("Video proposal sources/order differ from frozen source snapshot")
    if route == "videomae" and action is not None:
        retriever, features = _actions(videos, action, progress)
    elif route == "siglip" and image is not None:
        retriever, features = _images(videos, image, progress)
    else:
        raise ValueError("Route and selected model provider differ")
    partitions = {
        name: _partition(source, [v for v in videos if v.source.reference.partition == name], retriever)
        for name in ("dev", "validation", "test")
    }
    clipping = [
        {
            "asset_id": v.source.asset_id,
            "label": e.label,
            "raw": [e.start, e.end],
            "clipped": [min(e.start, v.proposals.duration_seconds), min(e.end, v.proposals.duration_seconds)],
            "removed_seconds": e.end - max(e.start, v.proposals.duration_seconds),
        }
        for v in videos
        for e in v.source.reference.events
        if e.end > v.proposals.duration_seconds
    ]
    numeric = [
        {key: value for key, value in row.items() if key != "seconds"}
        for row in features["windows"] + features.get("controls", [])
    ]
    ranked = {
        name: {
            "single": [row["candidate_ids"] for row in values["single"]],
            "sequence": [[r["unit_ids"] for r in row["ranking"]] for row in values["sequence"]],
        }
        for name, values in partitions.items()
    }
    return {
        "schema": "lentica-video-quality-v1",
        "dataset_identity": source.identity,
        "selection_identity": source.selection.identity,
        "proposal_versions": [v.proposals.version for v in videos],
        "route": route,
        "features": features,
        "numeric_sha256": reporting.digest({"windows_controls": numeric, "queries": features.get("query_vectors", {})}),
        "ranking_sha256": reporting.digest(ranked),
        "partitions": partitions,
        "clipped_references": clipping,
        "query_count": 108,
        "scope": "Frozen6-class/16-source external-reference control; cached features/prompts, no open-world rejection or temporal-language understanding/default change.",
    }


def compare(native: Mapping, single_frame: Mapping) -> reporting.JsonObject:
    """Apply the predeclared held-out gates; failed validation cannot be repaired by test."""
    keys = ("schema", "dataset_identity", "selection_identity", "proposal_versions", "query_count")
    if any(native[key] != single_frame[key] for key in keys) or (native["route"], single_frame["route"]) != (
        "videomae",
        "siglip",
    ):
        raise ValueError("Video comparison requires the same frozen sources/proposals and native/single-frame routes")
    gates = {}
    for part in ("validation", "test"):
        original = native["partitions"][part]["summary"]
        baseline = single_frame["partitions"][part]["summary"]
        gates[part] = {
            "unit_ndcg_at_10": original["unit_ndcg_at_10"],
            "unit_ndcg_delta": original["unit_ndcg_at_10"] - baseline["unit_ndcg_at_10"],
            "sequence_positive_hit_at_5": original["sequence_positive_hit_at_5"],
            "passed": original["unit_ndcg_at_10"] >= 0.5
            and original["unit_ndcg_at_10"] - baseline["unit_ndcg_at_10"] >= 0.05
            and original["sequence_positive_hit_at_5"] >= 0.5,
        }
    return {
        "dataset_identity": native["dataset_identity"],
        "gates": gates,
        "accepted": all(row["passed"] for row in gates.values())
        and not native["features"]["errors"]
        and not single_frame["features"]["errors"],
        "scope": "Small fixed-vocabulary quality gate only; no calibrated no-answer/open-world/serving acceptance.",
    }
