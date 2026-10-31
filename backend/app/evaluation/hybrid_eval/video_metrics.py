"""Independent reference timing/ranking scores; no labels enter model inputs."""

from __future__ import annotations

import math
from collections.abc import Sequence

from backend.app.retrieval.hybrid import contracts, temporal, units

from . import reporting, video_source, video_window_io


def overlap(start: float, end: float, event: video_source.VideoEvent, duration: float) -> float:
    """Measure intersection against the observable part of an unchanged reference."""
    return max(0.0, min(end, event.end, duration) - max(start, event.start))


def iou(start: float, end: float, event: video_source.VideoEvent, duration: float | None = None) -> float:
    """Return raw or clipped interval intersection-over-union, zero if unobservable."""
    reference_end = event.end if duration is None else min(duration, event.end)
    intersection = max(0.0, min(end, reference_end) - max(start, event.start))
    union = end - start + max(0.0, reference_end - event.start) - intersection
    return intersection / union if union else 0.0


def _coverage(unit: units.RetrievalUnit, events: Sequence[video_source.VideoEvent], duration: float) -> float:
    """Union event intersections, avoiding duplicate/overlapping annotation votes."""
    start, end = unit.locator.start_seconds, unit.locator.end_seconds
    intervals = sorted((max(start, event.start), min(end, event.end, duration)) for event in events)
    total, previous = 0.0, start
    for left, right in intervals:
        if right > max(left, previous):
            total += right - max(left, previous)
            previous = right
    return total / (end - start)


def ranking_scores(ranked: Sequence[int], relevant: set[int]) -> reporting.JsonObject:
    """Binary Recall/nDCG with empty qrels scored zero and separately identified."""
    if len(set(ranked)) != len(ranked):
        raise ValueError("Video ranking must contain unique IDs")
    top = ranked[:10]
    dcg = sum(1 / math.log2(index + 2) for index, identifier in enumerate(top) if identifier in relevant)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(10, len(relevant))))
    return {
        "relevant": sorted(relevant),
        "ranking": list(top),
        "recall_at_10": len(set(top) & relevant) / len(relevant) if relevant else 0.0,
        "ndcg_at_10": dcg / ideal if ideal else 0.0,
        "no_answer": not relevant,
        "no_answer_returned": not relevant and bool(ranked),
    }


def single(videos: Sequence[video_window_io.PreparedVideo], code: str, ranked: Sequence[int]) -> reporting.JsonObject:
    """Score all references, including events missed by every blind proposal."""
    lookup = {w.unit.unit_id: (v, w.unit) for v in videos for w in v.proposals.windows}
    if any(identifier not in lookup for identifier in ranked):
        raise ValueError("Video ranking escapes the selected reference partition")
    relevant_units, relevant_assets, event_rows = set(), set(), []
    for video in videos:
        events = [e for e in video.source.reference.events if e.label == code]
        if events:
            relevant_assets.add(video.source.asset_id)
        duration = video.proposals.duration_seconds
        for window in video.proposals.windows:
            if _coverage(window.unit, events, duration) >= 0.5:
                relevant_units.add(window.unit.unit_id)
        for event in events:
            selected = [lookup[i][1] for i in ranked[:10] if lookup[i][0].source.asset_id == video.source.asset_id]
            best = {
                kind: max(
                    (iou(u.locator.start_seconds, u.locator.end_seconds, event, limit) for u in selected), default=0.0
                )
                for kind, limit in (("raw_iou", None), ("clipped_iou", duration))
            }
            event_rows.append({"asset_id": video.source.asset_id, "start": event.start, "end": event.end, **best})
    resources = list(dict.fromkeys(lookup[i][1].asset_id for i in ranked))
    return {
        "unit": ranking_scores(ranked, relevant_units),
        "resource": ranking_scores(resources, relevant_assets),
        "events": event_rows,
        "event_recall_iou_03_at_10": sum(r["clipped_iou"] >= 0.3 for r in event_rows) / len(event_rows)
        if event_rows
        else 0.0,
        "event_recall_iou_05_at_10": sum(r["clipped_iou"] >= 0.5 for r in event_rows) / len(event_rows)
        if event_rows
        else 0.0,
    }


def reference_pairs(
    video: video_window_io.PreparedVideo, first: str, second: str
) -> tuple[tuple[video_source.VideoEvent, video_source.VideoEvent], ...]:
    """Keep raw reference order/gap, including entirely unobservable events."""
    return tuple(
        (a, b)
        for a in video.source.reference.events
        for b in video.source.reference.events
        if a.label == first and b.label == second and a.end + 0.5 <= b.start
    )


def oracle(videos: Sequence[video_window_io.PreparedVideo], first: str, second: str) -> reporting.JsonObject:
    """Join only observable human event intervals; diagnostic, never model input.

    This is a separate reference-support ceiling, not perfect retrieval over the
    fixed proposals. Unobservable raw gold pairs remain in its denominator.
    """
    groups = []
    for code in (first, second):
        candidates = []
        for video in videos:
            for index, event in enumerate(video.source.reference.events):
                end = min(event.end, video.proposals.duration_seconds)
                if event.label == code and event.start < end:
                    candidates.append(
                        units.RetrievalUnit(
                            100000 + video.source.asset_id * 1000 + index,
                            video.source.asset_id,
                            "video",
                            units.Locator(start_seconds=event.start, end_seconds=end),
                        )
                    )
        groups.append((code, candidates))
    matches = temporal.sequence(groups, contracts.SearchFilter(), min_gap_seconds=0.5)
    gold = {v.source.asset_id for v in videos if reference_pairs(v, first, second)}
    found = {match.asset_id for match in matches}
    return {
        "raw_gold_assets": sorted(gold),
        "observable_join_assets": sorted(found),
        "raw_gold_asset_recall": len(gold & found) / len(gold) if gold else 0.0,
        "reference_support_matches": len(matches),
        "scope": "Human intervals only, not blind proposals or model predictions.",
    }


def ordered(
    videos: Sequence[video_window_io.PreparedVideo],
    first: str,
    second: str,
    matches: Sequence[temporal.TemporalSequenceMatch],
) -> reporting.JsonObject:
    """Score resource order and each step's observable localization separately.

    Primary hit requires >=.5 of each proposal covered by its respective event.
    An independent IoU>=.3/.5 diagnostic prevents calling coverage exact timing.
    """
    gold = {v.source.asset_id: (v, reference_pairs(v, first, second)) for v in videos}
    relevant = {identifier for identifier, (_, pairs) in gold.items() if pairs}
    rows = []
    for match in matches[:5]:
        if match.asset_id not in gold or len(match.evidence) != 2:
            raise ValueError("Video sequence ranking escapes selected sources/steps")
        video, pairs = gold[match.asset_id]
        duration = video.proposals.duration_seconds
        supports = match.evidence
        coverage, localization, raw_localization = [], [], []
        for pair in pairs:
            coverage.append(min(_coverage(u, (event,), duration) for u, event in zip(supports, pair)))
            localization.append(
                min(
                    iou(u.locator.start_seconds, u.locator.end_seconds, event, duration)
                    for u, event in zip(supports, pair)
                )
            )
            raw_localization.append(
                min(iou(u.locator.start_seconds, u.locator.end_seconds, event) for u, event in zip(supports, pair))
            )
        rows.append(
            {
                "asset_id": match.asset_id,
                "unit_ids": [u.unit_id for u in supports],
                "step_intervals": [[u.locator.start_seconds, u.locator.end_seconds] for u in supports],
                "minimum_step_coverage": max(coverage, default=0.0),
                "minimum_step_clipped_iou": max(localization, default=0.0),
                "minimum_step_raw_iou": max(raw_localization, default=0.0),
            }
        )
    return {
        "gold_assets": sorted(relevant),
        "gold_pairs": sum(len(pairs) for _, pairs in gold.values()),
        "ranking": rows,
        "no_answer": not relevant,
        "no_answer_returned": not relevant and bool(matches),
        "resource_hit_at_5": any(row["asset_id"] in relevant for row in rows),
        "localized_hit_at_5": any(row["minimum_step_coverage"] >= 0.5 for row in rows),
        "iou_03_hit_at_5": any(row["minimum_step_clipped_iou"] >= 0.3 for row in rows),
        "iou_05_hit_at_5": any(row["minimum_step_clipped_iou"] >= 0.5 for row in rows),
    }


def summarize(
    singles: Sequence[reporting.JsonObject], sequences: Sequence[reporting.JsonObject]
) -> reporting.JsonObject:
    """Include every fixed query; failures/empty qrels are never removed."""
    positive = [row for row in sequences if not row["no_answer"]]
    absent = [row for row in sequences if row["no_answer"]]
    missing = [row for row in singles if row["resource"]["no_answer"]]
    return {
        "single_queries": len(singles),
        "single_positive_queries": len(singles) - len(missing),
        "resource_recall_at_10": sum(row["resource"]["recall_at_10"] for row in singles) / len(singles),
        "resource_ndcg_at_10": sum(row["resource"]["ndcg_at_10"] for row in singles) / len(singles),
        "unit_recall_at_10": sum(row["unit"]["recall_at_10"] for row in singles) / len(singles),
        "unit_ndcg_at_10": sum(row["unit"]["ndcg_at_10"] for row in singles) / len(singles),
        "single_no_answer_queries": len(missing),
        "single_no_answer_return_rate": sum(row["resource"]["no_answer_returned"] for row in missing) / len(missing)
        if missing
        else None,
        "sequence_queries": len(sequences),
        "sequence_positive_queries": len(positive),
        "sequence_positive_hit_at_5": sum(row["localized_hit_at_5"] for row in positive) / len(positive)
        if positive
        else 0.0,
        "sequence_positive_iou_03_hit_at_5": sum(row["iou_03_hit_at_5"] for row in positive) / len(positive)
        if positive
        else 0.0,
        "sequence_positive_iou_05_hit_at_5": sum(row["iou_05_hit_at_5"] for row in positive) / len(positive)
        if positive
        else 0.0,
        "sequence_no_answer_queries": len(absent),
        "sequence_no_answer_return_rate": sum(row["no_answer_returned"] for row in absent) / len(absent)
        if absent
        else None,
    }
