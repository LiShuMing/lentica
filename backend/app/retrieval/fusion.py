from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalHit:
    asset_id: int
    rank: int
    score: float
    source: str
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class FusedHit:
    asset_id: int
    score: float
    sources: list[str]


def weighted_rrf(
    result_lists: dict[str, list[RetrievalHit]], *, weights: dict[str, float] | None = None, k: int = 60
) -> list[FusedHit]:
    """Fuse ranks without assuming scores from different models are comparable."""
    weights = weights or {}
    scores: dict[int, float] = defaultdict(float)
    sources: dict[int, list[str]] = defaultdict(list)
    for source, results in result_lists.items():
        source_weight = max(0.0, float(weights.get(source, 1.0)))
        if source_weight == 0:
            continue
        for rank, hit in enumerate(results, start=1):
            scores[hit.asset_id] += source_weight / (k + rank)
            if source not in sources[hit.asset_id]:
                sources[hit.asset_id].append(source)
    return [
        FusedHit(asset_id=asset_id, score=score, sources=sources[asset_id])
        for asset_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]
