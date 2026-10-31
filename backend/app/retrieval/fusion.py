from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.retrieval.hybrid.fusion import fuse_ordered_ids


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
    rankings = {name: [hit.asset_id for hit in hits] for name, hits in result_lists.items()}
    active_weights = {name: max(0.0, float((weights or {}).get(name, 1.0))) for name in rankings}
    return [
        FusedHit(hit.asset_id, hit.score, [part.branch for part in hit.contributions])
        for hit in fuse_ordered_ids(rankings, weights=active_weights, constant=k)
    ]
