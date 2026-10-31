"""Bounded reciprocal rank fusion; native scoring systems never enter this module."""

from __future__ import annotations

import collections
import math
from collections.abc import Mapping, Sequence

from . import contracts


def unique_ids(ids: Sequence[int], limit: int) -> tuple[int, ...]:
    """Keep first-occurrence order, validate IDs and stop at the unique limit."""
    seen: set[int] = set()
    result = []
    if not limit:
        return ()
    for identifier in ids:
        identifier = contracts.asset_id(identifier)
        if identifier not in seen:
            seen.add(identifier)
            result.append(identifier)
            if len(result) == limit:
                break
    return tuple(result)


class RrfFusion:
    """Apply bounded weighted reciprocal rank fusion with stable ID ties."""

    def fuse(
        self, rankings: Mapping[str, Sequence[int]], config: contracts.SearchConfig
    ) -> tuple[contracts.FusedHit, ...]:
        """Deduplicate/truncate each branch, fuse contributions and select top-k."""
        return fuse_ordered_ids(
            {branch.name: rankings.get(branch.name, ()) for branch in config.branches},
            weights={branch.name: branch.weight for branch in config.branches},
            windows={branch.name: branch.window for branch in config.branches},
            constant=config.rrf_constant,
        )[: config.top_k]


def fuse_ordered_ids(
    rankings: Mapping[str, Sequence[int]],
    *,
    weights: Mapping[str, float] | None = None,
    windows: Mapping[str, int] | None = None,
    constant: float = 60,
) -> tuple[contracts.FusedHit, ...]:
    """Fuse ID rankings, also supporting the legacy unlimited-result facade.

    Args:
        rankings: Per-branch IDs, ordered best first.
        weights: Optional nonnegative branch weights; missing entries use one.
        windows: Unique-candidate budgets; missing entries use the full list.
        constant: Nonnegative smoothing constant in weight / (constant + rank).

    Returns:
        All fused candidates ordered by descending score then ascending ID.
        Ranks start at one after deduplication and windowing; repeated IDs never
        contribute twice within a branch. Scores are rank evidence, not calibrated
        probabilities or comparisons of native BM25/cosine scores.

    Raises:
        ValueError: An ID, weight, window or constant violates its contract.
    """
    if not math.isfinite(constant) or constant < 0:
        raise ValueError("RRF constant must be finite and nonnegative")
    contributions: dict[int, list[contracts.Contribution]] = collections.defaultdict(list)
    # Fix addition order as well as result ties: floating-point sums can depend
    # on branch iteration order even when the input ranks are identical.
    for name in sorted(rankings):
        weight = (weights or {}).get(name, 1.0)
        window = (windows or {}).get(name, len(rankings[name]))
        if not math.isfinite(weight) or weight < 0 or type(window) is not int or window < 0:
            raise ValueError("Weights and windows must be nonnegative and finite")
        if not weight or not window:
            continue
        for rank, identifier in enumerate(unique_ids(rankings[name], window), 1):
            contributions[identifier].append(contracts.Contribution(name, rank, weight / (constant + rank)))
    hits = [
        contracts.FusedHit(identifier, sum(item.score for item in parts), tuple(parts))
        for identifier, parts in contributions.items()
    ]
    return tuple(sorted(hits, key=lambda item: (-item.score, item.asset_id)))
