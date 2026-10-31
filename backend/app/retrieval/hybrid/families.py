"""Rank fusion that caps correlated routes at one contribution per family."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from . import contracts, fusion


class FamilyRrfFusion:
    """Sum independent evidence families, taking the best route within each.

    A family is an explicit provenance/configuration decision, not inferred from
    matching strings. Omitted routes remain independent. Contributions retain
    the winning original branch/rank; equal contributions choose branch name.
    """

    def __init__(self, families: Mapping[str, str]) -> None:
        """Snapshot nonempty branch-to-family names without loading any data."""
        if any(not isinstance(value, str) or not value for pair in families.items() for value in pair):
            raise ValueError("Branch and evidence family names must be nonempty strings")
        self.families = MappingProxyType(dict(families))

    def fuse(
        self, rankings: Mapping[str, Sequence[int]], config: contracts.SearchConfig
    ) -> tuple[contracts.FusedHit, ...]:
        """Apply branch budgets, then cap each unit's evidence by declared family."""
        if not self.families:
            return fusion.RrfFusion().fuse(rankings, config)
        unbounded = fusion.fuse_ordered_ids(
            {branch.name: rankings.get(branch.name, ()) for branch in config.branches},
            weights={branch.name: branch.weight for branch in config.branches},
            windows={branch.name: branch.window for branch in config.branches},
            constant=config.rrf_constant,
        )
        hits = []
        for hit in unbounded:
            grouped: dict[tuple[str, str], contracts.Contribution] = {}
            for part in hit.contributions:
                # Separate namespaces prevent a declared family name from
                # accidentally colliding with an ungrouped branch name.
                key = ("family", self.families[part.branch]) if part.branch in self.families else ("route", part.branch)
                if key not in grouped or part.score > grouped[key].score:
                    grouped[key] = part
            parts = tuple(sorted(grouped.values(), key=lambda part: part.branch))
            hits.append(contracts.FusedHit(hit.asset_id, math.fsum(part.score for part in parts), parts))
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.asset_id))[: config.top_k])
