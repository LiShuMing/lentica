"""Content-addressed multimodal cache keys, including plan and stage semantics."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence

from . import querying, snapshots, stages


def multimodal_key(
    snapshot: snapshots.Snapshot,
    query: querying.MultimodalQuery,
    config: stages.StageConfig,
    routes: Sequence[querying.RouteSpec],
    *,
    budget: querying.QueryBudget | None = None,
) -> str:
    """Hash actual image bytes and every filter/model/index/routing/rerank budget.

    A path/mtime alone is insufficient because an image can be replaced in place.
    Input bounds are checked before hashing. Arbitrary model/reranker extensions
    must identify their implementation/config in snapshot.identity.config_version;
    a truthful version identity remains the caller's responsibility.
    """
    budget = budget or querying.QueryBudget()
    budget.validate(query)
    value = {
        "schema": "lentica-multimodal-cache-v1",
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_sha256": snapshot.sha256,
        "identity": dataclasses.asdict(snapshot.identity),
        "text": query.text,
        "image_sha256": snapshots.sha256_file(query.image) if query.image is not None else None,
        "equals": dict(query.filters.equals),
        "allowed_ids": None if query.filters.allowed_ids is None else sorted(query.filters.allowed_ids),
        "stage_config": dataclasses.asdict(config),
        "budget": dataclasses.asdict(budget),
        "routes": [
            {
                "branch": dataclasses.asdict(route.branch),
                "kinds": sorted(route.kinds),
                "family": route.family,
                "state": route.state,
            }
            for route in sorted(routes, key=lambda route: route.branch.name)
        ],
    }
    return hashlib.sha256(snapshots.canonical(value).encode()).hexdigest()
