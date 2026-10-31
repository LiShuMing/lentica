"""Bind explicit multimodal requests to the compatible unit-stage pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from . import contracts, families, querying, stages, units

if TYPE_CHECKING:
    from numpy import typing as npt

    from . import vector


class MultimodalRetriever(Protocol):
    """Return pre-filtered unit IDs using the actual selected query modalities."""

    def retrieve(self, query: querying.MultimodalQuery, limit: int) -> Sequence[int]:
        """Use query.filters before limiting; do not substitute generated text."""
        ...


class TextRoute:
    """Expose an existing text retriever without pretending images contain text."""

    def __init__(self, retriever: contracts.Retriever) -> None:
        self.retriever = retriever

    def retrieve(self, query: querying.MultimodalQuery, limit: int) -> Sequence[int]:
        """Skip absent text rather than manufacture a textual image query."""
        if not query.text.strip():
            return ()
        return self.retriever.retrieve(contracts.SearchQuery(query.text, query.filters), limit)


class ImageVectorRoute:
    """Bind an image encoder only to a matching persisted/indexed vector space."""

    def __init__(
        self,
        retriever: vector.ExactVectorRetriever,
        encode_image: Callable[[Path], npt.ArrayLike],
        *,
        image_encoder_id: str,
    ) -> None:
        if image_encoder_id != retriever.encoder_id:
            raise ValueError("Image query and indexed vectors must identify the same aligned space")
        self.retriever, self.encode_image = retriever, encode_image

    def retrieve(self, query: querying.MultimodalQuery, limit: int) -> Sequence[int]:
        """Encode supplied pixels, retaining the shared pre-window filter domain."""
        if query.image is None or limit <= 0:
            return ()
        return self.retriever.retrieve_vector(self.encode_image(query.image), query.filters, limit)


@dataclass(frozen=True)
class RoutedResult:
    """Planning states and optional stage result; no capable route means no fetch."""

    kind: str
    plan: tuple[querying.PlannedRoute, ...]
    result: stages.StageResult | None


class _BoundRoute:
    """Keep the real multimodal request behind the legacy text-only interface."""

    def __init__(self, retriever: MultimodalRetriever, request: querying.MultimodalQuery) -> None:
        self.retriever, self.request = retriever, request

    def retrieve(self, query: contracts.SearchQuery, limit: int) -> Sequence[int]:
        return self.retriever.retrieve(self.request, limit)


class RoutedSearch:
    """Plan capabilities and family fusion, then reuse bounded resource stages.

    RouteSpec declares capabilities/family/state. The caller's StageConfig owns
    per-request windows and weights, overriding the declaration's defaults.
    Unregistered unavailable routes are permitted; a planned missing provider
    is a configuration error. Provider failures retain the explicit core policy.
    """

    def __init__(
        self,
        retrievers: Mapping[str, MultimodalRetriever],
        catalog: units.UnitStore,
        *,
        routes: Sequence[querying.RouteSpec],
        reranker: stages.Reranker | None = None,
        store: contracts.DocumentStore | None = None,
    ) -> None:
        self.retrievers = dict(retrievers)
        self.routes = {route.branch.name: route for route in routes}
        if len(self.routes) != len(routes):
            raise ValueError("Route names must be unique")
        self.catalog, self.reranker, self.store = catalog, reranker, store

    def search(
        self, query: querying.MultimodalQuery, config: stages.StageConfig, *, budget: querying.QueryBudget | None = None
    ) -> RoutedResult:
        """Validate inputs, plan and execute routes, retaining planning/run traces.

        Input limits fail before any provider runs. Only the actual request
        reaches providers; an internal image-only marker satisfies the old
        core's text invariant. Image-only requests skip the text reranker.
        """
        budget = budget or querying.QueryBudget()
        budget.validate(query)
        if any(branch.name not in self.routes for branch in config.search.branches):
            raise ValueError("Every branch requires an explicit capability declaration")
        planned = querying.plan(
            query, tuple(replace(self.routes[branch.name], branch=branch) for branch in config.search.branches), budget
        )
        active = [item for item in planned if item.status == "planned"]
        if not active:
            return RoutedResult(querying.classify(query), planned, None)
        if any(item.route.branch.name not in self.retrievers for item in active):
            raise ValueError("No provider registered for a planned route")
        branches = tuple(replace(item.route.branch, window=item.window) for item in active)
        bound = {item.route.branch.name: _BoundRoute(self.retrievers[item.route.branch.name], query) for item in active}
        fusion = families.FamilyRrfFusion(
            {item.route.branch.name: item.route.family for item in active if item.route.family is not None}
        )
        pipeline = stages.StagedSearch(
            bound, self.catalog, store=self.store, reranker=self.reranker if query.text.strip() else None, fusion=fusion
        )
        core_query = contracts.SearchQuery(query.text if query.text.strip() else "<image-only>", query.filters)
        result = pipeline.search(core_query, replace(config, search=replace(config.search, branches=branches)))
        return RoutedResult(querying.classify(query), planned, result)
