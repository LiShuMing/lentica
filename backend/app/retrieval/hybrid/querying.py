"""Explicit multimodal request capabilities and deterministic window planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import contracts, reranking

QUERY_KINDS = frozenset({"text", "identifier", "color", "image", "combined"})


@dataclass(frozen=True)
class MultimodalQuery:
    """Text, one explicitly selected image, or both, in the unit filter domain.

    Construction performs no filesystem access. Image size validation belongs to
    execution so imports/help and request planning never open private files.
    """

    text: str = ""
    image: Path | None = None
    filters: contracts.SearchFilter = field(default_factory=contracts.SearchFilter)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or (not self.text.strip() and self.image is None):
            raise ValueError("Supply nonempty text, an image, or both")
        if self.image is not None:
            object.__setattr__(self, "image", Path(self.image))


@dataclass(frozen=True)
class QueryBudget:
    """Bound input characters/pixels and the total planned branch windows."""

    max_characters: int = 8192
    max_image_pixels: int = 16_000_000
    max_image_bytes: int = 32 * 1024 * 1024
    total_candidates: int = 1000

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("Query budgets must be positive integers")
        if self.total_candidates > contracts.MAX_CANDIDATES:
            raise ValueError("Total candidates exceed the core execution bound")

    def validate(self, query: MultimodalQuery) -> None:
        """Open only a supplied image header and reject oversized inputs early."""
        if len(query.text) > self.max_characters:
            raise ValueError("Query exceeds character budget")
        if query.image is not None:
            if query.image.stat().st_size > self.max_image_bytes:
                raise ValueError("Query exceeds image byte budget")
            from PIL import Image

            with Image.open(query.image) as image:
                if image.width * image.height > self.max_image_pixels:
                    raise ValueError("Query exceeds image pixel budget")


@dataclass(frozen=True)
class RouteSpec:
    """A route's supported requests, evidence family and explicit index state."""

    branch: contracts.Branch
    kinds: frozenset[str] = QUERY_KINDS
    family: str | None = None
    state: str = "ready"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kinds", frozenset(self.kinds))
        if not self.kinds or not self.kinds <= QUERY_KINDS:
            raise ValueError("Unknown or empty route query capabilities")
        if self.family is not None and (not isinstance(self.family, str) or not self.family):
            raise ValueError("Evidence family must be a nonempty string")
        if self.state not in {"ready", "missing_modality", "not_indexed"}:
            raise ValueError("Unknown route availability state")


@dataclass(frozen=True)
class PlannedRoute:
    """The original route plus its allocated window and reason for exclusion."""

    route: RouteSpec
    window: int
    status: str


def classify(query: MultimodalQuery) -> str:
    """Recognize narrow color/code cases, without interpreting arbitrary prose."""
    if query.image is not None:
        return "combined" if query.text.strip() else "image"
    if query.text.strip().casefold() in {
        "red",
        "yellow",
        "green",
        "cyan",
        "blue",
        "magenta",
        "红色",
        "黄色",
        "绿色",
        "青色",
        "蓝色",
        "紫红色",
    }:
        return "color"
    return "identifier" if reranking.identifiers(query.text) else "text"


def plan(query: MultimodalQuery, routes: tuple[RouteSpec, ...], budget: QueryBudget) -> tuple[PlannedRoute, ...]:
    """Allocate shared candidate capacity fairly; retain explicit disabled states.

    Equal-share water filling honors small requested windows first. Any remainder
    goes in declared route order. This policy uses capabilities and budgets only;
    it does not select weights from test labels or compare native route scores.
    """
    if len({route.branch.name for route in routes}) != len(routes):
        raise ValueError("Route names must be unique")
    kind = classify(query)
    statuses = [
        "disabled"
        if not route.branch.window or not route.branch.weight
        else route.state
        if route.state != "ready"
        else "unsupported_query"
        if kind not in route.kinds
        else "planned"
        for route in routes
    ]
    windows = [0] * len(routes)
    remaining = budget.total_candidates
    pending = [index for index, status in enumerate(statuses) if status == "planned"]
    # At most MAX_WINDOW rounds, bounded by the existing total candidate budget.
    while pending and remaining:
        for index in pending:
            if not remaining:
                break
            windows[index] += 1
            remaining -= 1
        pending = [index for index in pending if windows[index] < routes[index].branch.window]
    return tuple(
        PlannedRoute(route, window, "budget_disabled" if status == "planned" and not window else status)
        for route, window, status in zip(routes, windows, statuses)
    )
