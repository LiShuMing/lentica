"""Resource-linked retrieval units and versioned, localizable text evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from . import contracts


@dataclass(frozen=True)
class Locator:
    """Source location: one-based page, normalized xyxy region or seconds.

    Coordinates are relative to the original page/image, not its thumbnail.
    Temporal bounds must occur together and describe a positive-length segment.
    """

    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.page is not None and (type(self.page) is not int or self.page < 1):
            raise ValueError("Page numbers must be positive integers")
        if self.bbox is not None:
            coordinates = tuple(self.bbox)
            if (
                len(coordinates) != 4
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates)
                or coordinates[0] >= coordinates[2]
                or coordinates[1] >= coordinates[3]
            ):
                raise ValueError("Bounding boxes require normalized nonempty xyxy coordinates")
            object.__setattr__(self, "bbox", coordinates)
        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("Temporal bounds must be supplied together")
        if self.start_seconds is not None and (
            not math.isfinite(self.start_seconds)
            or not math.isfinite(self.end_seconds)
            or not 0 <= self.start_seconds < self.end_seconds
        ):
            raise ValueError("Temporal bounds must be finite, nonnegative and increasing")


@dataclass(frozen=True)
class TextEvidence:
    """Extracted or generated text with the producer and version that made it."""

    kind: str
    text: str
    producer: str
    version: str

    def __post_init__(self) -> None:
        if self.kind not in {"native", "ocr", "caption", "transcript"}:
            raise ValueError("Unsupported text evidence kind")
        if not isinstance(self.text, str) or not self.producer or not self.version:
            raise ValueError("Text evidence requires text and explicit producer/version")


@dataclass(frozen=True)
class RetrievalUnit:
    """One searchable segment; unit_id and asset_id use separate ID domains.

    Scalar metadata inherits resource constraints before indexing. The top-level
    mapping is snapshotted. Evidence can be absent, e.g. for an image-only unit.
    """

    unit_id: int
    asset_id: int
    kind: str
    locator: Locator = field(default_factory=Locator)
    evidence: tuple[TextEvidence, ...] = ()
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        contracts.asset_id(self.unit_id)
        contracts.asset_id(self.asset_id)
        if self.kind not in {"text", "image", "page", "audio", "video"}:
            raise ValueError("Unsupported retrieval-unit kind")
        if self.kind == "page" and self.locator.page is None:
            raise ValueError("Page units require a page locator")
        if self.kind in {"audio", "video"} and self.locator.start_seconds is None:
            raise ValueError("Audio/video units require temporal bounds")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "metadata", contracts.SearchFilter(equals=self.metadata).equals)

    def text(self, kinds: frozenset[str] = frozenset({"native", "ocr", "transcript"})) -> str:
        """Join only selected evidence kinds; generated captions are opt-in."""
        return "\n".join(item.text for item in self.evidence if item.kind in kinds)


class UnitCatalog:
    """An explicit in-memory feature snapshot, independent of final asset stores."""

    def __init__(self, units: Sequence[RetrievalUnit]) -> None:
        """Validate unique unit IDs and retain frozen units by ID."""
        indexed = {unit.unit_id: unit for unit in units}
        if len(indexed) != len(units):
            raise ValueError("Unit IDs must be unique")
        self.units: Mapping[int, RetrievalUnit] = MappingProxyType(indexed)

    def fetch_units(self, ids: Sequence[int]) -> Mapping[int, RetrievalUnit]:
        """Resolve all candidate units; an unknown ID raises KeyError."""
        return {identifier: self.units[identifier] for identifier in ids}


class UnitStore(Protocol):
    """Read candidate features independently of final resource presentation."""

    def fetch_units(self, ids: Sequence[int]) -> Mapping[int, RetrievalUnit]:
        """Return exactly the requested units; missing records fail the search."""
        ...
