"""Rank native action-class features by ID under shared pre-window filters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

from . import action_model, contracts, units


def _label(text: str) -> str:
    return " ".join(text.casefold().split())


class ActionLabelRetriever:
    """Narrow native-vocabulary route; original sequence features only.

    Scores remain inside this adapter. Querying reuses explicit class features;
    it does not run a model, infer free-text semantics or reject unknown events.
    Unknown class names raise rather than pretending to be supported queries.
    """

    def __init__(self, catalog: units.UnitStore, features: Mapping[int, action_model.ActionScores]) -> None:
        """Capture aligned original-only class features and immutable unit metadata."""
        if not features:
            raise ValueError("Action index requires explicit nonempty features")
        captured = {contracts.asset_id(identifier): row for identifier, row in features.items()}
        first = next(iter(captured.values()))
        labels = tuple(_label(label) for label in first.labels)
        if not labels or len(labels) != len(set(labels)) or any(not label for label in labels):
            raise ValueError("Action vocabulary must be nonempty and unique")
        selected = catalog.fetch_units(tuple(captured))
        if set(selected) != set(captured):
            raise ValueError("Action feature/catalog identities differ")
        for identifier, row in captured.items():
            unit = selected[identifier]
            if unit.unit_id != identifier or unit.kind != "video" or unit.locator.start_seconds is None:
                raise ValueError("Action features require matching timed video units")
            if (
                row.input_kind != "original"
                or row.labels != first.labels
                or row.version != first.version
                or len(row.logits) != len(labels)
                or not all(math.isfinite(value) for value in row.logits)
            ):
                raise ValueError("Action features must share original sequence/model/vocabulary identity")
        self.features = MappingProxyType(captured)
        self.metadata = MappingProxyType({identifier: unit.metadata for identifier, unit in selected.items()})
        self.labels = MappingProxyType({label: index for index, label in enumerate(labels)})
        self.version = first.version

    def retrieve(self, query: contracts.SearchQuery, limit: int) -> tuple[int, ...]:
        """Rank eligible units for an exact native action class before limiting."""
        if type(limit) is not int or not 0 <= limit <= contracts.MAX_WINDOW:
            raise ValueError("Action window exceeds bounded retrieval limits")
        if limit == 0:
            return ()
        label = _label(query.text)
        if label not in self.labels:
            raise ValueError("Action route supports exact native class names only")
        column = self.labels[label]
        eligible = (
            identifier for identifier in self.features if query.filters.matches(identifier, self.metadata[identifier])
        )
        return tuple(
            sorted(eligible, key=lambda identifier: (-self.features[identifier].logits[column], identifier))[:limit]
        )
