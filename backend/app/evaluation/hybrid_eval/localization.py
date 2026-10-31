"""Separate resource/unit recall, localization and no-answer false positives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from backend.app.retrieval.hybrid import contracts, units


def locator_iou(actual: units.Locator, expected: units.Locator) -> float:
    """Compare every annotated dimension; page mismatch gives zero overlap.

    A page-only label uses page accuracy. When both box and time are annotated,
    their IoUs multiply. Missing predicted dimensions fail instead of being
    silently omitted. Unlocalized labels are not valid localization judgments.
    """
    if expected == units.Locator():
        raise ValueError("Localization labels must specify a page, region or interval")
    if expected.page is not None and actual.page != expected.page:
        return 0.0
    score = 1.0
    if expected.bbox is not None:
        if actual.bbox is None:
            return 0.0
        a, b = actual.bbox, expected.bbox
        intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        score *= intersection / union
    if expected.start_seconds is not None:
        if actual.start_seconds is None:
            return 0.0
        intersection = max(
            0.0, min(actual.end_seconds, expected.end_seconds) - max(actual.start_seconds, expected.start_seconds)
        )
        union = max(actual.end_seconds, expected.end_seconds) - min(actual.start_seconds, expected.start_seconds)
        score *= intersection / union
    return score


@dataclass(frozen=True)
class Judgment:
    """Explicit positives; an empty set means judged no-answer, not unjudged."""

    assets: frozenset[int]
    unit_ids: frozenset[int]
    locations: Mapping[int, tuple[units.Locator, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assets", frozenset(contracts.asset_id(item) for item in self.assets))
        object.__setattr__(self, "unit_ids", frozenset(contracts.asset_id(item) for item in self.unit_ids))
        locations = {contracts.asset_id(key): tuple(value) for key, value in self.locations.items()}
        if bool(self.assets) != bool(self.unit_ids) or not set(locations) <= self.assets:
            raise ValueError("Positive unit/resource labels must agree; locations identify relevant resources")
        if any(not labels or units.Locator() in labels for labels in locations.values()):
            raise ValueError("Location judgments require nonempty explicit locations")
        object.__setattr__(self, "locations", MappingProxyType(locations))


def score(judgment: Judgment, predicted: Sequence[units.RetrievalUnit], *, k: int = 10) -> dict[str, float | None]:
    """Score first-k distinct units/resources independently and locator overlap.

    No-answer cases have null recall/localization and explicit false-positive
    status. Aggregation must keep their denominator separate from positive recall.
    Localization averages best overlap per annotated resource, including misses.
    """
    if type(k) is not int or k <= 0:
        raise ValueError("Evaluation k must be positive")
    assets = tuple(dict.fromkeys(unit.asset_id for unit in predicted))[:k]
    identifiers = tuple(dict.fromkeys(unit.unit_id for unit in predicted))[:k]
    visible = [unit for unit in predicted if unit.unit_id in identifiers and unit.asset_id in assets]
    if not judgment.assets:
        return {
            "resource_recall": None,
            "unit_recall": None,
            "localization_iou": None,
            "no_answer_false_positive": float(bool(assets)),
        }
    overlaps = [
        max(
            (
                locator_iou(unit.locator, label)
                for unit in visible
                if unit.asset_id == identifier and identifier in assets
                for label in labels
            ),
            default=0.0,
        )
        for identifier, labels in judgment.locations.items()
    ]
    return {
        "resource_recall": len(judgment.assets & set(assets)) / len(judgment.assets),
        "unit_recall": len(judgment.unit_ids & set(identifiers)) / len(judgment.unit_ids),
        "localization_iou": sum(overlaps) / len(overlaps) if overlaps else None,
        "no_answer_false_positive": None,
    }
