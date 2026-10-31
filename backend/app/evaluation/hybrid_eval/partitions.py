"""Deterministic source-connected partitions for multimodal evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceItem:
    """One evaluation item and every original source used by input or labels.

    Crops, pages and neighboring frames use their parent's source ID. A query
    with positives from several sources joins those sources into one component.
    duplicate_key can join paraphrase/normalized duplicate query groups as well.
    """

    item_id: str
    source_ids: frozenset[str]
    duplicate_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ids", frozenset(self.source_ids))
        if (
            not self.item_id
            or not self.source_ids
            or any(not isinstance(item, str) or not item for item in self.source_ids)
        ):
            raise ValueError("Items require an identity and nonempty original source identities")


def source_partitions(items: Sequence[SourceItem], *, seed: int = 42) -> dict[str, tuple[str, ...]]:
    """Split source-connected components into 60/20/20 dev/validation/test.

    At least three independent components are required. Components, not individual
    pages/queries, determine proportions; large components can make item counts
    uneven. No label scores or test measurements affect this deterministic split.
    """
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("Evaluation item identities must be unique")
    parents: dict[tuple[str, str], tuple[str, str]] = {}

    def root(key: tuple[str, str]) -> tuple[str, str]:
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for item in items:
        keys = [("source", source) for source in sorted(item.source_ids)]
        if item.duplicate_key is not None:
            keys.append(("duplicate", item.duplicate_key))
        for key in keys[1:]:
            parents[root(key)] = root(keys[0])
    components: dict[tuple[str, str], list[str]] = {}
    for item in items:
        components.setdefault(root(("source", min(item.source_ids))), []).append(item.item_id)
    groups = [tuple(sorted(values)) for values in components.values()]
    if len(groups) < 3:
        raise ValueError("Three independent source components are required for dev/validation/test")
    groups.sort(key=lambda group: hashlib.sha256((str(seed) + repr(group)).encode()).hexdigest())
    validation_count = max(1, int(len(groups) * 0.2))
    test_count = max(1, int(len(groups) * 0.2))
    dev_end = len(groups) - validation_count - test_count
    parts = (groups[:dev_end], groups[dev_end:-test_count], groups[-test_count:])
    return {
        name: tuple(sorted(item for group in part for item in group))
        for name, part in zip(("dev", "validation", "test"), parts)
    }
