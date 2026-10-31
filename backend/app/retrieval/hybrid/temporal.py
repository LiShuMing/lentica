"""Bounded same-resource temporal conjunction over explicit candidate evidence."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from . import contracts, units


@dataclass(frozen=True)
class TemporalMatch:
    """Common interval and original support; overlap alone is not semantic truth.

    Coarse transcription chunks retain their original uncertainty. Intersecting
    a30-second ASR window with a frame cannot certify a word at that frame time.
    """

    asset_id: int
    locator: units.Locator
    evidence: tuple[units.RetrievalUnit, ...]


@dataclass(frozen=True)
class TemporalSequenceMatch:
    """Ordered support with a covering interval, rather than a common overlap.

    Each step retains its original unit/time uncertainty. The covering interval
    includes gaps and does not certify that an event persisted throughout it.
    """

    asset_id: int
    locator: units.Locator
    evidence: tuple[units.RetrievalUnit, ...]


def _checked_groups(
    groups: Sequence[tuple[str, Sequence[units.RetrievalUnit]]], filters: contracts.SearchFilter
) -> list[tuple[units.RetrievalUnit, ...]]:
    """Snapshot and check candidate identities before any temporal combination."""
    if not 2 <= len(groups) <= 4 or any(not isinstance(name, str) or not name for name, _ in groups):
        raise ValueError("Temporal groups require2..4 explicit nonempty names")
    if len({name for name, _ in groups}) != len(groups):
        raise ValueError("Temporal group names must be unique")
    selected = [tuple(values) for _, values in groups]
    identities = {}
    for values in selected:
        if len(values) > 200 or len({unit.unit_id for unit in values}) != len(values):
            raise ValueError("Temporal groups require at most200 unique candidate units")
        for unit in values:
            if unit.unit_id in identities and unit != identities[unit.unit_id]:
                raise ValueError("Temporal unit identity differs across groups")
            identities[unit.unit_id] = unit
            if unit.kind not in {"audio", "video"} or unit.locator.start_seconds is None:
                raise ValueError("Temporal conjunction requires localized audio/video evidence")
            if not filters.matches(unit.unit_id, unit.metadata):
                raise ValueError("Temporal candidate violates shared filters")
    return selected


def intersect(
    groups: Mapping[str, Sequence[units.RetrievalUnit]],
    filters: contracts.SearchFilter,
    *,
    max_combinations: int = 10000,
    max_matches: int = 1000,
) -> tuple[TemporalMatch, ...]:
    """Join2..4 named required groups within one resource and a common interval.

    Inputs are already retrieved/filter-constrained unit candidates, at most200
    per group. No native/model scores, relevance labels, resource fetches or new
    units enter the operation. Stable output order uses parent/time/support IDs.

    Raises:
        ValueError: Candidate identity/filter/time or combination/result budgets
            fail. Excess never silently discards evidence or weakens conjunction.
    """
    if (
        not 2 <= len(groups) <= 4
        or any(not isinstance(name, str) or not name for name in groups)
        or type(max_combinations) is not int
        or not 1 <= max_combinations <= 100000
        or type(max_matches) is not int
        or not 1 <= max_matches <= 1000
    ):
        raise ValueError("Temporal groups/combination/result budgets are invalid")
    selected = _checked_groups([(name, groups[name]) for name in sorted(groups)], filters)
    if any(not values for values in selected):
        return ()
    partial = [TemporalMatch(unit.asset_id, unit.locator, (unit,)) for unit in selected[0]]
    attempts = 0
    for values in selected[1:]:
        by_asset = defaultdict(list)
        for unit in values:
            by_asset[unit.asset_id].append(unit)
        joined = []
        for previous in partial:
            for unit in by_asset[previous.asset_id]:
                attempts += 1
                if attempts > max_combinations:
                    raise ValueError("Temporal combination budget exceeded")
                start = max(previous.locator.start_seconds, unit.locator.start_seconds)
                end = min(previous.locator.end_seconds, unit.locator.end_seconds)
                if start < end:
                    joined.append(
                        TemporalMatch(
                            unit.asset_id,
                            units.Locator(start_seconds=start, end_seconds=end),
                            previous.evidence + (unit,),
                        )
                    )
                    if len(joined) > max_matches:
                        raise ValueError("Temporal result budget exceeded")
        partial = joined
    return tuple(
        sorted(
            partial,
            key=lambda match: (
                match.asset_id,
                match.locator.start_seconds,
                match.locator.end_seconds,
                tuple(unit.unit_id for unit in match.evidence),
            ),
        )
    )


def sequence(
    groups: Sequence[tuple[str, Sequence[units.RetrievalUnit]]],
    filters: contracts.SearchFilter,
    *,
    min_gap_seconds: float = 0.0,
    max_gap_seconds: float | None = None,
    max_combinations: int = 10000,
    max_matches: int = 1000,
) -> tuple[TemporalSequenceMatch, ...]:
    """Join explicitly ordered steps within one resource, preserving evidence.

    Args:
        groups: Two to four distinct named candidate groups in the required order.
            Names are descriptive; their lexical order never changes step order.
        filters: The shared filter domain, checked before any combinations.
        min_gap_seconds: Minimum end-to-next-start gap; zero permits touching.
        max_gap_seconds: Optional maximum adjacent-step gap; None leaves it unbounded.
        max_combinations: Maximum same-resource combination attempts.
        max_matches: Maximum intermediate/final results; excess fails without truncation.

    Returns:
        Matches sorted by resource, covering time and original support IDs.
        A match proves the candidate time relation, not model semantic accuracy.

    Raises:
        ValueError: Invalid identities, filters, gaps or exhausted budgets.
    """
    if (
        isinstance(groups, (str, bytes, Mapping))
        or type(min_gap_seconds) not in (int, float)
        or not math.isfinite(min_gap_seconds)
        or not 0 <= min_gap_seconds <= 3600
        or max_gap_seconds is not None
        and (
            type(max_gap_seconds) not in (int, float)
            or not math.isfinite(max_gap_seconds)
            or not min_gap_seconds <= max_gap_seconds <= 3600
        )
        or type(max_combinations) is not int
        or not 1 <= max_combinations <= 100000
        or type(max_matches) is not int
        or not 1 <= max_matches <= 1000
    ):
        raise ValueError("Temporal sequence gaps/groups/budgets are invalid")
    selected = _checked_groups(groups, filters)
    if any(not values for values in selected):
        return ()
    partial = [TemporalSequenceMatch(unit.asset_id, unit.locator, (unit,)) for unit in selected[0]]
    attempts = 0
    for values in selected[1:]:
        by_asset = defaultdict(list)
        for unit in values:
            by_asset[unit.asset_id].append(unit)
        joined = []
        for previous in partial:
            for unit in by_asset[previous.asset_id]:
                attempts += 1
                if attempts > max_combinations:
                    raise ValueError("Temporal sequence combination budget exceeded")
                gap = unit.locator.start_seconds - previous.locator.end_seconds
                if gap < min_gap_seconds or max_gap_seconds is not None and gap > max_gap_seconds:
                    continue
                joined.append(
                    TemporalSequenceMatch(
                        unit.asset_id,
                        units.Locator(
                            start_seconds=previous.locator.start_seconds, end_seconds=unit.locator.end_seconds
                        ),
                        previous.evidence + (unit,),
                    )
                )
                if len(joined) > max_matches:
                    raise ValueError("Temporal sequence result budget exceeded")
        partial = joined
    return tuple(
        sorted(
            partial,
            key=lambda match: (
                match.asset_id,
                match.locator.start_seconds,
                match.locator.end_seconds,
                tuple(unit.unit_id for unit in match.evidence),
            ),
        )
    )
