"""Small, auditable reranking baselines; no models or relevance labels."""

from __future__ import annotations

import re
from collections.abc import Sequence

from . import contracts, units

_TOKENS = re.compile(r"[a-z0-9_]+(?:[.-][a-z0-9_]+)*", re.IGNORECASE)


def identifiers(text: str) -> frozenset[str]:
    """Extract code-like tokens, excluding plain words and standalone numbers.

    Underscores or a combination of letters and digits suggest an identifier.
    This deliberately narrow baseline does not recognize every identifier syntax.
    """
    return frozenset(
        token
        for token in _TOKENS.findall(text.casefold())
        if "_" in token or (any(char.isalpha() for char in token) and any(char.isdigit() for char in token))
    )


class ExactIdentifierReranker:
    """Prefer exact code evidence while preserving fusion order for ties.

    Native/OCR/transcript text is eligible, generated captions are not. OCR and
    transcription can themselves be wrong: this is exact extracted-text evidence,
    not human verification or an image-understanding claim. No candidates are
    deleted and unrelated queries retain their input order.
    """

    def rerank(self, query: contracts.SearchQuery, candidates: Sequence[units.RetrievalUnit]) -> tuple[int, ...]:
        """Order by matched distinct query identifiers, then original position."""
        wanted = identifiers(query.text)
        if not wanted:
            return tuple(unit.unit_id for unit in candidates)
        evidence = {unit.unit_id: set(_TOKENS.findall(unit.text().casefold())) for unit in candidates}
        return tuple(
            unit.unit_id
            for _, unit in sorted(
                enumerate(candidates), key=lambda item: (-len(wanted & evidence[item[1].unit_id]), item[0])
            )
        )
