"""In-memory BM25 retrieval with filters applied before the candidate window."""

from __future__ import annotations

import collections
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from . import contracts
from . import documents as corpus


def tokenize(text: str) -> list[str]:
    """Split lowercase ASCII words/identifiers and individual CJK characters.

    Underscores remain inside identifiers. There is no stemming, stop-word
    removal or general Unicode word segmentation.
    """
    return re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.lower())


@dataclass(frozen=True)
class Bm25Statistics:
    """Mergeable corpus statistics, independent of per-query eligible domains."""

    document_count: int
    total_length: int
    document_frequencies: Mapping[str, int]

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in (self.document_count, self.total_length)):
            raise ValueError("Corpus count and length must be nonnegative integers")
        frequencies = dict(self.document_frequencies)
        if any(
            not isinstance(key, str) or type(value) is not int or not 1 <= value <= self.document_count
            for key, value in frequencies.items()
        ):
            raise ValueError("Document frequencies must lie within the corpus count")
        object.__setattr__(self, "document_frequencies", MappingProxyType(frequencies))

    @classmethod
    def merge(cls, shards: Sequence[Bm25Statistics]) -> Bm25Statistics:
        """Sum disjoint shards; callers must ensure document domains do not overlap."""
        frequencies: collections.Counter[str] = collections.Counter()
        for shard in shards:
            frequencies.update(shard.document_frequencies)
        return cls(sum(item.document_count for item in shards), sum(item.total_length for item in shards), frequencies)


class Bm25Retriever:
    """Rank title/body matches with BM25 and ascending-ID tie-breaking."""

    def __init__(
        self,
        documents: Sequence[corpus.CorpusDocument],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
        statistics: Bm25Statistics | None = None,
    ) -> None:
        """Build an inverted index over title and body as one field.

        Args:
            documents: Corpus with unique asset IDs and optional filter metadata.
            k1: Positive term-frequency saturation parameter. Larger values let
                repeated document terms contribute for longer before saturating.
            b: Length normalization strength in [0, 1]; zero ignores length and
                one uses k1 * document/average length as the normalization term.
            tokenizer: Shared document/query tokenizer.
            statistics: Optional shared corpus statistics for disjoint shards.
                They must cover this shard; filtering never recomputes statistics.

        Raises:
            ValueError: IDs are duplicated or BM25 parameters are invalid.
        """
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        if len({item.asset_id for item in documents}) != len(documents):
            raise ValueError("Corpus IDs must be unique")
        self.k1, self.b, self.tokenizer = k1, b, tokenizer
        self.metadata = {item.asset_id: dict(item.metadata) for item in documents}
        self.postings: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        lengths: dict[int, int] = {}
        for document in documents:
            terms = collections.Counter(tokenizer(document.title + " " + document.text))
            lengths[document.asset_id] = sum(terms.values())
            for term, frequency in terms.items():
                self.postings[term].append((document.asset_id, frequency))
        count = len(documents)
        self.local_statistics = Bm25Statistics(
            count, sum(lengths.values()), {term: len(rows) for term, rows in self.postings.items()}
        )
        selected = statistics or self.local_statistics
        if (
            selected.document_count < count
            or selected.total_length < sum(lengths.values())
            or any(selected.document_frequencies.get(term, 0) < len(rows) for term, rows in self.postings.items())
        ):
            raise ValueError("Shared statistics must cover every local document/term")
        self.statistics = selected
        count = selected.document_count
        average = selected.total_length / count if count else 1.0
        # Cache the length-dependent denominator term. The fallback also keeps
        # an empty or entirely untokenizable corpus well-defined.
        self.norms = {identifier: k1 * (1 - b + b * length / (average or 1)) for identifier, length in lengths.items()}
        # Smoothed positive IDF: rarer terms carry more evidence. These corpus
        # statistics stay global when a query filter narrows eligible documents.
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in selected.document_frequencies.items()
        }

    def retrieve(self, query: contracts.SearchQuery, limit: int) -> tuple[int, ...]:
        """Return matching IDs after filtering, ordered by score then asset ID.

        Each term contributes query_tf * idf * document_tf * (k1 + 1) /
        (document_tf + length_norm). Query repetitions contribute linearly;
        documents without any matching token are omitted.
        """
        if limit <= 0:
            return ()
        scores = self.score_documents(query)
        return tuple(sorted(scores, key=lambda identifier: (-scores[identifier], identifier))[:limit])

    def score_documents(self, query: contracts.SearchQuery) -> dict[int, float]:
        """Return native scores for same-algorithm shard reduction/diagnostics.

        These scores must never be compared with dense scores or passed into
        rank fusion. Empty/nonmatching documents remain omitted.
        """
        scores: dict[int, float] = collections.defaultdict(float)
        for term, query_frequency in collections.Counter(self.tokenizer(query.text)).items():
            for identifier, tf in self.postings.get(term, ()):
                if query.filters.matches(identifier, self.metadata[identifier]):
                    scores[identifier] += (
                        query_frequency * self.idf[term] * tf * (self.k1 + 1) / (tf + self.norms[identifier])
                    )
        return dict(scores)
