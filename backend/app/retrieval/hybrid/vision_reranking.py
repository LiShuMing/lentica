"""Bounded independent or pair-conditioned image/text ranking in explicit domains."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from . import contracts, units

if TYPE_CHECKING:
    from numpy import typing as npt


class ImageTextEncoder(Protocol):
    """Concrete consumer contract shared by local CLIP and SigLIP adapters."""

    encoder_id: str

    def encode_texts(self, texts: Sequence[str]) -> npt.NDArray:
        """Return aligned normalized text representations in input order."""
        ...

    def encode_query(self, text: str) -> npt.NDArray:
        """Return one normalized text representation."""
        ...

    def encode_images(self, paths: Sequence[Path]) -> npt.NDArray:
        """Return aligned normalized image representations in input order."""
        ...


class ImageTextPairScorer(Protocol):
    """Actual joint image/text pair scores, separate from ranked-ID fusion."""

    scorer_id: str

    def score_pairs(self, texts: Sequence[str], paths: Sequence[Path]) -> Sequence[float]:
        """Return one finite native matching score for each aligned pair."""
        ...


def _path_mapping(image_paths: Mapping[int, Path], max_candidates: int) -> Mapping[int, Path]:
    if type(max_candidates) is not int or not 1 <= max_candidates <= 200:
        raise ValueError("Image reranking candidate budget must be within1..200")
    for identifier in image_paths:
        contracts.asset_id(identifier)
    return MappingProxyType({identifier: Path(path) for identifier, path in image_paths.items()})


def _candidates(
    query: contracts.SearchQuery,
    candidates: Sequence[units.RetrievalUnit],
    image_paths: Mapping[int, Path],
    max_candidates: int,
) -> tuple[list[int], list[Path]]:
    """Share domain checks before either independent or joint model inference."""
    if len(candidates) > max_candidates or not query.text.strip() or len(query.text) > 32_000:
        raise ValueError("Text/image reranking query or candidate budget exceeded")
    ids = [unit.unit_id for unit in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Image reranking requires unique candidate IDs")
    for unit in candidates:
        if unit.kind not in {"image", "video"} or unit.unit_id not in image_paths:
            raise ValueError("Image reranker requires explicit image/frame paths for supported candidates")
        if not query.filters.matches(unit.unit_id, unit.metadata):
            raise ValueError("Image reranking candidate violates query filters")
    return ids, [image_paths[identifier] for identifier in ids]


def order_candidates(ids: Sequence[int], scores: Sequence[float]) -> tuple[int, ...]:
    """Return a full permutation with input-order ties, rejecting invalid scores."""
    import math

    if len(ids) != len(scores) or len(set(ids)) != len(ids) or any(not math.isfinite(score) for score in scores):
        raise ValueError("Image reranking requires unique IDs and aligned finite scores")
    return tuple(ids[index] for index in sorted(range(len(ids)), key=lambda index: (-scores[index], index)))


class TextImageReranker:
    """Score explicit candidate pixels against a text query, with no hidden cache.

    This independent-encoder similarity is not cross-attention. Paths are a
    snapshotted ID mapping rather than user metadata/captions or a directory
    enumeration. All candidates must be supported image units in the query's
    unit filter domain; explicitly mapped video units use sampled frame pixels.
    Failure does not silently skip or reorder unknown units.
    """

    def __init__(self, encoder: ImageTextEncoder, image_paths: Mapping[int, Path], *, max_candidates: int = 20) -> None:
        self.encoder, self.max_candidates = encoder, max_candidates
        self.image_paths = _path_mapping(image_paths, max_candidates)

    def rerank(self, query: contracts.SearchQuery, candidates: Sequence[units.RetrievalUnit]) -> tuple[int, ...]:
        """Actually encode at most the declared bound; preserve a complete order."""
        ids, paths = _candidates(query, candidates, self.image_paths, self.max_candidates)
        if not candidates:
            return ()
        import numpy as np

        images = np.asarray(self.encoder.encode_images(paths))
        text = np.asarray(self.encoder.encode_query(query.text))
        if images.ndim != 2 or text.ndim != 1 or images.shape != (len(ids), len(text)):
            raise ValueError("Image reranking encoder returned misaligned features")
        return order_candidates(ids, images @ text)


class JointTextImageReranker:
    """Bound actual pair-conditioned inference while preserving candidate domains."""

    def __init__(
        self,
        scorer: ImageTextPairScorer,
        image_paths: Mapping[int, Path],
        *,
        max_candidates: int = 20,
        batch_size: int = 4,
    ) -> None:
        """Capture explicit pixels/scorer identity; construction runs no inference."""
        if type(batch_size) is not int or not 1 <= batch_size <= 4 or not scorer.scorer_id:
            raise ValueError("Joint image reranking requires bounded batches and explicit model identity")
        self.scorer, self.scorer_id = scorer, scorer.scorer_id
        self.image_paths = _path_mapping(image_paths, max_candidates)
        self.max_candidates, self.batch_size = max_candidates, batch_size

    def rerank(self, query: contracts.SearchQuery, candidates: Sequence[units.RetrievalUnit]) -> tuple[int, ...]:
        """Score all supplied candidates freshly; return a complete stable permutation."""
        ids, paths = _candidates(query, candidates, self.image_paths, self.max_candidates)
        scores = []
        for start in range(0, len(ids), self.batch_size):
            if self.scorer.scorer_id != self.scorer_id:
                raise ValueError("Joint image scorer identity changed")
            chosen = paths[start : start + self.batch_size]
            values = tuple(self.scorer.score_pairs([query.text] * len(chosen), chosen))
            if len(values) != len(chosen) or self.scorer.scorer_id != self.scorer_id:
                raise ValueError("Joint image scores/model identity differ")
            scores.extend(values)
        return order_candidates(ids, scores)
