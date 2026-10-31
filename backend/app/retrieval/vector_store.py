"""Scoring boundary: replaceable without changing search plans or rank fusion."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Protocol


class VectorScorer(Protocol):
    def score(self, query: list[float], vectors: list[list[float]]) -> list[float]: ...


class NumpyVectorScorer:
    def score(self, query: list[float], vectors: list[list[float]]) -> list[float]:
        import numpy as np

        if not vectors:
            return []
        return (np.asarray(vectors, dtype=np.float32) @ np.asarray(query, dtype=np.float32)).tolist()


def score_rows(
    rows: Iterable, visual, text, visual_query, text_query, scorer: VectorScorer, diagnostics: dict | None = None
) -> Iterator:
    """Bound vector decoding/matrix memory; never compare unknown semantic spaces."""
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(scanned_vectors=0, incompatible_vectors=0, invalid_vectors=0)
    iterator = iter(rows)
    while batch := list(islice(iterator, 256)):
        spaces: dict[tuple[str, int], list] = {}
        for row in batch:
            diagnostics["scanned_vectors"] += 1
            vector = row[0]
            is_image = vector.namespace.split(":")[-1] == "image"
            encoder = visual if is_image else text
            query = visual_query if is_image else text_query
            identity = vector.encoder_id
            if identity is None and vector.namespace.startswith("hash:"):
                identity = "hash:sha256-expand-v1:384"
            if identity != encoder.encoder_id or vector.dims != len(query):
                diagnostics["incompatible_vectors"] += 1
                continue
            try:
                values = json.loads(vector.vector_json)
            except (ValueError, TypeError):
                diagnostics["invalid_vectors"] += 1
                continue
            if not isinstance(values, list) or len(values) != len(query):
                diagnostics["invalid_vectors"] += 1
                continue
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                diagnostics["invalid_vectors"] += 1
                continue
            key = (encoder.encoder_id, int(is_image))
            spaces.setdefault(key, []).append((row, values, query))
        for group in spaces.values():
            scores = scorer.score(group[0][2], [item[1] for item in group])
            for (row, _values, _query), score in zip(group, scores):
                yield row, float(score)
