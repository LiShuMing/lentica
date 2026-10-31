"""Domain/budget/ordering checks; deterministic fixtures do not prove accuracy."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.retrieval.hybrid import contracts, units, vision_reranking


class Features:
    encoder_id = "fixture-only"

    def __init__(self):
        self.calls = 0

    def encode_query(self, text):
        import numpy as np

        self.calls += 1
        return np.array([1.0, 0.0])

    def encode_images(self, paths):
        import numpy as np

        self.calls += 1
        return np.array([[float(path.name), 0.0] for path in paths])


def test_actual_input_path_mapping_ties_and_complete_permutation():
    encoder = Features()
    paths = {3: Path("0"), 7: Path("1"), 1: Path("1")}
    reranker = vision_reranking.TextImageReranker(encoder, paths)
    paths[3] = Path("9")  # Caller changes cannot retarget an existing reranker.
    candidates = [units.RetrievalUnit(identifier, identifier, "image") for identifier in (3, 7, 1)]
    assert reranker.rerank(contracts.SearchQuery("pixels"), candidates) == (7, 1, 3)
    assert encoder.calls == 2
    with pytest.raises(ValueError, match="finite"):
        vision_reranking.order_candidates((1, 2), (float("nan"), 1.0))


def test_rejects_filters_missing_images_unsupported_units_duplicates_before_inference():
    encoder = Features()
    reranker = vision_reranking.TextImageReranker(encoder, {1: Path("1")}, max_candidates=2)
    image = units.RetrievalUnit(1, 1, "image", metadata={"album": "private"})
    invalid = [
        (contracts.SearchQuery("x", contracts.SearchFilter(equals={"album": "public"})), [image]),
        (contracts.SearchQuery("x", contracts.SearchFilter(allowed_ids=frozenset())), [image]),
        (contracts.SearchQuery("x"), [units.RetrievalUnit(2, 2, "image")]),
        (contracts.SearchQuery("x"), [units.RetrievalUnit(1, 1, "text")]),
        (contracts.SearchQuery("x"), [image, image]),
        (contracts.SearchQuery("x"), [image] * 3),
    ]
    for query, candidates in invalid:
        with pytest.raises(ValueError):
            reranker.rerank(query, candidates)
    assert reranker.rerank(contracts.SearchQuery("x"), []) == ()
    assert encoder.calls == 0
