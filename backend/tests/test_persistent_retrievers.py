"""Actual optional Tantivy/FAISS persistence, filtering and exact parity."""

from __future__ import annotations

import pytest

from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import ann, documents, fulltext, vector


def test_tantivy_reopens_and_filters_before_window_with_stable_ties(tmp_path):
    pytest.importorskip("tantivy")
    corpus = [
        documents.CorpusDocument(identifier, "hello ERR_42 中文", metadata={"tenant": "a" if identifier >= 90 else "b"})
        for identifier in reversed(range(100))
    ]
    built = fulltext.TantivyRetriever.build(tmp_path / "text", corpus, source_version="fixture-v1")
    query = hybrid.SearchQuery("hello")
    assert built.retrieve(query, 1) == (0,)
    filtered = hybrid.SearchQuery("hello", hybrid.SearchFilter(equals={"tenant": "a"}))
    assert built.retrieve(filtered, 2) == (90, 91)
    reloaded = fulltext.TantivyRetriever(tmp_path / "text")
    assert reloaded.retrieve(filtered, 2) == built.retrieve(filtered, 2)
    assert reloaded.retrieve(hybrid.SearchQuery("ERR_42", hybrid.SearchFilter(allowed_ids=frozenset({98}))), 1) == (98,)
    assert reloaded.retrieve(hybrid.SearchQuery("中文", hybrid.SearchFilter(allowed_ids=frozenset({99}))), 1) == (99,)
    assert reloaded.retrieve(hybrid.SearchQuery("hello", hybrid.SearchFilter(allowed_ids=frozenset())), 2) == ()
    assert reloaded.retrieve(hybrid.SearchQuery("absent"), 10) == ()
    with pytest.raises(FileExistsError):
        fulltext.TantivyRetriever.build(tmp_path / "text", corpus, source_version="v2")


def test_tantivy_metadata_filter_uses_shared_python_scalar_equality(tmp_path):
    pytest.importorskip("tantivy")
    corpus = [
        documents.CorpusDocument(1, "needle", metadata={"x": 1}),
        documents.CorpusDocument(2, "needle", metadata={"x": True}),
        documents.CorpusDocument(3, "needle", metadata={"x": None}),
        documents.CorpusDocument(4, "needle"),
    ]
    index = fulltext.TantivyRetriever.build(tmp_path / "text", corpus, source_version="v1")
    assert index.retrieve(hybrid.SearchQuery("needle", hybrid.SearchFilter(equals={"x": True})), 10) == (1, 2)
    assert index.retrieve(hybrid.SearchQuery("needle", hybrid.SearchFilter(equals={"x": None})), 10) == (3,)


def test_hnsw_real_selector_same_domain_exact_parity_and_reload(tmp_path):
    faiss = pytest.importorskip("faiss")
    import numpy as np

    faiss.omp_set_num_threads(1)
    generator = np.random.default_rng(42)
    matrix = generator.normal(size=(512, 24)).astype(np.float32)
    ids = tuple(range(1000, 1512))
    metadata = {identifier: {"tenant": identifier % 4} for identifier in ids}
    encode = lambda text: matrix[int(text)]
    approximate = ann.HnswRetriever.build(
        tmp_path / "vector",
        ids,
        matrix,
        encode,
        encoder_id="fixture-v1",
        source_version="source-v1",
        metadata=metadata,
        ef_search=512,
        exact_filter_threshold=8,
    )
    exact = vector.ExactVectorRetriever(
        ids, matrix, encode, encoder_id="fixture-v1", query_encoder_id="fixture-v1", metadata=metadata
    )
    for filters in [
        hybrid.SearchFilter(),
        hybrid.SearchFilter(equals={"tenant": 2}),
        hybrid.SearchFilter(allowed_ids=frozenset({1050, 1051})),
        hybrid.SearchFilter(allowed_ids=frozenset()),
    ]:
        query = hybrid.SearchQuery("50", filters)
        assert approximate.retrieve(query, 10) == exact.retrieve(query, 10)
    reloaded = ann.HnswRetriever(tmp_path / "vector", encode, query_encoder_id="fixture-v1")
    assert reloaded.retrieve(hybrid.SearchQuery("50"), 10) == approximate.retrieve(hybrid.SearchQuery("50"), 10)
    with pytest.raises(ValueError, match="same vector space"):
        ann.HnswRetriever(tmp_path / "vector", encode, query_encoder_id="other-model")
    with (tmp_path / "vector" / "index.faiss").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        ann.HnswRetriever(tmp_path / "vector", encode, query_encoder_id="fixture-v1")


def test_hnsw_zero_query_and_all_ties_have_exact_fallback(tmp_path):
    faiss = pytest.importorskip("faiss")
    import numpy as np

    faiss.omp_set_num_threads(1)
    ids = tuple(reversed(range(100)))
    matrix = np.ones((100, 4), dtype=np.float32)
    encode = lambda text: np.zeros(4) if text == "zero" else np.ones(4)
    index = ann.HnswRetriever.build(
        tmp_path / "vector",
        ids,
        matrix,
        encode,
        encoder_id="v1",
        source_version="s1",
        ef_search=8,
        exact_filter_threshold=0,
    )
    assert index.retrieve(hybrid.SearchQuery("nonzero"), 3) == (0, 1, 2)
    assert index.retrieve(hybrid.SearchQuery("zero"), 3) == ()
