from __future__ import annotations

import math
import random
import subprocess
import sys
from dataclasses import replace

import pytest

from backend.app.retrieval.hybrid import (
    Branch,
    HybridSearch,
    HydrationError,
    RetrievalError,
    RrfFusion,
    SearchConfig,
    SearchFilter,
    SearchQuery,
)
from backend.app.retrieval.hybrid.adapters import (
    Bm25Retriever,
    CorpusDocument,
    ExactVectorRetriever,
    MemoryDocumentStore,
    RankedListRetriever,
)


def test_rrf_matches_independent_reference_on_random_lists():
    rng = random.Random(713)
    for _ in range(100):
        rankings = {name: [rng.randrange(12) for _ in range(25)] for name in ("text", "vector", "third")}
        branches = tuple(Branch(name, rng.randrange(9), rng.choice((0, 0.5, 1, 2))) for name in rankings)
        config = SearchConfig(branches, top_k=10, rrf_constant=rng.choice((0, 1, 10, 60, 1000)))
        rank_tables = {}
        for branch in branches:
            ids = list(dict.fromkeys(rankings[branch.name]))[: branch.window]
            rank_tables[branch.name] = {identifier: rank for rank, identifier in enumerate(ids, 1)}
        reference = {}
        for identifier in range(12):
            score = sum(
                branch.weight / (config.rrf_constant + rank_tables[branch.name][identifier])
                for branch in sorted(branches, key=lambda b: b.name)
                if branch.weight and identifier in rank_tables[branch.name]
            )
            if score:
                reference[identifier] = score
        expected = sorted(reference, key=lambda identifier: (-reference[identifier], identifier))[:10]
        hits = RrfFusion().fuse(rankings, config)
        assert [hit.asset_id for hit in hits] == expected
        assert [hit.score for hit in hits] == pytest.approx([reference[identifier] for identifier in expected])


def test_duplicate_rank_window_and_missing_branch_contributions():
    config = SearchConfig((Branch("a", 2), Branch("b", 1)), top_k=10)
    hits = RrfFusion().fuse({"a": [3, 3, 1, 99], "b": [2]}, config)
    assert [hit.asset_id for hit in hits] == [2, 3, 1]
    assert hits[-1].contributions[0].rank == 2
    assert hits[-1].score == 1 / 62
    assert all(hit.asset_id != 99 for hit in hits)


def test_rrf_constant_can_change_ranking():
    config = SearchConfig((Branch("text"), Branch("vector")), top_k=1, rrf_constant=0)
    rankings = {"text": [1, 2], "vector": [3, 2]}
    assert RrfFusion().fuse(rankings, config)[0].asset_id == 1
    assert RrfFusion().fuse(rankings, replace(config, rrf_constant=60))[0].asset_id == 2


class SpyStore(MemoryDocumentStore):
    def __init__(self, docs):
        super().__init__(docs)
        self.calls = []

    def fetch(self, ids):
        self.calls.append(tuple(ids))
        return super().fetch(ids)


def test_disabled_routes_and_single_final_fetch_are_repeatable():
    class Disabled:
        def retrieve(self, *_):
            raise AssertionError("Disabled retriever called")

    store = SpyStore([CorpusDocument(index, f"body {index}") for index in range(4)])
    search = HybridSearch({"text": RankedListRetriever([1, 2, 3]), "vector": Disabled()}, store=store)
    config = SearchConfig((Branch("text", 2), Branch("vector", 10, 0)), top_k=1)
    results = [search.search(SearchQuery("query"), config) for _ in range(3)]
    assert [result.hits for result in results] == [results[0].hits] * 3
    assert store.calls == [(1,)] * 3
    assert results[0].records[1].text == "body 1"
    assert results[0].branches[-1].status == "disabled"
    assert not results[0].degraded


def test_zero_window_does_not_require_registration_or_fetch():
    store = SpyStore([])
    result = HybridSearch({}, store=store).search(SearchQuery("none"), SearchConfig((Branch("off", 0),)))
    assert result.hits == () and store.calls == []


def test_partial_failure_is_explicit_and_strict_is_default():
    class Broken:
        def retrieve(self, *_):
            raise OSError("Failed")

    search = HybridSearch({"text": RankedListRetriever([1]), "vector": Broken()})
    config = SearchConfig((Branch("text"), Branch("vector")))
    with pytest.raises(RetrievalError, match="vector"):
        search.search(SearchQuery("x"), config)
    result = search.search(SearchQuery("x"), replace(config, failure_policy="partial"))
    assert result.degraded and result.hits[0].asset_id == 1
    assert result.branches[-1].status == "failed" and result.branches[-1].error == "OSError"
    with pytest.raises(ValueError, match="No retriever"):
        HybridSearch({}).search(SearchQuery("x"), replace(config, failure_policy="partial"))


def test_missing_final_record_is_always_an_error():
    search = HybridSearch({"text": RankedListRetriever([1])}, store=MemoryDocumentStore([]))
    with pytest.raises(HydrationError):
        search.search(SearchQuery("x"), SearchConfig((Branch("text"),), failure_policy="partial"))


def test_fusion_is_replaceable_and_invalid_outputs_do_not_reach_store():
    from backend.app.retrieval.hybrid.contracts import FusedHit

    class SelectLast:
        def fuse(self, rankings, config):
            return (FusedHit(rankings["text"][-1], 1.0, ()),)

    store = SpyStore([CorpusDocument(1, "one"), CorpusDocument(2, "two")])
    engine = HybridSearch({"text": RankedListRetriever([1, 2])}, store=store, fusion=SelectLast())
    config = SearchConfig((Branch("text"),), top_k=1)
    assert engine.search(SearchQuery("x"), config).hits[0].asset_id == 2
    assert store.calls == [(2,)]

    class InvalidFusion:
        def fuse(self, *_):
            return (FusedHit(999, 1.0, ()),)

    engine.fusion = InvalidFusion()
    with pytest.raises(RetrievalError, match="candidate contract"):
        engine.search(SearchQuery("x"), config)
    assert store.calls == [(2,)]


def test_both_adapters_filter_before_their_candidate_windows():
    docs = [
        CorpusDocument(1, "error error error", metadata={"category": "other"}),
        CorpusDocument(2, "error", metadata={"category": "target"}),
    ]
    text = Bm25Retriever(docs)
    vector = ExactVectorRetriever(
        [1, 2],
        [[1, 0], [0.1, 1]],
        lambda _: [1, 0],
        encoder_id="fixture",
        query_encoder_id="fixture",
        metadata={doc.asset_id: doc.metadata for doc in docs},
    )
    query = SearchQuery("error", SearchFilter(equals={"category": "target"}))
    for retriever in (text, vector):
        assert retriever.retrieve(query, 1) == (2,)
        assert retriever.retrieve(SearchQuery("error"), 1) == (1,)
        assert retriever.retrieve(SearchQuery("error", SearchFilter(allowed_ids=frozenset())), 1) == ()
    result = HybridSearch({"text": text, "vector": vector}).search(
        query, SearchConfig((Branch("text", 1), Branch("vector", 1)), top_k=1)
    )
    assert result.hits[0].asset_id == 2 and len(result.hits[0].contributions) == 2


def test_filters_copy_mutable_inputs_and_do_not_match_absent_null_keys():
    values = {"category": "a"}
    query_filter = SearchFilter(equals=values)
    values["category"] = "b"
    assert query_filter.matches(1, {"category": "a"})
    assert not SearchFilter(equals={"absent": None}).matches(1, {})
    with pytest.raises(TypeError):
        query_filter.equals["category"] = "c"


@pytest.mark.parametrize(
    "branch",
    [lambda: Branch("x", -1), lambda: Branch("x", 10001), lambda: Branch("x", 1, math.nan), lambda: Branch("x", 1, -1)],
)
def test_invalid_branch_parameters_fail(branch):
    with pytest.raises(ValueError):
        branch()


@pytest.mark.parametrize(
    "kwargs",
    [{"top_k": 0}, {"top_k": True}, {"rrf_constant": -1}, {"rrf_constant": math.inf}, {"failure_policy": "ignore"}],
)
def test_invalid_search_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        SearchConfig((Branch("x"),), **kwargs)


def test_duplicate_names_total_budget_and_adapter_contracts():
    with pytest.raises(ValueError):
        SearchConfig((Branch("x"), Branch("x")))
    with pytest.raises(ValueError):
        SearchConfig(tuple(Branch(str(index), 10000) for index in range(6)))

    class Invalid:
        def retrieve(self, *_):
            return [-1]

    with pytest.raises(RetrievalError):
        HybridSearch({"x": Invalid()}).search(SearchQuery("x"), SearchConfig((Branch("x"),)))


def test_vector_space_and_query_dimensions_are_checked():
    with pytest.raises(ValueError, match="same vector space"):
        ExactVectorRetriever([1], [[1, 0]], lambda _: [1, 0], encoder_id="a", query_encoder_id="b")
    vector = ExactVectorRetriever([1], [[1, 0]], lambda _: [1], encoder_id="a", query_encoder_id="a")
    with pytest.raises(ValueError, match="dimension"):
        vector.retrieve(SearchQuery("x"), 1)


def test_core_imports_without_site_packages_or_application_configuration():
    code = """
import sys
from backend.app.retrieval.hybrid import HybridSearch, SearchConfig
assert not any(name.startswith(('numpy', 'sqlalchemy', 'torch', 'backend.app.core')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-S", "-c", code], check=True, timeout=10)
