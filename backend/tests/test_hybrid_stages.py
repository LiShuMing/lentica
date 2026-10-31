"""Acceptance of evidence locations, stage budgets and failure boundaries."""

from __future__ import annotations

import subprocess
import sys

import pytest

from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import adapters, media, reranking, stages, units


def make_unit(identifier, parent, text="ERR_301", *, kind="native", metadata=None):
    return units.RetrievalUnit(
        identifier,
        parent,
        "page",
        locator=units.Locator(page=identifier + 1),
        evidence=(units.TextEvidence(kind, text, "test-extractor", "v1"),),
        metadata=metadata or {},
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"page": 0},
        {"page": True},
        {"bbox": (0.5, 0, 0.1, 1)},
        {"bbox": (0, 0, 1, float("nan"))},
        {"start_seconds": 1},
        {"start_seconds": 2, "end_seconds": 1},
    ],
)
def test_invalid_locations_fail(kwargs):
    with pytest.raises(ValueError):
        units.Locator(**kwargs)


def test_locations_and_evidence_are_snapshotted_and_separate_from_parent_ids():
    metadata = {"source": "a"}
    unit = make_unit(2, 99, metadata=metadata)
    metadata["source"] = "b"
    assert unit.metadata["source"] == "a"
    assert unit.unit_id == 2 and unit.asset_id == 99 and unit.locator.page == 3
    with pytest.raises(TypeError):
        unit.metadata["source"] = "c"
    with pytest.raises(ValueError, match="unique"):
        units.UnitCatalog([unit, unit])
    with pytest.raises(ValueError, match="page locator"):
        units.RetrievalUnit(1, 2, "page")
    with pytest.raises(ValueError, match="temporal"):
        units.RetrievalUnit(1, 2, "video")
    clip = units.RetrievalUnit(1, 2, "audio", units.Locator(start_seconds=1, end_seconds=3))
    assert clip.locator.end_seconds == 3


def test_stage_prefix_budget_parent_grouping_locations_and_single_final_fetch():
    corpus = [make_unit(1, 100), make_unit(2, 100), make_unit(3, 100), make_unit(4, 200), make_unit(5, 300)]
    calls = []

    class Store:
        def fetch(self, ids):
            calls.append(tuple(ids))
            return {identifier: str(identifier) for identifier in ids}

    class Reverse:
        def rerank(self, query, candidates):
            assert [unit.unit_id for unit in candidates] == [1, 4]
            return [unit.unit_id for unit in reversed(candidates)]

    config = stages.StageConfig(
        hybrid.SearchConfig((hybrid.Branch("text", window=5),), top_k=2),
        candidate_window=5,
        rerank_window=2,
        max_units_per_asset=1,
    )
    pipeline = stages.StagedSearch(
        {"text": adapters.RankedListRetriever([1, 2, 3, 4, 5])},
        units.UnitCatalog(corpus),
        reranker=Reverse(),
        store=Store(),
    )
    result = pipeline.search(hybrid.SearchQuery("ERR_301"), config)
    assert result.ordered_unit_ids == (4, 1, 5)
    assert [hit.asset_id for hit in result.hits] == [200, 100]
    assert result.hits[0].evidence[0].unit.locator.page == 5
    assert result.feature_units == 5 and result.reranked_units == 2
    assert result.final_fetch_calls == 1 and calls == [(200, 100)]
    assert config.search.top_k == 2


@pytest.mark.parametrize("returned", [[1, 1], [1], [1, 99], [True, 2]])
def test_invalid_reranker_permutations_fail_before_fetch(returned):
    class Broken:
        def rerank(self, *_):
            return returned

    pipeline = stages.StagedSearch(
        {"text": adapters.RankedListRetriever([1, 2])},
        units.UnitCatalog([make_unit(1, 10), make_unit(2, 20)]),
        reranker=Broken(),
    )
    config = stages.StageConfig(hybrid.SearchConfig((hybrid.Branch("text"),), top_k=1), 2, 2)
    with pytest.raises(hybrid.RetrievalError, match="permutation"):
        pipeline.search(hybrid.SearchQuery("ERR_301"), config)


def test_provider_filter_leak_and_unknown_unit_are_rejected():
    config = stages.StageConfig(hybrid.SearchConfig((hybrid.Branch("text"),), top_k=1))
    catalog = units.UnitCatalog([make_unit(1, 10, metadata={"source": "other"})])
    query = hybrid.SearchQuery("ERR_301", hybrid.SearchFilter(equals={"source": "allowed"}))

    class Leaking:
        def retrieve(self, *_):
            return [1]

    with pytest.raises(hybrid.RetrievalError, match="filter"):
        stages.StagedSearch({"text": Leaking()}, catalog).search(query, config)
    pipeline = stages.StagedSearch({"text": adapters.RankedListRetriever([99])}, catalog)
    with pytest.raises(hybrid.RetrievalError, match="catalog") as caught:
        pipeline.search(hybrid.SearchQuery("ERR_301"), config)
    assert isinstance(caught.value.__cause__, KeyError)


def test_feature_store_identity_and_partial_branch_boundaries():
    class IncompleteFeatures:
        def fetch_units(self, ids):
            return {}

    config = stages.StageConfig(hybrid.SearchConfig((hybrid.Branch("text"),), top_k=1))
    pipeline = stages.StagedSearch({"text": adapters.RankedListRetriever([1])}, IncompleteFeatures())
    with pytest.raises(hybrid.RetrievalError, match="identities"):
        pipeline.search(hybrid.SearchQuery("ERR_301"), config)

    class Broken:
        def retrieve(self, *_):
            raise OSError("branch failed")

    pipeline = stages.StagedSearch(
        {"text": adapters.RankedListRetriever([1]), "other": Broken()}, units.UnitCatalog([make_unit(1, 10)])
    )
    config = stages.StageConfig(
        hybrid.SearchConfig((hybrid.Branch("text"), hybrid.Branch("other")), top_k=1, failure_policy="partial")
    )
    result = pipeline.search(hybrid.SearchQuery("ERR_301"), config)
    assert result.degraded and result.hits[0].asset_id == 10
    assert result.branches[1].error == "OSError"


@pytest.mark.parametrize(
    "kwargs",
    [{"candidate_window": 9}, {"rerank_window": 9}, {"rerank_window": 101}, {"max_units_per_asset": 0}],
)
def test_stage_budget_validation(kwargs):
    with pytest.raises(ValueError, match="top_k"):
        stages.StageConfig(hybrid.SearchConfig((hybrid.Branch("text"),)), **kwargs)


def test_final_fetch_failure_is_chained_and_reranker_failure_propagates():
    failure = OSError("store unavailable")

    class BrokenStore:
        def fetch(self, ids):
            raise failure

    catalog = units.UnitCatalog([make_unit(1, 10)])
    config = stages.StageConfig(hybrid.SearchConfig((hybrid.Branch("text"),), top_k=1))
    pipeline = stages.StagedSearch({"text": adapters.RankedListRetriever([1])}, catalog, store=BrokenStore())
    with pytest.raises(hybrid.HydrationError) as caught:
        pipeline.search(hybrid.SearchQuery("ERR_301"), config)
    assert caught.value.__cause__ is failure

    class BrokenReranker:
        def rerank(self, *_):
            raise failure

    pipeline.reranker = BrokenReranker()
    with pytest.raises(OSError) as caught:
        pipeline.search(hybrid.SearchQuery("ERR_301"), config)
    assert caught.value is failure


def test_identifier_reranker_uses_exact_extracted_tokens_and_retains_ties():
    candidates = [
        make_unit(1, 10, "ERR_301", kind="caption"),
        make_unit(2, 20, "ERR_3010"),
        make_unit(3, 30, "err_301", kind="ocr"),
        make_unit(4, 40, "ERR_301"),
    ]
    reranker = reranking.ExactIdentifierReranker()
    assert reranker.rerank(hybrid.SearchQuery("show ERR_301"), candidates) == (3, 4, 1, 2)
    assert reranker.rerank(hybrid.SearchQuery("red photo"), candidates) == (1, 2, 3, 4)
    assert reranking.identifiers("p53 ERR_301 2026 receptor") == {"p53", "err_301"}


def test_text_evidence_routes_filter_before_window_and_work_without_other_modalities():
    catalog = units.UnitCatalog(
        [make_unit(1, 10, metadata={"source": "a"}), make_unit(2, 20, metadata={"source": "b"})]
    )
    route = media.EvidenceTextRetriever(catalog, frozenset({"native"}))
    query = hybrid.SearchQuery("ERR_301", hybrid.SearchFilter(equals={"source": "b"}))
    assert route.retrieve(query, 1) == (2,)
    assert media.EvidenceTextRetriever(catalog, frozenset({"caption"})).retrieve(query, 1) == ()


def test_new_core_imports_without_model_numeric_database_or_cli_side_effects():
    code = """
import sys
from backend.app.retrieval.hybrid import units, stages, reranking, media
from backend.app.evaluation import hybrid_diagnostics, multimodal_benchmark
assert not any(name.startswith(('numpy', 'PIL', 'fitz', 'torch', 'transformers',
    'sklearn', 'sqlalchemy', 'backend.app.core', 'backend.app.db')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-S", "-c", code], check=True, timeout=10)
