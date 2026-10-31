"""Joint-model adapter boundaries and final reads, independent of model quality."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import adapters, stages, units, vision_reranking


class PairScores:
    scorer_id = "fixture-joint"

    def __init__(self, *, mode="valid"):
        self.calls, self.mode = [], mode

    def score_pairs(self, texts, paths):
        self.calls.append((tuple(texts), tuple(paths)))
        if self.mode == "drift":
            self.scorer_id = "changed"
        if self.mode == "short":
            return []
        return [float("nan") if self.mode == "nan" else float(path.name) for path in paths]


def test_joint_batches_complete_stable_permutation_and_snapshotted_paths():
    model = PairScores()
    paths = {identifier: Path(str(identifier % 2)) for identifier in range(7)}
    reranker = vision_reranking.JointTextImageReranker(model, paths)
    paths[0] = Path("99")
    candidates = [units.RetrievalUnit(identifier, identifier, "image") for identifier in range(7)]
    assert reranker.rerank(hybrid.SearchQuery("query"), candidates) == (1, 3, 5, 0, 2, 4, 6)
    assert [len(paths) for _, paths in model.calls] == [4, 3]
    assert model.calls[0][1][0] == Path("0")
    with pytest.raises(TypeError):
        reranker.image_paths[0] = Path("10")


@pytest.mark.parametrize("mode", ["short", "nan", "drift"])
def test_joint_rejects_bad_scores_or_model_drift(mode):
    scorer = PairScores(mode=mode)
    reranker = vision_reranking.JointTextImageReranker(scorer, {1: Path("1")})
    with pytest.raises(ValueError):
        reranker.rerank(hybrid.SearchQuery("x"), [units.RetrievalUnit(1, 1, "image")])


def test_joint_domain_and_budget_checks_happen_before_scoring():
    scorer = PairScores()
    reranker = vision_reranking.JointTextImageReranker(scorer, {1: Path("1")}, max_candidates=2)
    unit = units.RetrievalUnit(1, 1, "image", metadata={"tenant": "a"})
    for query, candidates in (
        (hybrid.SearchQuery("x", hybrid.SearchFilter(equals={"tenant": "b"})), [unit]),
        (hybrid.SearchQuery("x", hybrid.SearchFilter(allowed_ids=frozenset())), [unit]),
        (hybrid.SearchQuery("x"), [unit, unit]),
        (hybrid.SearchQuery("x"), [unit] * 3),
        (hybrid.SearchQuery("x"), [units.RetrievalUnit(2, 2, "image")]),
        (hybrid.SearchQuery("x"), [units.RetrievalUnit(1, 1, "text")]),
        (hybrid.SearchQuery("x" * 32_001), [unit]),
    ):
        with pytest.raises(ValueError):
            reranker.rerank(query, candidates)
    assert reranker.rerank(hybrid.SearchQuery("x"), []) == ()
    assert not scorer.calls
    scorer.scorer_id = "different-before-first-call"
    with pytest.raises(ValueError, match="identity changed"):
        reranker.rerank(hybrid.SearchQuery("x"), [unit])
    assert not scorer.calls


def test_joint_staged_search_fetches_only_final_parents_once_and_filters_before_scoring():
    model, fetched = PairScores(), []

    class Store:
        def fetch(self, ids):
            fetched.append(tuple(ids))
            return {identifier: {"parent": identifier} for identifier in ids}

    catalog = units.UnitCatalog(
        [
            units.RetrievalUnit(identifier, identifier + 100, "image", metadata={"tenant": "a"})
            for identifier in (1, 2, 3)
        ]
    )
    pipeline = stages.StagedSearch(
        {"image": adapters.RankedListRetriever([1, 2, 3], metadata={i: {"tenant": "a"} for i in (1, 2, 3)})},
        catalog,
        reranker=vision_reranking.JointTextImageReranker(model, {i: Path(str(i)) for i in (1, 2, 3)}),
        store=Store(),
    )
    config = stages.StageConfig(hybrid.SearchConfig((hybrid.Branch("image", window=3),), top_k=1), 3, 3)
    result = pipeline.search(hybrid.SearchQuery("x"), config)
    assert result.ordered_unit_ids == (3, 2, 1)
    assert fetched == [(103,)] and result.final_fetch_calls == 1
    calls = len(model.calls)
    result = pipeline.search(hybrid.SearchQuery("x", hybrid.SearchFilter(equals={"tenant": "b"})), config)
    assert not result.hits and len(model.calls) == calls and fetched == [(103,)]
