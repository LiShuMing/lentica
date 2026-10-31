"""Statistics injection and fixed filtered-denominator acceptance."""

from __future__ import annotations

import pytest

from backend.app.evaluation import dataset
from backend.app.evaluation.hybrid_eval import index_quality
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import documents, lexical


def test_merged_statistics_match_whole_corpus_scores_and_ranks_with_filters():
    corpus = [
        documents.CorpusDocument(index, "needle " * (index + 1) + ("rare" if index % 3 == 0 else "common"))
        for index in range(12)
    ]
    groups = index_quality.shard_groups(corpus)
    locals_ = [lexical.Bm25Retriever(group) for group in groups]
    stats = lexical.Bm25Statistics.merge([item.local_statistics for item in locals_])
    shards = [lexical.Bm25Retriever(group, statistics=stats) for group in groups]
    whole = lexical.Bm25Retriever(corpus)
    query = hybrid.SearchQuery("needle rare needle", hybrid.SearchFilter(allowed_ids=frozenset(range(3, 10))))
    scores = {key: score for shard in shards for key, score in shard.score_documents(query).items()}
    assert scores == whole.score_documents(query)
    assert tuple(sorted(scores, key=lambda key: (-scores[key], key))) == whole.retrieve(query, 50)
    with pytest.raises(ValueError, match="cover"):
        lexical.Bm25Retriever(corpus, statistics=lexical.Bm25Statistics(1, 1, {}))


def test_filtered_quality_excludes_no_eligible_positive_queries():
    cases = [dataset.EvaluationCase("a", "first", {1}, "test"), dataset.EvaluationCase("b", "second", set(), "test")]
    actual = index_quality.summarize(cases, {"a": [1], "b": []})
    assert actual["positive_queries"] == actual["no_eligible_relevant_queries"] == 1
    assert actual["metrics"]["recall_at_10"] == 1
