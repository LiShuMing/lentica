from __future__ import annotations

import copy
import json
import subprocess
import sys
import zipfile

import pytest

from backend.app.evaluation.dataset import EvaluationCase
from backend.app.evaluation.hybrid_benchmark import Dataset, load_dataset
from backend.app.evaluation.hybrid_optimize import (
    choose_validation,
    evaluate,
    normalized_query,
    paired_comparison,
    partition_training,
    select,
    validate_lock,
)
from backend.app.retrieval.hybrid import SearchFilter, SearchQuery
from backend.app.retrieval.hybrid.adapters import CorpusDocument
from backend.app.retrieval.hybrid.sparse import SparseTfidfRetriever


@pytest.mark.parametrize("analyzer,ngrams", [("word", (1, 2)), ("char_wb", (3, 5))])
def test_sparse_filtering_ties_and_empty_queries(analyzer, ngrams):
    documents = [
        CorpusDocument(20, "receptor kinase", metadata={"group": "b"}),
        CorpusDocument(10, "receptor kinase", metadata={"group": "a"}),
        CorpusDocument(30, "battery power"),
    ]
    retriever = SparseTfidfRetriever(documents, analyzer=analyzer, ngram_range=ngrams)
    assert retriever.retrieve(SearchQuery("receptor"), 1) == (10,)
    assert retriever.retrieve(SearchQuery("receptor"), 10) == (10, 20)
    # A post-top-k filter would incorrectly produce an empty result here.
    query = SearchQuery("receptor", SearchFilter(equals={"group": "b"}, allowed_ids=frozenset({20})))
    assert retriever.retrieve(query, 1) == (20,)
    with pytest.raises(ValueError, match="nonempty"):
        SearchQuery("")
    assert retriever.retrieve(SearchQuery("zzzzzzz"), 10) == ()
    assert retriever.retrieve(SearchQuery("receptor"), 0) == ()


def test_sparse_rare_terms_and_character_variants():
    documents = [CorpusDocument(1, "ERR_301 receptors"), CorpusDocument(2, "receptive battery")]
    word = SparseTfidfRetriever(documents)
    assert word.retrieve(SearchQuery("ERR_301"), 1) == (1,)
    assert word.retrieve(SearchQuery("receptor"), 1) == ()
    char = SparseTfidfRetriever(documents, analyzer="char_wb", ngram_range=(3, 5))
    assert char.retrieve(SearchQuery("receptor"), 1) == (1,)
    with pytest.raises(ValueError, match="unique"):
        SparseTfidfRetriever([documents[0], documents[0]])
    with pytest.raises(ValueError, match="analyzer"):
        SparseTfidfRetriever(documents, analyzer="unknown")


def make_partition_datasets():
    documents = [CorpusDocument(0, "some text"), CorpusDocument(1, "other text")]
    train_cases = [EvaluationCase(str(i), f"distinct question {i}", {0}, "fixture") for i in range(10)]
    train_cases += [
        EvaluationCase("dup", "DISTINCT, question 1!", {1}, "fixture"),
        EvaluationCase("overlap", "Previously SEEN?", {0}, "fixture"),
    ]
    train = Dataset(documents, train_cases, {0: "zero", 1: "one"}, {})
    test = Dataset(documents, [EvaluationCase("test", "previously seen", {0}, "fixture")], train.source_ids, {})
    return train, test


def test_partition_prevents_duplicate_and_test_text_leakage():
    train, test = make_partition_datasets()
    development, validation, report = partition_training(train, test)
    assert report["excluded_train_query_ids"] == ["overlap"]
    assert len(development) + len(validation) == 11
    assert {normalized_query(case.query) for case in development}.isdisjoint(
        normalized_query(case.query) for case in validation
    )
    assert report["development_groups"] == 8
    assert report["validation_groups"] == 2
    train.cases.reverse()
    assert partition_training(train, test)[2] == report
    test.documents = [CorpusDocument(0, "changed"), test.documents[1]]
    with pytest.raises(ValueError, match="corpus mismatch"):
        partition_training(train, test)


def test_validation_guard_and_tie_break():
    def row(ndcg, recall=0.8, mrr=0.6):
        return {"metrics": {"ndcg_at_10": ndcg, "recall_at_10": recall, "mrr": mrr}}

    baseline = row(0.6)
    assert choose_validation({"baseline": baseline, "tiny": row(0.604)}) == "baseline"
    assert choose_validation({"baseline": baseline, "lossy": row(0.7, 0.78)}) == "baseline"
    assert choose_validation({"baseline": baseline, "b": row(0.65), "a": row(0.65)}) == "a"
    assert choose_validation({"baseline": baseline, "a": row(0.65), "b": row(0.65, 0.9)}) == "b"


def write_archive(path):
    corpus = [
        {"_id": str(i), "title": title, "text": text}
        for i, (title, text) in enumerate(
            [
                ("battery", "charging connector power adapter"),
                ("kinase", "receptors activate cell signaling"),
                ("network", "socket connection error ERR_301"),
                ("camera", "photo lens shutter aperture"),
                ("protein", "proteins interact with receptors"),
                ("engine", "motor fuel consumption"),
            ]
        )
    ]
    queries = [{"_id": f"train{i}", "text": f"{corpus[i % 6]['title']} search {i}"} for i in range(15)] + [
        {"_id": "test1", "text": "cell receptor signaling"},
        {"_id": "test2", "text": "ERR_301"},
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("tiny/corpus.jsonl", "\n".join(map(json.dumps, corpus)))
        archive.writestr("tiny/queries.jsonl", "\n".join(map(json.dumps, queries)))
        archive.writestr(
            "tiny/qrels/train.tsv",
            "query-id\tcorpus-id\tscore\n" + "\n".join(f"train{i}\t{i % 6}\t1" for i in range(15)),
        )
        archive.writestr("tiny/qrels/test.tsv", "query-id\tcorpus-id\tscore\ntest1\t1\t1\ntest2\t2\t1\n")


def test_separate_selection_and_locked_evaluation(tmp_path):
    archive = tmp_path / "fixture.zip"
    write_archive(archive)
    train, test = load_dataset(archive, split="train"), load_dataset(archive)
    assert len(train.cases) == 15 and len(test.cases) == 2
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataset(archive, split="dev")
    selection = select(train, test)
    assert len(selection["development"]) == 81
    assert len(selection["validation"]) == 5
    assert "per_query" not in selection
    assert "test1" not in json.dumps(selection["development"])
    # Roundtrip exercises the actual disk artifact's lists/dicts, not Python-only types.
    selection = json.loads(json.dumps(selection))
    first, second = evaluate(selection, train, test), evaluate(selection, train, test)
    assert first["ranking_sha256"] == second["ranking_sha256"]
    assert first["comparisons"]["baseline"]["metrics"]["recall_at_10"] == 1.0
    for live in first["live_pipeline"].values():
        assert live["queries"] == live["fetch_calls"] == 2
        assert live["max_records_per_fetch"] <= 10
    tampered = copy.deepcopy(selection)
    tampered["comparisons"]["selected"]["branches"][0]["weight"] = 5
    with pytest.raises(ValueError, match="checksum"):
        validate_lock(tampered, train, test)
    train.cases[0].relevant_asset_ids.add(5)
    with pytest.raises(ValueError, match="partition mismatch"):
        validate_lock(selection, train, test)


def test_paired_bootstrap_identical_and_strictly_better():
    cases = [EvaluationCase("a", "query", {1}, "fixture"), EvaluationCase("b", "another", {1}, "fixture")]
    baseline = {"a": [2, 1], "b": [3, 1]}
    identical = paired_comparison(cases, baseline, baseline)
    assert identical["bootstrap_95_percentile_interval"] == [0.0, 0.0]
    better = paired_comparison(cases, baseline, {"a": [1], "b": [1]})
    assert better["bootstrap_95_percentile_interval"][0] > 0
    assert better["improved"] == 2 and better["regressed"] == 0


def test_optimizer_help_has_no_numeric_or_application_imports():
    subprocess.run(
        [sys.executable, "-S", "-m", "backend.app.evaluation.hybrid_optimize", "--help"],
        check=True,
        capture_output=True,
        timeout=10,
    )
