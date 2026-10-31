"""Actual optional Parquet ID/qrel/metadata round trip and bounded reload."""

from __future__ import annotations

import pytest

from backend.app.evaluation.dataset import EvaluationCase
from backend.app.evaluation.hybrid_eval import dataset, parquet_io
from backend.app.retrieval.hybrid import documents


def test_parquet_round_trip_preserves_ids_metadata_queries_and_judgments(tmp_path):
    pytest.importorskip("pyarrow")
    source = dataset.Dataset(
        [
            documents.CorpusDocument(1, "错误码 ERR_42", "说明", {"tenant": "a", "flag": None}),
            documents.CorpusDocument(5, "native text", metadata={"tenant": "b"}),
        ],
        [EvaluationCase("q1", "ERR_42", {1, 5}, "identifier")],
        {1: "original-pdf", 5: "original-image"},
        {"kind": "test"},
    )
    manifest = parquet_io.write(tmp_path / "parquet", source)
    restored = parquet_io.read(tmp_path / "parquet")
    assert restored.documents == source.documents
    assert restored.cases == source.cases and restored.source_ids == source.source_ids
    assert manifest["report"]["positive_pairs"] == 2
    assert manifest["report"]["ids_contiguous_from_zero"] is False
    with pytest.raises(FileExistsError):
        parquet_io.write(tmp_path / "parquet", source)
    with pytest.raises(ValueError, match="budget"):
        parquet_io.read(tmp_path / "parquet", max_bytes=1)
    with (tmp_path / "parquet" / "qrels.parquet").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        parquet_io.read(tmp_path / "parquet")


def test_invalid_qrels_are_rejected_before_writing(tmp_path):
    pytest.importorskip("pyarrow")
    source = dataset.Dataset(
        [documents.CorpusDocument(0, "doc")], [EvaluationCase("q", "query", {9}, "test")], {0: "source"}, {}
    )
    with pytest.raises(ValueError, match="known positive"):
        parquet_io.write(tmp_path / "invalid", source)
    assert not (tmp_path / "invalid").exists()
