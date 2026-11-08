"""Verify public-ID mapping, failure denominators and real isolated measurements."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db import models
from backend.app.evaluation import dataset as cases
from backend.app.evaluation.hybrid_eval import dataset, product_controls, product_corpus, product_quality
from backend.app.retrieval.hybrid import documents


@pytest.fixture
def source():
    return dataset.Dataset(
        documents=[
            documents.CorpusDocument(0, "Neutrino scattering measurements from detectors.", "Particle physics"),
            documents.CorpusDocument(2, "Cellular protein expression and gene regulation.", "Cell biology"),
        ],
        cases=[
            cases.EvaluationCase("first", "neutrino scattering", {0}, "public-control"),
            cases.EvaluationCase("second", "protein expression", {2}, "public-control"),
        ],
        source_ids={0: "source-a", 2: "source-c"},
        provenance={"kind": "synthetic test control"},
    )


def test_documents_map_noncontiguous_ids_and_repeat_real_product(source, tmp_path):
    report = product_quality.documents(source, tmp_path / "documents", repeats=2)
    assert report["quality"]["overall"]["recall_at_10"] == 1
    assert report["repeat_rankings_identical"]
    assert report["warm_latency_ms"]["samples"] == 2
    assert not report["failures"]
    assert report["per_query"][0]["ranked_ids"][0] == 0
    assert report["per_query"][1]["ranked_ids"][0] == 2
    assert report["first_query_including_cold_index_ms"] > 0
    assert report["revision_rebuild_query_ms"] > 0


def test_failed_queries_remain_in_quality_denominator(source, tmp_path, monkeypatch):
    assert product_quality.latency([]) == {"samples": 0, "p50": None, "p95": None, "p99": None, "max": None}
    original = product_quality._document_query

    def failing(engine, query):
        if query == "protein expression":
            raise RuntimeError("explicit benchmark provider failure")
        return original(engine, query)

    monkeypatch.setattr(product_quality, "_document_query", failing)
    report = product_quality.documents(source, tmp_path / "failed", repeats=2)
    assert report["quality"]["queries"] == 2
    assert report["quality"]["overall"]["recall_at_10"] == 0.5
    assert len(report["failures"]) == 2
    assert report["per_query"][1]["elapsed_ms"] is None
    assert report["warm_latency_ms"]["samples"] == 1


def test_scale_executes_all_vectors_and_reports_determinism(tmp_path):
    report = product_quality.scale(tmp_path, [4, 8], dimensions=8, queries=12)
    for row in report["scales"]:
        assert row["repeat_rankings_identical"]
        assert all(query["scanned_vectors"] == row["files"] * 2 for query in row["per_query"])
        assert all(query["final_fetch_calls"] == 1 for query in row["per_query"])


def test_ingestion_preserves_public_sources_and_replaces_only_change(source, tmp_path):
    report = product_quality.ingestion(source, tmp_path / "ingestion", count=2)
    initial, unchanged, changed = report["phases"]
    assert initial["indexed_files"] == 2
    assert unchanged["indexed_files"] == 0 and unchanged["skipped_files"] == 2
    assert changed["indexed_files"] == 1 and changed["skipped_files"] == 1
    assert all(row["source_bytes_preserved"] and row["status"] == "succeeded" for row in report["phases"])
    assert all(row["chunks"] == 2 for row in report["phases"])


def test_seed_contains_opaque_paths_and_no_judgment_features(source, tmp_path):
    engine = product_corpus.create_database(tmp_path / "fixture.db")
    try:
        product_corpus.seed_documents(engine, source, tmp_path / "opaque")
        with Session(engine) as db:
            files = list(db.scalars(select(models.File).order_by(models.File.id)))
            assert [file.id for file in files] == [1, 3]
            assert files[0].windows_path == "item-000000.txt"
            assert not (tmp_path / "opaque").exists()
        with pytest.raises(FileExistsError):
            product_corpus.create_database(tmp_path / "fixture.db")
    finally:
        engine.dispose()


def test_mixed_controls_expose_current_branch_and_window_policy(tmp_path):
    report = product_controls.mixed(tmp_path)
    assert report["multiplicity"]["mixed_selected_photos"] == 10
    assert report["multiplicity"]["mixed_selected_documents"] == 0
    assert report["multiplicity"]["document_scope_selected"] == 10
    assert report["chunk_window"]["returned_files"] < report["chunk_window"]["requested_files"]


def test_asgi_controls_use_isolated_corpus_and_preserve_validation(source, tmp_path):
    report = product_controls.api(source, tmp_path, count=2)
    for route in report["routes"].values():
        assert route["statuses"] == {"200": 2}
        assert not route["errors"]
        assert route["rankings"]["first"][0] == 1
        assert route["rankings"]["second"][0] == 3
    assert report["invalid_limit_status"] == 422
    assert report["blank_query_status"] == 400


def test_imports_and_help_do_not_initialize_product_state(tmp_path):
    environment = {**os.environ, "LENTICA_DATA_DIR": str(tmp_path / "untouched")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from backend.app.evaluation.hybrid_eval import product_corpus, product_quality, product_controls; "
            "import sys; assert 'backend.app.db.session' not in sys.modules",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "backend.app.evaluation.product_benchmark", "--help"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "untouched").exists()
