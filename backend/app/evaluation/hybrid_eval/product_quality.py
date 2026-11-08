"""Measure unchanged product routes on explicit public/synthetic local fixtures."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.evaluation import metrics
from backend.app.evaluation.hybrid_eval import dataset, product_corpus, reporting
from backend.app.models import local_clip


def latency(values: list[float]) -> dict[str, float | int | None]:
    """Summarize seconds as observed milliseconds, retaining sample counts."""
    if not values:
        return {"samples": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "samples": len(values),
        "p50": reporting.percentile(values, 0.50),
        "p95": reporting.percentile(values, 0.95),
        "p99": reporting.percentile(values, 0.99),
        "max": round(max(values, default=0) * 1000, 3),
    }


def _document_query(engine: Engine, query: str) -> tuple[list[int], dict[str, object], float]:
    from backend.app.retrieval import asset_search

    started = time.perf_counter()
    with Session(engine) as db:
        result = asset_search.search_assets(db, query, limit=10, asset_types=["document"])
        identifiers = [hit.file_id - 1 for hit in result.results]
    return identifiers, result.debug, time.perf_counter() - started


def documents(source: dataset.Dataset, directory: Path, *, repeats: int = 3, progress: bool = False) -> dict[str, Any]:
    """Measure full public product search with fresh SQL fixtures and warm repeats.

    Query failures retain empty rankings in quality denominators and explicit
    errors. The corpus is materialized once; import/index construction, warm
    queries and committed revision rebuilding have separate elapsed times.
    """
    from backend.app.db import models

    if repeats < 2 or repeats > 10:
        raise ValueError("Document benchmark requires 2..10 ranking repeats")
    engine = product_corpus.create_database(directory / "documents.db")
    try:
        started = time.perf_counter()
        product_corpus.seed_documents(engine, source, directory / "opaque-public-paths")
        materialization = time.perf_counter() - started
        if progress:
            print(f"Materialized {len(source.documents)} documents in {materialization:.2f}s", flush=True)
        _, _, cold = _document_query(engine, source.cases[0].query)
        if progress:
            print(f"Cold product query {cold:.2f}s", flush=True)
        rows, hashes, failures = [], [], []
        for repeat in range(repeats):
            times, rankings, scored = [], {}, []
            for case in source.cases:
                try:
                    ids, debug, elapsed = _document_query(engine, case.query)
                    if debug["final_fetch_calls"] != int(bool(ids)):
                        raise ValueError("Product final-fetch contract differs")
                except Exception as exc:  # Benchmark extension boundary: preserve failures in denominators.
                    failures.append({"repeat": repeat, "query_id": case.query_id, "error": repr(exc)})
                    ids, debug, elapsed = [], {}, None
                rankings[case.query_id] = ids
                scored.append((case, ids))
                if debug:
                    times.append(elapsed)
                rows.append(
                    {
                        "repeat": repeat,
                        "query_id": case.query_id,
                        "ranked_ids": ids,
                        "elapsed_ms": elapsed * 1000 if elapsed is not None else None,
                    }
                )
                if progress and len(scored) % 50 == 0:
                    checkpoint = {
                        "schema": "lentica-product-documents-partial-v1",
                        "completed_repeats": repeat,
                        "queries_in_current_repeat": len(scored),
                        "expected_queries_per_repeat": len(source.cases),
                        "partial_quality": metrics.aggregate_metrics(scored),
                        "cold_query_ms": cold * 1000,
                        "latency_ms": latency(times),
                        "per_query": rows,
                        "failures": failures,
                    }
                    temporary = directory / "documents-progress.tmp"
                    temporary.write_text(json.dumps(checkpoint) + "\n", encoding="utf-8")
                    temporary.replace(directory / "documents-progress.json")
                    print(f"Product repeat {repeat + 1}: {len(scored)}/{len(source.cases)} queries", flush=True)
            hashes.append(reporting.digest(rankings))
            if repeat == 0:
                quality = metrics.aggregate_metrics(scored)
                timing = latency(times)
            else:
                timing = {**timing, f"repeat_{repeat + 1}": latency(times)}
        # A committed update of an indexed title invalidates the real revision
        # cache, while preserving title contents and all relevance judgments.
        with engine.begin() as connection:
            connection.execute(update(models.Document).values(title=models.Document.title))
        _, _, rebuild = _document_query(engine, source.cases[0].query)
        return {
            "schema": "lentica-product-documents-v1",
            "source": source.provenance,
            "corpus_sha256": dataset.dataset_identity(source),
            "cases_sha256": dataset.case_identity(source.cases),
            "documents": len(source.documents),
            "queries": len(source.cases),
            "judgments": sum(len(case.relevant_asset_ids) for case in source.cases),
            "fixture_materialization_seconds": materialization,
            "first_query_including_cold_index_ms": cold * 1000,
            "revision_rebuild_query_ms": rebuild * 1000,
            "warm_latency_ms": timing,
            "quality": quality,
            "ranking_sha256": hashes,
            "repeat_rankings_identical": len(set(hashes)) == 1,
            "failures": failures,
            "per_query": rows,
            "scope": "Actual product TF-IDF/staged/final-SQL path; one abstract/chunk, opaque paths, historical test. "
            "No parsing/model/HTTP/browser/serving measurement; latency is serial shared-host observation.",
        }
    finally:
        engine.dispose()


class SyntheticEncoder(local_clip.Embedder):
    """Explicit nonsemantic query features for product scan/SQL scale controls."""

    def __init__(self, dimensions: int) -> None:
        self.dims = dimensions
        self._encoder_id = f"benchmark:synthetic-normalized-v1:{dimensions}"

    def embed_text(self, text: str) -> list[float]:
        import numpy as np

        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        values = np.random.default_rng(seed).normal(size=self.dims)
        return (values / np.linalg.norm(values)).tolist()


def scale(directory: Path, sizes: list[int], *, dimensions: int = 384, queries: int = 20) -> dict[str, Any]:
    """Measure real product vector decoding/scoring/fusion and selected SQL reads."""
    from backend.app.retrieval import asset_search

    if not sizes or any(type(size) is not int or not 1 <= size <= 10000 for size in sizes):
        raise ValueError("Require bounded positive corpus sizes")
    if not 2 <= dimensions <= 2048 or not 1 <= queries <= 100:
        raise ValueError("Invalid bounded vector/query configuration")
    reports = []
    for size in sizes:
        engine = product_corpus.create_database(directory / f"photos-{size}.db")
        try:
            started = time.perf_counter()
            identity = product_corpus.seed_vectors(engine, size, dimensions)
            build_seconds = time.perf_counter() - started
            encoder = SyntheticEncoder(dimensions)
            times, rows = [], []
            for index in range(queries):
                started = time.perf_counter()
                with Session(engine) as db:
                    result = asset_search.search_assets(
                        db,
                        f"synthetic-control-{index % 10}",
                        limit=10,
                        asset_types=["photo"],
                        embedders=lambda: (encoder, encoder),
                    )
                times.append(time.perf_counter() - started)
                rows.append(
                    {
                        "query": index % 10,
                        "ranked_ids": [row.file_id for row in result.results],
                        "scanned_vectors": result.debug["vector_scoring"]["scanned_vectors"],
                        "candidate_count": result.debug["candidate_count"],
                        "final_fetch_calls": result.debug["final_fetch_calls"],
                        "preparation_ms": result.debug["candidate_preparation_ms"],
                    }
                )
            reports.append(
                {
                    "files": size,
                    "vectors": size * 2,
                    "dimensions": dimensions,
                    "encoder_id": identity,
                    "build_seconds": build_seconds,
                    "latency_ms": latency(times),
                    "elapsed_ms": [value * 1000 for value in times],
                    "per_query": rows,
                    "repeat_rankings_identical": all(
                        row["ranked_ids"] == rows[index % 10]["ranked_ids"] for index, row in enumerate(rows)
                    ),
                }
            )
        finally:
            engine.dispose()
    return {
        "schema": "lentica-product-scale-v1",
        "scope": "Synthetic vectors, real product CPU scan; no semantic quality.",
        "scales": reports,
    }


def ingestion(source: dataset.Dataset, directory: Path, *, count: int = 100) -> dict[str, Any]:
    """Measure parser/index/re-scan/change handling on copied public abstracts."""
    from backend.app.db import models
    from backend.app.ingestion import parsers, scanner

    if not 1 <= count <= min(1000, len(source.documents)):
        raise ValueError("Invalid public ingestion sample size")
    originals = directory / "public-text"
    originals.mkdir(parents=True, exist_ok=False)
    for index, document in enumerate(source.documents[:count]):
        (originals / f"item-{index:06d}.txt").write_text(document.text, encoding="utf-8")
    engine = product_corpus.create_database(directory / "ingestion.db")
    try:
        with Session(engine) as db:
            selected = models.Source(name="public ingestion", root_path=str(originals), kind="documents")
            db.add(selected)
            db.commit()
            source_id = selected.id
        phases = []
        for phase in ("initial", "unchanged", "one_changed"):
            if phase == "one_changed":
                changed = originals / "item-000000.txt"
                changed.write_text(changed.read_text() + "\nexplicit benchmark change marker", encoding="utf-8")
            before = {path.name: parsers.sha256_file(path) for path in originals.iterdir()}
            started = time.perf_counter()
            with Session(engine) as db:
                job = scanner.scan_documents(db, source_id=source_id, max_files=None)
                values = {
                    key: getattr(job, key)
                    for key in ("status", "total_files", "indexed_files", "skipped_files", "failed_files", "error")
                }
                chunks = db.scalar(select(func.count(models.Chunk.id)))
            elapsed = time.perf_counter() - started
            after = {path.name: parsers.sha256_file(path) for path in originals.iterdir()}
            phases.append(
                {
                    "phase": phase,
                    **values,
                    "chunks": chunks,
                    "elapsed_seconds": elapsed,
                    "files_per_second": count / elapsed,
                    "source_bytes_preserved": before == after,
                }
            )
        return {
            "schema": "lentica-product-ingestion-v1",
            "scope": "Actual text parser/scanner/SQLite commits on first bounded public abstracts; no photo/model throughput.",
            "corpus_sha256": dataset.dataset_identity(source),
            "files": count,
            "source_selection": "first corpus-ID-sorted public abstracts; no query/label selection",
            "phases": phases,
        }
    finally:
        engine.dispose()
