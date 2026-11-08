"""Explicit synthetic mixed-ranking and real ASGI product boundary measurements."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from sqlalchemy import insert, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.evaluation.hybrid_eval import dataset, product_corpus, product_quality, reporting


def _seed_control_documents(engine: Engine, *, count: int, chunks_in_first: int = 1) -> None:
    """Create identical synthetic evidence; labels are not semantic model inputs."""
    from backend.app.db import models

    with engine.begin() as connection:
        connection.execute(
            insert(models.Source),
            [{"id": 2, "name": "document controls", "root_path": "/control-documents", "kind": "documents"}],
        )
        for index in range(count):
            identifier = 100 + index
            connection.execute(
                insert(models.File),
                [
                    {
                        "id": identifier,
                        "source_id": 2,
                        "canonical_path": f"/control-documents/item-{index:06d}.txt",
                        "windows_path": f"item-{index:06d}.txt",
                        "file_type": "txt",
                        "ingestion_status": "indexed",
                    }
                ],
            )
            connection.execute(
                insert(models.Document), [{"file_id": identifier, "title": "control", "parser": "synthetic"}]
            )
            connection.execute(
                insert(models.Chunk),
                [
                    {
                        "file_id": identifier,
                        "text": "needle identical evidence",
                        "page_or_segment": f"seg.{chunk}",
                        "chunk_hash": hashlib.sha256(str(chunk).encode()).hexdigest(),
                    }
                    for chunk in range(chunks_in_first if index == 0 else 1)
                ],
            )


def mixed(directory: Path) -> dict[str, Any]:
    """Expose branch multiplicity and chunk-window competition without quality claims."""
    from backend.app.db import models
    from backend.app.retrieval import asset_search

    engine = product_corpus.create_database(directory / "mixed.db")
    try:
        product_corpus.seed_vectors(engine, 10, 8)
        encoder = product_quality.SyntheticEncoder(8)
        encoded = json.dumps(encoder.embed_text("needle"))
        with engine.begin() as connection:
            connection.execute(update(models.PhotoVector).values(vector_json=encoded))
            connection.execute(
                insert(models.PhotoVector),
                [
                    {
                        "file_id": identifier,
                        "namespace": "caption",
                        "dims": 8,
                        "encoder_id": encoder.encoder_id,
                        "vector_json": encoded,
                    }
                    for identifier in range(1, 11)
                ],
            )
        _seed_control_documents(engine, count=10)
        with Session(engine) as db:
            combined = asset_search.search_assets(
                db, "needle", limit=10, asset_types=["photo", "document"], embedders=lambda: (encoder, encoder)
            )
            doc_only = asset_search.search_assets(db, "needle", limit=10, asset_types=["document"])
        multiplicity = {
            "input_photos": 10,
            "input_documents": 10,
            "photo_branches": ["visual", "caption", "context"],
            "document_branches": ["tfidf"],
            "mixed_selected_photos": len(combined.photos),
            "mixed_selected_documents": len({row.file_id for row in combined.documents}),
            "document_scope_selected": len(doc_only.results),
            "mixed_scores": [{"type": row.asset_type, "score": row.score} for row in combined.results],
            "scope": "Identical synthetic evidence, fixed existing weights. Demonstrates branch-count effects; not cross-modal relevance judgments.",
        }
    finally:
        engine.dispose()
    engine = product_corpus.create_database(directory / "chunk-window.db")
    try:
        _seed_control_documents(engine, count=20, chunks_in_first=200)
        with Session(engine) as db:
            result = asset_search.search_assets(db, "needle identical evidence", limit=20, asset_types=["document"])
        window = {
            "input_documents": 20,
            "input_chunks": 219,
            "requested_files": 20,
            "returned_files": len(result.results),
            "branch_window": result.debug["branch_window"],
            "returned_supporting_chunks": len(result.documents),
        }
    finally:
        engine.dispose()
    return {
        "schema": "lentica-product-mixed-controls-v1",
        "multiplicity": multiplicity,
        "chunk_window": window,
        "scope": "Mechanism stress controls, not public-data accuracy or tuning evidence.",
    }


def api(source: dataset.Dataset, directory: Path, *, count: int = 30) -> dict[str, Any]:
    """Run actual V2/chat ASGI requests with isolated SQL and no server lifespan.

    This covers routing, validation, serialization and local extractive replies,
    not TCP/browser latency, model serving or background controller startup.
    """
    import httpx

    from backend.app import main
    from backend.app.db import session

    engine = product_corpus.create_database(directory / "api.db")
    product_corpus.seed_documents(engine, source, directory / "opaque-public-paths")

    def database():
        with Session(engine) as db:
            yield db

    previous = dict(main.app.dependency_overrides)
    main.app.dependency_overrides[session.get_db] = database

    async def requests():
        routes = {}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://benchmark"
        ) as client:
            # Explicit warmup; full product cold-index latency is measured by
            # documents(), rather than folded into the steady ASGI distribution.
            await client.post("/api/v2/search", json={"query": source.cases[0].query, "asset_types": ["document"]})
            for route in ("/api/v2/search", "/api/chat"):
                times, statuses, errors, rankings = [], {}, [], {}
                for case in source.cases[:count]:
                    body = (
                        {"query": case.query, "asset_types": ["document"], "limit": 10}
                        if route.endswith("search")
                        else {"question": case.query, "limit": 10}
                    )
                    started = time.perf_counter()
                    response = await client.post(route, json=body)
                    times.append(time.perf_counter() - started)
                    statuses[str(response.status_code)] = statuses.get(str(response.status_code), 0) + 1
                    if response.status_code != 200:
                        errors.append({"query_id": case.query_id, "status": response.status_code})
                        continue
                    payload = response.json()
                    hits = payload["results"] if route.endswith("search") else payload["sources"]
                    rankings[case.query_id] = list(dict.fromkeys(row["file_id"] for row in hits))
                    if route.endswith("chat") and payload["answer_mode"] != "extractive_local":
                        errors.append({"query_id": case.query_id, "error": "Unexpected answer mode"})
                routes[route] = {
                    "requests": len(times),
                    "statuses": statuses,
                    "latency_ms": product_quality.latency(times),
                    "errors": errors,
                    "rankings": rankings,
                }
            invalid = await client.post("/api/chat", json={"question": "control", "limit": 101})
            blank = await client.post("/api/v2/search", json={"query": " ", "asset_types": ["document"]})
            return {
                "routes": routes,
                "invalid_limit_status": invalid.status_code,
                "blank_query_status": blank.status_code,
            }

    try:
        return {
            "schema": "lentica-product-asgi-v1",
            **asyncio.run(requests()),
            "scope": "Actual in-process ASGI/API and SQL; no TCP server, lifespan, browser or concurrent-load measurement.",
            "corpus_sha256": dataset.dataset_identity(source),
            "cases": [case.query_id for case in source.cases[:count]],
            "environment": reporting.environment(),
        }
    finally:
        main.app.dependency_overrides.clear()
        main.app.dependency_overrides.update(previous)
        engine.dispose()
