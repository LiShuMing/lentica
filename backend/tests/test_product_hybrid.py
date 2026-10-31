"""Real temporary-SQL/API integration through the shared hybrid coordinator."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict

import httpx
import pytest
from sqlalchemy import delete, event, select

from backend.app.db import models
from backend.app.retrieval import asset_search, asset_store, photo_search
from backend.app.retrieval.hybrid import engine, stages
from backend.tests.test_search_consistency import FixtureEncoder, add_photo


def add_document(db, source, name, texts):
    file = models.File(
        source_id=source.id,
        canonical_path=f"/fixture/{name}.md",
        windows_path=f"{name}.md",
        file_type="md",
        ingestion_status="indexed",
    )
    db.add(file)
    db.flush()
    db.add(models.Document(file_id=file.id, title=name, parser="fixture", page_count=1))
    for index, text in enumerate(texts):
        db.add(
            models.Chunk(
                file_id=file.id,
                text=text,
                page_or_segment=f"seg.{index + 1}",
                chunk_hash=hashlib.sha256(f"{index}:{text}".encode()).hexdigest(),
            )
        )
    db.flush()
    return file


@pytest.fixture
def source(db):
    row = models.Source(name="mixed fixture", root_path="/fixture", kind="documents")
    db.add(row)
    db.commit()
    return row


def embeddings():
    encoder = FixtureEncoder()
    return encoder, encoder


def test_mixed_global_resource_limit_and_disjoint_unit_namespaces(db, source):
    photo = add_photo(db, source, "red-photo", [1, 0], FixtureEncoder())
    document = add_document(db, source, "red-doc", ["red object"])
    db.commit()
    result = asset_search.search_assets(
        db, "red object", limit=2, asset_types=["photo", "document"], embedders=embeddings
    )
    assert {hit.file_id for hit in result.results} == {photo.id, document.id}
    assert [hit.rank for hit in result.results] == [1, 2]
    assert [hit.score for hit in result.results] == sorted((hit.score for hit in result.results), reverse=True)
    chunk = db.scalar(select(models.Chunk))
    assert photo.id == chunk.id  # real colliding database IDs
    assert {unit for hit in result.results for unit in hit.unit_ids} == {photo.id * 2, chunk.id * 2 + 1}
    assert result.debug["final_fetch_calls"] == 1
    assert result.debug["final_file_ids"] == [hit.file_id for hit in result.results]
    one = asset_search.search_assets(db, "red object", limit=1, asset_types=["photo", "document"], embedders=embeddings)
    assert len(one.results) == 1 and len(one.photos) + len({row.file_id for row in one.documents}) == 1
    repeat = asset_search.search_assets(
        db, "red object", limit=2, asset_types=["photo", "document"], embedders=embeddings
    )
    assert [asdict(hit) for hit in repeat.results] == [asdict(hit) for hit in result.results]


def test_document_grouping_reads_exact_selected_support_and_needs_no_image_model(db, source):
    first = add_document(db, source, "first", ["red object"] * 5)
    second = add_document(db, source, "second", ["red object"])
    db.commit()

    def forbidden():
        raise AssertionError("document scope must not construct image models")

    result = asset_search.search_assets(db, "red", limit=2, asset_types=["document"], embedders=forbidden)
    assert {hit.file_id for hit in result.results} == {first.id, second.id}
    assert all(1 <= len(hit.documents) <= 2 for hit in result.results)
    assert set(result.debug["final_chunk_ids"]) == {row.chunk_id for row in result.documents}
    assert len(result.debug["final_chunk_ids"]) == 3
    assert result.debug["final_fetch_calls"] == 1
    assert all(row.text == "red object" and row.page_or_segment.startswith("seg.") for row in result.documents)

    # The cold index reads corpus features. With that cache warm, only final
    # supporting Chunk entities may be loaded, rather than every chunk of a file.
    db.expunge_all()
    loaded = []

    def chunk_loaded(row, _context):
        loaded.append(row.id)

    event.listen(models.Chunk, "load", chunk_loaded)
    try:
        warm = asset_search.search_assets(db, "red", limit=2, asset_types=["document"], embedders=forbidden)
    finally:
        event.remove(models.Chunk, "load", chunk_loaded)
    assert set(loaded) == set(warm.debug["final_chunk_ids"]) and len(loaded) == 3


def test_only_final_photo_entities_are_hydrated_and_empty_search_does_not_fetch(db, source):
    for index in range(7):
        add_photo(db, source, f"red-{index}", [1, 0], FixtureEncoder())
    db.commit()
    db.expunge_all()
    files, images = [], []

    def file_loaded(row, _context):
        files.append(row.id)

    def image_loaded(row, _context):
        images.append(row.file_id)

    event.listen(models.File, "load", file_loaded)
    event.listen(models.Image, "load", image_loaded)
    try:
        result = asset_search.search_assets(db, "red", limit=1, asset_types=["photo"], embedders=embeddings)
    finally:
        event.remove(models.File, "load", file_loaded)
        event.remove(models.Image, "load", image_loaded)
    assert files == images == [result.results[0].file_id]
    assert result.debug["candidate_count"] == 7 and result.debug["final_fetch_calls"] == 1
    empty = asset_search.search_assets(db, "red", asset_types=["document"])
    assert not empty.results and empty.debug["final_fetch_calls"] == 0


def test_photo_only_policy_matches_legacy_without_using_legacy_fusion_in_main(db, source, monkeypatch):
    encoder = FixtureEncoder()
    for index, vector in enumerate(([1, 0], [0.8, 0.2], [0.1, 1])):
        add_photo(db, source, f"photo-{index}", vector, encoder)
    db.commit()
    expected = photo_search.PhotoSearch(db, encoder).search("red", limit=3)[1]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("main flow cannot call legacy weighted_rrf")

    monkeypatch.setattr(photo_search, "weighted_rrf", forbidden)
    result = asset_search.search_assets(db, "red", limit=3, asset_types=["photo"], embedders=embeddings)
    assert [asdict(hit) for hit in result.photos] == [asdict(hit) for hit in expected]
    assert result.debug["pipeline"] == "staged-hybrid"


def test_main_photo_constraints_precede_window(db, source):
    for index in range(65):
        add_photo(db, source, f"wrong-{index}", [1, 0], FixtureEncoder(), year=2025)
    target = add_photo(db, source, "target", [0.01, 1], FixtureEncoder(), year=2024, asset_type="screenshot")
    db.commit()
    result = asset_search.search_assets(db, "2024", limit=1, asset_types=["screenshot"], embedders=embeddings)
    assert [hit.file_id for hit in result.results] == [target.id]
    assert result.results[0].asset_type == "screenshot"
    source.enabled = False
    db.commit()
    assert not asset_search.search_assets(db, "red", asset_types=["photo"], embedders=embeddings).results


def test_document_changes_and_rollback_are_visible_through_main(db, source):
    add_document(db, source, "knowledge", ["committed alpha"])
    db.commit()

    def search(text):
        return asset_search.search_assets(db, text, limit=1, asset_types=["document"]).documents

    assert search("alpha")[0].text == "committed alpha"
    chunk = db.scalar(select(models.Chunk))
    chunk.text = "uncommitted beta"
    db.flush()
    assert search("beta")[0].text == "uncommitted beta"
    db.rollback()
    assert search("alpha")[0].text == "committed alpha"
    chunk.text = "replacement gamma"
    db.commit()
    assert search("gamma")[0].text == "replacement gamma"
    source.enabled = False
    db.commit()
    assert not search("gamma")


@pytest.mark.parametrize("context", ["{invalid", "[]", "null"])
def test_main_keeps_legacy_malformed_context_fallback(db, source, context):
    file = add_photo(db, source, "legacy", [1, 0], FixtureEncoder())
    db.scalar(select(models.Image)).context_json = context
    db.commit()
    result = asset_search.search_assets(db, "red", limit=1, asset_types=["photo"], embedders=embeddings)
    assert result.results[0].file_id == file.id and result.results[0].asset_type == "photo"


def test_main_ocr_does_not_need_vectors_but_semantic_route_rejects_encoder_drift(db, source):
    file = add_photo(db, source, "old-vector", [1, 0], FixtureEncoder("b"), asset_type="screenshot")
    image = db.scalar(select(models.Image))
    image.ocr_text = "ERR_12345"
    db.commit()
    assert not asset_search.search_assets(db, "red", asset_types=["screenshot"], embedders=embeddings).results
    db.execute(delete(models.PhotoVector))
    db.commit()
    result = asset_search.search_assets(db, "截图 ERR_12345", limit=1, asset_types=["screenshot"], embedders=embeddings)
    assert [hit.file_id for hit in result.results] == [file.id]
    assert result.photos[0].ocr_text == "ERR_12345"
    assert result.debug["retriever_counts"]["ocr_lexical"] == 1


def test_main_person_and_location_filter_before_branch_window(db, source):
    for index in range(65):
        add_photo(db, source, f"distractor-{index}", [1, 0], FixtureEncoder())
    target = add_photo(db, source, "target", [0.01, 1], FixtureEncoder())
    image = db.scalar(select(models.Image).where(models.Image.file_id == target.id))
    image.location = "上海"
    image.context_json = json.dumps({"asset_type": "photo"})
    cluster = models.PersonCluster(label="Alice", display_name="Alice")
    db.add(cluster)
    db.flush()
    db.add(models.FaceDetection(file_id=target.id, cluster_id=cluster.id, bbox_json="[0,0,1,1]"))
    db.commit()
    result = asset_search.search_assets(db, "Alice 上海", limit=1, asset_types=["photo"], embedders=embeddings)
    assert [hit.file_id for hit in result.results] == [target.id]
    assert "Alice" in result.photos[0].matched_terms


@pytest.mark.parametrize("failure", ["chunk", "source"])
def test_final_missing_or_disabled_records_fail_closed(db, source, monkeypatch, failure):
    add_document(db, source, "doc", ["red object"])
    db.commit()
    original = asset_store.SqlAssetStore.fetch_evidence

    def invalidate(store, hits):
        if failure == "chunk":
            store.db.execute(delete(models.Chunk))
        else:
            source.enabled = False
            store.db.flush()
        return original(store, hits)

    monkeypatch.setattr(asset_store.SqlAssetStore, "fetch_evidence", invalidate)
    with pytest.raises(engine.HydrationError):
        asset_search.search_assets(db, "red", asset_types=["document"])


def test_v2_legacy_and_chat_reach_actual_shared_engine(db, source, monkeypatch):
    from backend.app.api import routes
    from backend.app.db.session import get_db
    from backend.app.main import app

    add_photo(db, source, "red-photo", [1, 0], FixtureEncoder())
    add_document(db, source, "red-doc", ["red object evidence"])
    db.commit()
    monkeypatch.setattr(routes, "_photo_search_embedders", lambda _request: embeddings())
    calls = []
    original = engine.HybridSearch.search

    def track(searcher, query, config):
        calls.append(tuple(branch.name for branch in config.branches))
        return original(searcher, query, config)

    monkeypatch.setattr(engine.HybridSearch, "search", track)
    app.dependency_overrides[get_db] = lambda: db

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v2/search", json={"query": "red", "limit": 2, "asset_types": ["photo", "document"], "debug": True}
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert len(body["results"]) == 2 and len(body["hits"]) == len(body["document_hits"]) == 1
            assert body["debug"]["pipeline"] == "staged-hybrid" and body["debug"]["final_fetch_calls"] == 1
            legacy = await client.post("/api/photos/search", json={"query": "red", "limit": 1})
            assert legacy.status_code == 200 and len(legacy.json()["hits"]) == 1
            chat = await client.post("/api/chat", json={"question": "red evidence", "limit": 1})
            assert chat.status_code == 200 and chat.json()["sources"][0]["excerpt"] == "red object evidence"
            assert "retrieve: staged_hybrid_document_tfidf" in chat.json()["trace"]
            invalid = await client.post("/api/chat", json={"question": "red", "limit": 0})
            assert invalid.status_code == 422

    try:
        asyncio.run(check())
    finally:
        app.dependency_overrides.clear()
    assert len(calls) == 3 and "document.tfidf" in calls[0] and calls[-1] == ("document.tfidf",)


def test_evidence_store_extension_is_exclusive_and_missing_records_raise():
    from backend.app.retrieval.hybrid import contracts, units

    unit = units.RetrievalUnit(1, 42, "text")
    with pytest.raises(ValueError, match="Choose one"):
        stages.StagedSearch({}, units.UnitCatalog([unit]), store=object(), evidence_store=object())

    class Missing:
        def fetch_evidence(self, hits):
            assert hits[0].asset_id == 42 and hits[0].evidence[0].unit.unit_id == 1
            return {}

    from backend.app.retrieval.hybrid import adapters

    searcher = stages.StagedSearch(
        {"text": adapters.RankedListRetriever([1])}, units.UnitCatalog([unit]), evidence_store=Missing()
    )
    with pytest.raises(engine.HydrationError):
        searcher.search(
            contracts.SearchQuery("x"), stages.StageConfig(contracts.SearchConfig((contracts.Branch("text"),), top_k=1))
        )
