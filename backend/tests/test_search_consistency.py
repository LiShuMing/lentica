from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime

import httpx
import pytest
from sqlalchemy import select

from backend.app.db.models import FaceDetection, File, Image, PersonCluster, Source
from backend.app.evaluation.dataset import EvaluationCase
from backend.app.evaluation.metrics import metrics_for_case
from backend.app.ingestion.scanner import scan_documents
from backend.app.models.encoder_space import space_id
from backend.app.models.local_clip import Embedder
from backend.app.retrieval.photo_search import PhotoSearch, upsert_photo_vector
from backend.app.retrieval.query_plan import build_retrieval_plan
from backend.app.retrieval.service import search_knowledge


class FixtureEncoder(Embedder):
    dims = 2

    def __init__(self, revision="a"):
        self._encoder_id = space_id("clip", "fixture", revision, dimensions=2, encoding="test-v1")

    def embed_text(self, text):
        return [1.0, 0.0]


def add_photo(db, source, name, vector, encoder, *, year=2025, asset_type="photo"):
    file = File(
        source_id=source.id,
        canonical_path=f"/fixture/{name}.png",
        windows_path=f"{name}.png",
        file_type="png",
        ingestion_status="indexed",
    )
    db.add(file)
    db.flush()
    db.add(
        Image(
            file_id=file.id,
            exif_time=datetime(year, 1, 1),
            caption="red object",
            context_json=json.dumps({"asset_type": asset_type}),
        )
    )
    upsert_photo_vector(db, file.id, "clip:image", vector, encoder_id=encoder.encoder_id)
    db.flush()
    return file


@pytest.fixture
def photo_source(db):
    source = Source(name="Photos", root_path="/fixture", kind="images")
    db.add(source)
    db.commit()
    return source


def test_year_filter_precedes_candidate_window(db, photo_source):
    encoder = FixtureEncoder()
    for index in range(65):
        add_photo(db, photo_source, f"wrong-{index}", [1, 0], encoder)
    target = add_photo(db, photo_source, "correct", [0.01, 1], encoder, year=2024)
    db.commit()
    _, hits = PhotoSearch(db, encoder).search("2024", limit=1)
    assert [hit.file_id for hit in hits] == [target.id]


def test_explicit_people_do_not_fall_back_to_unrelated_photos(db, photo_source):
    encoder = FixtureEncoder()
    for name in ("Alice", "Bob"):
        file = add_photo(db, photo_source, name, [1, 0], encoder)
        cluster = PersonCluster(label=name, display_name=name)
        db.add(cluster)
        db.flush()
        db.add(FaceDetection(file_id=file.id, cluster_id=cluster.id, bbox_json="[0,0,1,1]"))
    db.commit()
    _, hits = PhotoSearch(db, encoder).search("Alice Bob")
    assert hits == []


@pytest.mark.parametrize("revision,vector", [("b", [1, 0]), ("a", [1, 0, 0])])
def test_incompatible_spaces_are_excluded(db, photo_source, revision, vector):
    stored = FixtureEncoder(revision)
    add_photo(db, photo_source, "incompatible", vector, stored)
    db.commit()
    assert PhotoSearch(db, FixtureEncoder()).search("red")[1] == []


def test_unknown_legacy_semantic_vectors_are_not_silently_trusted(db, photo_source):
    encoder = FixtureEncoder()
    file = add_photo(db, photo_source, "legacy", [1, 0], encoder)
    from backend.app.db.models import PhotoVector

    row = db.scalar(select(PhotoVector).where(PhotoVector.file_id == file.id))
    row.encoder_id = None
    db.commit()
    assert PhotoSearch(db, encoder).search("red")[1] == []


def test_plan_flags_and_explicit_asset_type_are_executed(db, photo_source):
    encoder = FixtureEncoder()
    screenshot = add_photo(db, photo_source, "screenshot", [1, 0], encoder, asset_type="screenshot")
    add_photo(db, photo_source, "photo", [1, 0], encoder)
    db.commit()
    plan = build_retrieval_plan(db, "red", expand=lambda q: q, asset_types=["screenshot"])
    plan = replace(plan, use_visual=False, use_caption_dense=False, use_context_dense=False)
    _, hits, debug = PhotoSearch(db, encoder).search_with_debug("red", plan=plan)
    assert [hit.file_id for hit in hits] == [screenshot.id]
    assert set(debug["retriever_counts"]) == {"metadata"}


def test_document_index_reused_and_invalidated(db, tmp_path, monkeypatch):
    from backend.app.retrieval import service

    source = Source(name="Documents", root_path=str(tmp_path), kind="documents")
    db.add(source)
    db.commit()
    path = tmp_path / "document.md"
    path.write_text("materialized view evidence")
    scan_documents(db, source_id=source.id)
    builds = []
    original = service._build_index

    def tracked(session):
        builds.append(True)
        return original(session)

    monkeypatch.setattr(service, "_build_index", tracked)
    assert search_knowledge(db, "materialized view")
    assert search_knowledge(db, "evidence")
    assert len(builds) == 1
    path.write_text("replacement knowledge")
    scan_documents(db, source_id=source.id)
    assert search_knowledge(db, "replacement knowledge")[0].text == "replacement knowledge"
    assert len(builds) == 2
    source.enabled = False
    db.commit()
    assert search_knowledge(db, "replacement knowledge") == []


def test_uncommitted_cache_does_not_survive_rollback(db, tmp_path):
    from backend.app.db.models import Chunk

    source = Source(name="Rollback", root_path=str(tmp_path), kind="documents")
    db.add(source)
    db.commit()
    (tmp_path / "doc.md").write_text("committed alpha")
    scan_documents(db, source_id=source.id)
    chunk = db.scalar(select(Chunk))
    chunk.text = "uncommitted beta"
    db.flush()
    assert search_knowledge(db, "beta")[0].text == "uncommitted beta"
    db.rollback()
    assert search_knowledge(db, "alpha")[0].text == "committed alpha"


def test_v2_document_branch_has_citations_without_loading_models(db, tmp_path, monkeypatch):
    from backend.app.api import routes
    from backend.app.db.session import get_db
    from backend.app.main import app

    source = Source(name="API docs", root_path=str(tmp_path), kind="documents")
    db.add(source)
    db.commit()
    (tmp_path / "doc.md").write_text("materialized view evidence")
    scan_documents(db, source_id=source.id)

    def no_models(*_):
        raise AssertionError("Document-only search must not load image models")

    monkeypatch.setattr(routes, "_photo_search_embedders", no_models)
    app.dependency_overrides[get_db] = lambda: db

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v2/search",
                json={
                    "query": "materialized view",
                    "asset_types": ["document"],
                    "debug": True,
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert body["hits"] == []
            assert body["document_hits"][0]["page_or_segment"] == "seg.1"
            assert body["document_hits"][0]["excerpt"] == "materialized view evidence"
            invalid = await client.post("/api/v2/search", json={"query": "x", "asset_types": ["unknown"]})
            assert invalid.status_code == 422

    try:
        asyncio.run(check())
    finally:
        app.dependency_overrides.clear()


def test_recall_is_fraction_and_hit_is_binary():
    case = EvaluationCase("query", "fixture", {1, 2, 3, 4}, "test")
    values = metrics_for_case(case, [1])
    assert values["recall_at_5"] == 0.25
    assert values["hit_at_5"] == 1.0


def test_ocr_lexical_branch_uses_exact_identifier(db, photo_source):
    encoder = FixtureEncoder()
    first = add_photo(db, photo_source, "exact", [0.01, 1], encoder)
    second = add_photo(db, photo_source, "other", [1, 0], encoder)
    image = db.scalar(select(Image).where(Image.file_id == first.id))
    image.ocr_text = "ERR_12345"
    db.commit()
    _, hits, debug = PhotoSearch(db, encoder).search_with_debug("截图 ERR_12345")
    assert debug["retriever_counts"]["ocr_lexical"] == 1
    assert first.id in [hit.file_id for hit in hits]


def test_location_filter_precedes_rank_fusion(db, photo_source):
    encoder = FixtureEncoder()
    wrong = add_photo(db, photo_source, "wrong", [1, 0], encoder)
    correct = add_photo(db, photo_source, "correct", [0.01, 1], encoder)
    db.scalar(select(Image).where(Image.file_id == correct.id)).location = "上海"
    db.commit()
    _, hits = PhotoSearch(db, encoder).search("上海", limit=1)
    assert [hit.file_id for hit in hits] == [correct.id]


def test_candidates_beyond_retriever_window_are_not_emitted(db, photo_source):
    encoder = FixtureEncoder()
    files = [add_photo(db, photo_source, f"candidate-{index}", [1, 0], encoder) for index in range(65)]
    db.commit()
    _, hits, debug = PhotoSearch(db, encoder).search_with_debug("red", limit=60)
    assert debug["candidate_count"] == 65
    assert len(hits) == 60
    assert [hit.file_id for hit in hits] == [file.id for file in files[:60]]


def test_ocr_lexical_does_not_require_compatible_vectors(db, photo_source):
    from sqlalchemy import delete
    from backend.app.db.models import PhotoVector

    encoder = FixtureEncoder()
    file = add_photo(db, photo_source, "text-only", [1, 0], encoder)
    db.scalar(select(Image).where(Image.file_id == file.id)).ocr_text = "ERR_12345"
    db.execute(delete(PhotoVector).where(PhotoVector.file_id == file.id))
    db.commit()
    _, hits, debug = PhotoSearch(db, encoder).search_with_debug("截图 ERR_12345")
    assert [hit.file_id for hit in hits] == [file.id]
    assert hits[0].namespace_scores == {}
    assert debug["retriever_counts"]["ocr_lexical"] == 1
