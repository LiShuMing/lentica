from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
from PIL import Image as PilImage
from sqlalchemy import func, select

from backend.app.db.models import File, Image, PhotoVector, ProcessingRun, Source
from backend.app.ingestion.scanner import scan_documents
from backend.app.models.local_clip import Embedder, HashEmbedder
from backend.app.retrieval.photo_search import PhotoSearch
from backend.app.retrieval.service import search_knowledge
from backend.app.workers.image_worker import ImageWorker, enqueue_photo_index
from backend.app.workers.queue_controller import QueueSettings


def test_document_ingestion_is_incremental_and_retrievable(db, tmp_path):
    source = Source(name="Fixture documents", root_path=str(tmp_path), kind="documents")
    db.add(source)
    db.commit()
    path = tmp_path / "join.md"
    path.write_text("# Incremental joins\nA materialized view updates when a source table changes.\n", encoding="utf-8")

    first = scan_documents(db, source_id=source.id)
    second = scan_documents(db, source_id=source.id)
    assert first.indexed_files == 1
    assert second.skipped_files == 1
    hits = search_knowledge(db, "materialized view")
    assert hits[0].path == str(path)
    assert hits[0].page_or_segment == "seg.1"


def test_photo_job_records_vectors_thumbnail_and_run(db, tmp_path):
    photo = tmp_path / "fixture.png"
    PilImage.new("RGB", (64, 48), color="green").save(photo)
    task = enqueue_photo_index(db, {
        "root": str(tmp_path), "limit": 1, "max_indexed": 1,
        "captioner_kind": "basic", "vision_backend": "basic",
        "embedder_kind": "hash", "text_embedder_kind": "hash", "device": "cpu",
    })
    worker = ImageWorker("fixture-worker")
    claimed = worker._claim_task(db)
    assert claimed.id == task.id
    worker._run_task(db, claimed)

    assert task.state == "succeeded"
    run = db.scalar(select(ProcessingRun).where(ProcessingRun.task_id == task.id))
    assert run.finished_at is not None
    assert json.loads(run.output_artifact_ids)["indexed"] == 1
    image = db.scalar(select(Image))
    assert image.thumb_path and Path(image.thumb_path).exists()
    assert db.scalar(select(func.count(PhotoVector.id))) > 1
    assert db.scalar(select(func.count(File.id))) == 1
    _answer, hits = PhotoSearch(db, HashEmbedder()).search("fixture")
    assert hits[0].path == str(photo)

    class SemanticStub(Embedder):
        def embed_text(self, text):
            return HashEmbedder().embed_text(text)

    _answer, real_hits = PhotoSearch(db, SemanticStub()).search("fixture")
    assert real_hits == []


def test_api_runs_without_model_downloads(db):
    from backend.app.db.session import get_db
    from backend.app.main import app

    app.dependency_overrides[get_db] = lambda: db

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            health = await client.get("/health")
            assert health.json() == {"status": "ok"}
            assert (await client.get("/api/tasks")).json() == []
            response = await client.post("/api/chat", json={"question": "materialized view"})
            assert response.status_code == 200
            assert response.json()["sources"] == []
    try:
        asyncio.run(check())
    finally:
        app.dependency_overrides.clear()


def test_environment_paths_do_not_depend_on_original_workstation(tmp_path):
    env = os.environ | {
        "LENTICA_DATA_DIR": str(tmp_path / "state"),
        "LENTICA_DOCUMENTS_SOURCE": str(tmp_path / "notes"),
        "LENTICA_VISION_ENDPOINT": "http://127.0.0.2:8100",
    }
    code = (
        "import json; from backend.app.core.config import DATA_DIR, DOCUMENTS_SOURCE, VISION_ENDPOINT; "
        "print(json.dumps([str(DATA_DIR), str(DOCUMENTS_SOURCE), VISION_ENDPOINT]))"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [str(tmp_path / "state"), str(tmp_path / "notes"), env["LENTICA_VISION_ENDPOINT"]]
    assert QueueSettings().auto_enqueue_faces is False
