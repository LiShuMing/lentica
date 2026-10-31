from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image as PilImage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db.models import File, Image, PhotoVector, Source
from backend.app.ingestion.scanner import scan_documents
from backend.app.ingestion.traversal import iter_source_files
from backend.app.workers.image_worker import (
    ImageWorker,
    PHOTO_CONTEXT_VERSION,
    _first_unindexed_image,
    enqueue_photo_index,
)


def test_document_batches_resume_across_sessions_and_wrap(db, tmp_path):
    source = Source(name="Paged", root_path=str(tmp_path), kind="documents")
    db.add(source)
    db.commit()
    source_id = source.id
    for index in range(215):
        (tmp_path / f"{index:04}.md").write_text(f"document {index}")
    assert scan_documents(db, source_id=source_id, max_files=100).indexed_files == 100
    with Session(db.get_bind()) as resumed:
        assert scan_documents(resumed, source_id=source_id, max_files=100).indexed_files == 100
        assert scan_documents(resumed, source_id=source_id, max_files=100).indexed_files == 15
        assert scan_documents(resumed, source_id=source_id, max_files=100).skipped_files == 100
    assert db.scalar(select(func.count(File.id)).where(File.source_id == source_id)) == 215


def test_unavailable_full_scan_does_not_mark_documents_missing(db, tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.md").write_text("original evidence")
    source = Source(name="Unavailable", root_path=str(root), kind="documents")
    db.add(source)
    db.commit()
    assert scan_documents(db, source_id=source.id).indexed_files == 1
    root.rename(tmp_path / "unmounted")
    assert scan_documents(db, source_id=source.id, max_files=None).status == "failed"
    assert db.scalar(select(File.ingestion_status).where(File.source_id == source.id)) == "indexed"


def test_photo_probe_advances_past_thousand_indexed_files(db, tmp_path):
    source = Source(name="Photos fixture", root_path=str(tmp_path), kind="images")
    db.add(source)
    db.flush()
    digest = hashlib.sha256(b"fixture").hexdigest()
    for index in range(1001):
        path = tmp_path / f"{index:04}.png"
        path.write_bytes(b"fixture")
        photo = File(
            source_id=source.id,
            canonical_path=str(path),
            windows_path=str(path),
            file_type="png",
            sha256=digest,
            ingestion_status="indexed",
        )
        db.add(photo)
        db.flush()
        db.add(
            Image(
                file_id=photo.id,
                context_version=PHOTO_CONTEXT_VERSION,
                context_json=json.dumps({"metadata_candidates": []}),
            )
        )
    target = tmp_path / "1001.png"
    target.write_bytes(b"new")
    db.commit()
    assert _first_unindexed_image(db, tmp_path, probe_limit=1000) is None
    with Session(db.get_bind()) as resumed:
        assert _first_unindexed_image(resumed, tmp_path, probe_limit=1000) == target


def test_traversal_order_resume_and_symlink_boundary(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "file.md").write_text("nested")
    (tmp_path / "a.md").write_text("root")
    (tmp_path / ".private.md").write_text("hidden")
    (tmp_path / "link.md").symlink_to(tmp_path / "a.md")
    paths = list(iter_source_files(tmp_path, {".md"}))
    assert paths == sorted(paths, key=str)
    assert len(paths) == 2
    assert list(iter_source_files(tmp_path, {".md"}, after=str(paths[0]))) == paths[1:]


def test_same_path_photo_change_replaces_old_vectors(db, tmp_path):
    path = tmp_path / "photo.png"
    PilImage.new("RGB", (32, 24), "green").save(path)
    payload = {"root": str(tmp_path), "vision_backend": "basic", "embedder_kind": "hash", "device": "cpu"}
    worker = ImageWorker("fixture")
    task = enqueue_photo_index(db, payload)
    worker._run_task(db, worker._claim_task(db))
    assert task.state == "succeeded"
    original = db.scalar(select(File.sha256))
    file_id = db.scalar(select(File.id))
    db.add(PhotoVector(file_id=file_id, namespace="bge:stale", dims=2, vector_json="[1,0]", encoder_id="old"))
    db.commit()
    PilImage.new("RGB", (32, 24), "blue").save(path)
    assert _first_unindexed_image(db, tmp_path) == path
    task = enqueue_photo_index(db, payload)
    worker._run_task(db, worker._claim_task(db))
    assert task.state == "succeeded"
    assert db.scalar(select(File.sha256)) != original
    assert db.scalar(select(PhotoVector.id).where(PhotoVector.namespace == "bge:stale")) is None
    assert all(identity for identity in db.scalars(select(PhotoVector.encoder_id)))


def test_unavailable_source_does_not_block_other_document_sources(db, tmp_path):
    (tmp_path / "doc.md").write_text("available evidence")
    db.add(Source(name="Unavailable first", root_path=str(tmp_path / "missing"), kind="documents"))
    db.add(Source(name="Available next", root_path=str(tmp_path), kind="documents"))
    db.commit()
    job = scan_documents(db)
    assert job.status == "failed"
    assert job.indexed_files == 1
