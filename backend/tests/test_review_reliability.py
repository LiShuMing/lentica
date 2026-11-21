from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from backend.app.db.models import FaceDetection, File, Image, PersonCluster, ProcessingRun, Source
from backend.app.ingestion import scanner
from backend.app.ingestion.faces import FaceIndexer, matching_person_file_ids
from backend.app.ingestion.scanner import scan_documents
from backend.app.models.local_clip import Embedder
from backend.app.retrieval.photo_search import PhotoSearch, upsert_photo_vector
from backend.app.workers import image_worker, queue_controller
from backend.app.workers.image_worker import ImageWorker, IndexStats, enqueue_document_index, enqueue_photo_index


def document_source(db, tmp_path):
    source = Source(name="Review fixture", root_path=str(tmp_path), kind="documents")
    db.add(source)
    db.commit()
    return source


def test_bounded_document_scans_advance_past_unchanged_files(db, tmp_path):
    source = document_source(db, tmp_path)
    for index in range(3):
        (tmp_path / f"{index}.txt").write_text(f"Document {index}")
    for _ in range(3):
        job = scan_documents(db, source_id=source.id, max_files=1)
        assert job.status == "succeeded"
        assert job.indexed_files == 1
    assert all(file.ingestion_status == "indexed" for file in db.scalars(select(File)))
    revisit_job = scan_documents(db, source_id=source.id, max_files=1)
    assert revisit_job.status == "succeeded"
    assert revisit_job.total_files == 1
    assert revisit_job.skipped_files == 1


def test_unavailable_document_source_does_not_mark_records_missing(db, tmp_path):
    source = document_source(db, tmp_path / "disconnected")
    file = File(
        source_id=source.id,
        canonical_path=str(tmp_path / "disconnected/a.txt"),
        windows_path="a.txt",
        file_type="txt",
        ingestion_status="indexed",
    )
    db.add(file)
    db.commit()
    job = scan_documents(db, source_id=source.id, max_files=None)
    assert job.status == "failed"
    assert "unavailable" in job.error
    assert file.ingestion_status == "indexed"


def test_full_scan_does_not_mark_oversized_existing_document_missing(db, tmp_path):
    source = document_source(db, tmp_path)
    (tmp_path / "a.txt").write_text("long document")
    scan_documents(db, source_id=source.id)
    scan_documents(db, source_id=source.id, max_files=None, max_bytes=1)
    assert db.scalar(select(File)).ingestion_status == "indexed"


def test_document_parser_failure_is_visible_in_job_status(db, tmp_path, monkeypatch):
    source = document_source(db, tmp_path)
    (tmp_path / "broken.txt").write_text("fixture")

    def fail_parse(path):
        raise ValueError("fixture parser failure")

    monkeypatch.setattr(scanner, "parse_document", fail_parse)
    job = scan_documents(db, source_id=source.id)
    assert job.status == "failed"
    assert job.failed_files == 1
    assert "fixture parser failure" in db.scalar(select(File)).error


class FixedEmbedder(Embedder):
    dims = 2

    def embed_text(self, text):
        return [1.0, 0.0]


def photo(db, source, name, year=2025, vector=None):
    file = File(
        source_id=source.id,
        canonical_path=f"/fixture/{name}.jpg",
        windows_path=f"{name}.jpg",
        file_type="jpg",
        ingestion_status="indexed",
    )
    db.add(file)
    db.flush()
    db.add(Image(file_id=file.id, caption="fixture", exif_time=datetime(year, 1, 1)))
    upsert_photo_vector(db, file.id, "clip:image", vector or [1.0, 0.0], encoder_id=FixedEmbedder().encoder_id)
    return file


def person(db, name, files=()):
    cluster = PersonCluster(label=f"person_{name}", display_name=name)
    db.add(cluster)
    db.flush()
    for file in files:
        db.add(FaceDetection(file_id=file.id, cluster_id=cluster.id, bbox_json="[]"))
    db.flush()
    return cluster


def test_person_and_year_filter_happen_before_recall_limit(db, tmp_path):
    source = document_source(db, tmp_path)
    for index in range(70):
        photo(db, source, f"distractor_{index}", year=2024)
    target = photo(db, source, "target", vector=[0.0, 1.0])
    person(db, "Alice", [target])
    db.commit()
    search = PhotoSearch(db, FixedEmbedder())
    for query in ("Alice", "2025", "Alice 2025"):
        _, hits = search.search(query, limit=1)
        assert [hit.file_id for hit in hits] == [target.id]


def test_person_without_photos_does_not_return_unrelated_hits(db, tmp_path):
    source = document_source(db, tmp_path)
    photo(db, source, "unrelated")
    person(db, "Alice")
    db.commit()
    assert PhotoSearch(db, FixedEmbedder()).search("Alice")[1] == []


def test_multiple_people_require_cooccurrence(db, tmp_path):
    source = document_source(db, tmp_path)
    alice_photo = photo(db, source, "alice")
    bob_photo = photo(db, source, "bob")
    person(db, "Alice", [alice_photo])
    person(db, "Bob", [bob_photo])
    db.commit()
    ids, names = matching_person_file_ids(db, "Alice Bob")
    assert names == ["Alice", "Bob"]
    assert ids == set()
    assert PhotoSearch(db, FixedEmbedder()).search("Alice Bob")[1] == []


def test_split_clusters_with_same_name_remain_one_person(db, tmp_path):
    source = document_source(db, tmp_path)
    first = photo(db, source, "first")
    second = photo(db, source, "second")
    cluster = person(db, "Alice", [first])
    cluster.label = "person_alice_first"
    person(db, "Alice", [second])
    db.commit()
    assert matching_person_file_ids(db, "Alice")[0] == {first.id, second.id}


def test_partial_photo_failure_marks_task_failed_and_preserves_counters(db, monkeypatch):
    task = enqueue_photo_index(db, {})
    worker = ImageWorker("review-fixture")
    monkeypatch.setattr(worker, "_index_photos", lambda *args: IndexStats(indexed=2, failed=1))
    worker._run_task(db, worker._claim_task(db))
    run = db.scalar(select(ProcessingRun).where(ProcessingRun.task_id == task.id))
    assert task.state == "failed"
    assert run.error and run.finished_at and run.model_profile
    assert json.loads(run.output_artifact_ids)["indexed"] == 2


def test_failed_document_job_is_not_reported_as_success(db, monkeypatch):
    task = enqueue_document_index(db, {})
    monkeypatch.setattr(
        image_worker,
        "scan_documents",
        lambda *args, **kwargs: SimpleNamespace(
            id=1,
            status="failed",
            total_files=0,
            indexed_files=0,
            skipped_files=0,
            failed_files=0,
            error="Source unavailable",
        ),
    )
    worker = ImageWorker("review-fixture")
    worker._run_task(db, worker._claim_task(db))
    assert task.state == "failed"
    assert "Source unavailable" in task.error


def test_explicit_face_job_with_unavailable_runtime_fails(db, monkeypatch):
    task = enqueue_photo_index(db, {})
    task.pipeline_stage = "index_faces"
    db.commit()
    worker = ImageWorker("review-fixture")
    monkeypatch.setattr(worker, "_index_faces", lambda *args: IndexStats(face_unavailable=1, skipped=1))
    worker._run_task(db, worker._claim_task(db))
    assert task.state == "failed"


def test_face_detector_error_is_not_counted_as_indexed(db, tmp_path):
    source = document_source(db, tmp_path)
    file = photo(db, source, "failed_face")
    db.commit()

    class FailedDetector:
        def detect(self, path):
            raise ValueError("fixture detector failure")

    worker = ImageWorker("review-fixture")
    worker.models._face_indexer = FaceIndexer(detector=FailedDetector())
    stats = worker._index_faces(db, {})
    assert stats.failed == 1
    assert stats.indexed == 0
    assert db.scalar(select(Image).where(Image.file_id == file.id)).face_analysis_status == "failed"


def test_zero_concurrency_overrides_gpu_fallback():
    settings = queue_controller.QueueSettings(enabled=True, worker_mode="gpu", max_concurrency=0)
    assert queue_controller.QueueController()._desired_concurrency(settings, {"memory_free_mb": 0}) == 0


@pytest.mark.parametrize("content", ["{", "[]", '{"enabled": "false"}', '{"max_concurrency": -1}'])
def test_queue_settings_invalid_json_does_not_enable_defaults(tmp_path, monkeypatch, content):
    path = tmp_path / "queue.json"
    path.write_text(content)
    monkeypatch.setattr(queue_controller, "SETTINGS_PATH", path)
    with pytest.raises(ValueError, match="refusing to enable defaults"):
        queue_controller.load_queue_settings()


def test_queue_settings_replacement_failure_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "queue.json"
    monkeypatch.setattr(queue_controller, "SETTINGS_PATH", path)
    queue_controller.save_queue_settings(queue_controller.QueueSettings(enabled=False))
    original = path.read_text()

    def fail_replace(*args):
        raise OSError("fixture failure")

    monkeypatch.setattr(queue_controller.os, "replace", fail_replace)
    with pytest.raises(OSError):
        queue_controller.save_queue_settings(queue_controller.QueueSettings(enabled=True))
    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]
    assert queue_controller.load_queue_settings().enabled is False
