from __future__ import annotations

import json
import signal
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from backend.app.db.models import File, ProcessingRun, ProcessingTask, Source
from backend.app.workers.image_worker import ImageWorker, IndexStats, enqueue_document_index, enqueue_photo_index
from backend.app.workers.queue_controller import QueueController, QueueSettings
from backend.app.workers.task_store import LostLease, TaskLease, recover_expired


def test_active_request_deduplication_and_retry_history(db):
    first = enqueue_photo_index(db, {"limit": 3, "device": "cpu"})
    assert enqueue_photo_index(db, {"device": "cpu", "limit": 3}).id == first.id
    worker = ImageWorker("first")
    claimed = worker._claim_task(db)
    assert claimed.id == first.id
    assert enqueue_photo_index(db, {"limit": 3, "device": "cpu"}).id == first.id
    first.state = "failed"
    db.commit()
    assert enqueue_photo_index(db, {"limit": 3, "device": "cpu"}).id != first.id


def test_database_rejects_duplicate_active_request(db):
    from sqlalchemy.exc import IntegrityError

    task = enqueue_photo_index(db, {})
    db.add(
        ProcessingTask(
            idempotency_key="bypass-enqueue",
            request_key=task.request_key,
            pipeline_stage="index_photos",
            state="queued",
        )
    )
    with pytest.raises(IntegrityError, match="duplicate active task request"):
        db.commit()
    db.rollback()
    historical = ProcessingTask(
        idempotency_key="history", request_key=task.request_key, pipeline_stage="index_photos", state="failed"
    )
    db.add(historical)
    db.commit()
    historical.state = "queued"
    with pytest.raises(IntegrityError, match="duplicate active task request"):
        db.commit()
    db.rollback()


def test_real_photo_batch_preserves_success_when_an_image_fails(db, tmp_path, monkeypatch):
    from PIL import Image as PilImage
    from backend.app.db.models import PhotoVector
    from backend.app.workers import image_worker

    PilImage.new("RGB", (16, 16), "green").save(tmp_path / "a-valid.png")
    PilImage.new("RGB", (16, 16), "red").save(tmp_path / "b-unreadable.png")
    inspect = image_worker.inspect_image

    def fail_one_read(path):
        if path.name == "b-unreadable.png":
            raise OSError("Injected file read failure")
        return inspect(path)

    monkeypatch.setattr(image_worker, "inspect_image", fail_one_read)
    task = enqueue_photo_index(
        db, {"root": str(tmp_path), "vision_backend": "basic", "embedder_kind": "hash", "device": "cpu"}
    )
    worker = ImageWorker("partial-real")
    worker._run_task(db, worker._claim_task(db))
    assert task.state == "failed"
    run = db.scalar(select(ProcessingRun).where(ProcessingRun.task_id == task.id))
    stats = json.loads(run.output_artifact_ids)
    assert (stats["scanned"], stats["indexed"], stats["failed"]) == (2, 1, 1)
    assert db.scalar(select(File).where(File.ingestion_status == "indexed"))
    assert db.scalar(select(PhotoVector.id))


def test_expiry_recovery_and_old_owner_cannot_write(db):
    task = enqueue_photo_index(db, {})
    worker = ImageWorker("old")
    claimed = worker._claim_task(db)
    lease = TaskLease(claimed.id, claimed.claim_token, lambda: None)
    run = ProcessingRun(task_id=task.id)
    db.add(run)
    db.commit()
    assert recover_expired(db) == 0
    task.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db.commit()
    assert recover_expired(db) == 1
    assert run.finished_at is not None
    replacement = ImageWorker("new")._claim_task(db)
    assert replacement.attempt_count == 2
    task.state = "succeeded"  # Pending stale output must not flush before ownership check.
    with pytest.raises(LostLease):
        lease.commit(db)
    db.rollback()
    db.refresh(task)
    assert task.worker_id == "new" and task.state == "running"


def test_next_attempt_time_is_respected(db):
    task = enqueue_photo_index(db, {})
    task.next_attempt_at = datetime.utcnow() + timedelta(hours=1)
    db.commit()
    assert ImageWorker()._claim_task(db) is None


def test_partial_failure_is_failed_with_output_statistics(db, monkeypatch):
    worker = ImageWorker("partial")
    monkeypatch.setattr(worker, "_index_photos", lambda *_: IndexStats(scanned=2, indexed=1, failed=1))
    task = enqueue_photo_index(db, {})
    worker._run_task(db, worker._claim_task(db))
    assert task.state == "failed"
    run = db.scalar(select(ProcessingRun).where(ProcessingRun.task_id == task.id))
    assert json.loads(run.output_artifact_ids)["indexed"] == 1
    assert run.error


def test_document_handler_failure_propagates_to_task(db, tmp_path):
    source = Source(name="Lost", root_path=str(tmp_path / "unavailable"), kind="documents")
    db.add(source)
    db.commit()
    task = enqueue_document_index(db, {"source_id": source.id})
    worker = ImageWorker("documents")
    worker._run_task(db, worker._claim_task(db))
    assert task.state == "failed"
    run = db.scalar(select(ProcessingRun))
    assert json.loads(run.output_artifact_ids)["status"] == "failed"


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid
        self.signals = []
        self.returncode = None

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)


def test_pause_and_shrink_drain_without_discarding_workers(monkeypatch):
    from backend.app.workers import queue_controller as module

    monkeypatch.setattr(module, "read_gpu_status", lambda: {})
    controller = QueueController()
    first, second = FakeProcess(101), FakeProcess(102)
    controller._workers = [first, second]
    controller._reconcile_workers(QueueSettings(enabled=True, worker_mode="smoke", max_concurrency=1))
    assert first.signals == [] and second.signals == [signal.SIGUSR1]
    assert controller._workers == [first, second]
    controller._reconcile_workers(QueueSettings(enabled=False))
    assert first.signals == [signal.SIGUSR1] and len(second.signals) == 1
    second.returncode = 0
    controller._reconcile_workers(QueueSettings(enabled=False))
    assert controller._workers == [first]


def test_zero_concurrency_stays_zero_with_low_gpu_memory():
    assert QueueController()._desired_concurrency(QueueSettings(max_concurrency=0), {}) == 0


def test_heartbeat_renews_without_holding_handler_session(tmp_path, monkeypatch):
    import time
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.db.session import Base
    from backend.app.workers import task_store

    engine = create_engine(f"sqlite:///{tmp_path / 'heartbeat.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine)
    with sessions() as session:
        task = enqueue_photo_index(session, {})
        task.state = "running"
        task.claim_token = "heartbeat"
        initial = datetime.utcnow() - timedelta(seconds=1)
        task.lease_expires_at = initial
        session.commit()
        task_id = task.id
    monkeypatch.setattr(task_store, "LEASE_SECONDS", 0.3)
    lease = TaskLease(task_id, "heartbeat", sessions)
    with lease.heartbeat():
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with sessions() as session:
                value = session.get(ProcessingTask, task_id).lease_expires_at
            if value > initial:
                break
            time.sleep(0.02)
        assert value > initial
    engine.dispose()


def test_missing_photo_source_does_not_block_document_queue(db, monkeypatch):
    from sqlalchemy.orm import Session
    from backend.app.workers import queue_controller as module

    monkeypatch.setattr(module, "SessionLocal", lambda: Session(db.get_bind()))
    controller = QueueController()
    errors = controller._enqueue_work(QueueSettings(worker_mode="smoke", document_interval_seconds=0))
    assert errors and errors[0].startswith("photos:")
    assert db.scalar(select(ProcessingTask.id).where(ProcessingTask.pipeline_stage == "index_documents"))


def test_repeated_worker_loss_stops_at_retry_limit(db):
    task = enqueue_photo_index(db, {})
    ImageWorker("crashed")._claim_task(db)
    task.attempt_count = 5
    task.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db.commit()
    recover_expired(db)
    assert task.state == "failed"
    assert "retry limit" in task.error
    assert ImageWorker("replacement")._claim_task(db) is None
