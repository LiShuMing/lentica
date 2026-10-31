"""Durable request identity and fenced task ownership, independent of handlers."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError

from backend.app.db.models import ProcessingRun, ProcessingTask, TaskPayload


LEASE_SECONDS = 120


class LostLease(RuntimeError):
    pass


def enqueue_task(db, stage: str, payload: dict, *, priority: int = 100) -> ProcessingTask:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    request_key = hashlib.sha256(f"{stage}:{encoded}".encode()).hexdigest()
    active = select(ProcessingTask).where(
        ProcessingTask.request_key == request_key, ProcessingTask.state.in_(["queued", "running"])
    )
    existing = db.scalar(active)
    if existing:
        return existing
    task = ProcessingTask(
        idempotency_key=f"{stage}:{uuid.uuid4().hex}",
        request_key=request_key,
        pipeline_stage=stage,
        priority=priority,
        state="queued",
    )
    try:
        db.add(task)
        db.flush()
        db.add(TaskPayload(task_id=task.id, payload_json=encoded))
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.scalar(active)
        if existing is None:
            raise
        return existing
    return task


def recover_expired(db, *, now: datetime | None = None, max_attempts: int = 5) -> int:
    now = now or datetime.utcnow()
    tasks = db.scalars(
        select(ProcessingTask).where(
            ProcessingTask.state == "running",
            or_(ProcessingTask.lease_expires_at.is_(None), ProcessingTask.lease_expires_at <= now),
        )
    ).all()
    recovered = 0
    for task in tasks:
        exhausted = task.attempt_count >= max_attempts
        error = "Worker lease expired; retry limit reached." if exhausted else "Worker lease expired; task requeued."
        result = db.execute(
            update(ProcessingTask)
            .where(
                ProcessingTask.id == task.id,
                ProcessingTask.state == "running",
                or_(ProcessingTask.lease_expires_at.is_(None), ProcessingTask.lease_expires_at <= now),
            )
            .values(
                state="failed" if exhausted else "queued",
                worker_id=None,
                claim_token=None,
                lease_expires_at=None,
                error=error if exhausted else None,
            )
        )
        if result.rowcount:
            db.execute(
                update(ProcessingRun)
                .where(
                    ProcessingRun.task_id == task.id,
                    ProcessingRun.finished_at.is_(None),
                )
                .values(finished_at=now, error=error)
            )
            recovered += 1
    db.commit()
    return recovered


class TaskLease:
    def __init__(self, task_id: int, token: str, session_factory):
        self.task_id = task_id
        self.token = token
        self.session_factory = session_factory

    def renew(self, db) -> None:
        now = datetime.utcnow()
        with db.no_autoflush:
            result = db.execute(
                update(ProcessingTask)
                .where(
                    ProcessingTask.id == self.task_id,
                    ProcessingTask.state == "running",
                    ProcessingTask.claim_token == self.token,
                )
                .values(lease_expires_at=now + timedelta(seconds=LEASE_SECONDS))
            )
        if result.rowcount != 1:
            raise LostLease(f"Task {self.task_id} ownership expired or changed")

    def commit(self, db) -> None:
        # Ownership check and output flush share one SQLite write transaction.
        # An expired lease may renew only while its token is still current. If
        # recovery wins the SQLite write lock first, this owner is fenced out.
        self.renew(db)
        db.commit()

    @contextmanager
    def heartbeat(self):
        stop = threading.Event()

        def run():
            while not stop.wait(LEASE_SECONDS / 3):
                with self.session_factory() as db:
                    try:
                        self.commit(db)
                    except LostLease:
                        return
                    except Exception:
                        db.rollback()  # A busy DB is retried; output commits still fence.

        thread = threading.Thread(target=run, name=f"lease-{self.task_id}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5)
