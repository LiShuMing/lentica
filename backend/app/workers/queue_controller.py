from __future__ import annotations

import json
import os
import subprocess
import signal
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from backend.app.core.config import (
    DATA_DIR,
    PROJECT_ROOT,
    VISION_ENDPOINT,
    VISION_MODEL,
    WORKER_PYTHON,
    env_bool,
    get_settings,
)
from backend.app.db.models import ProcessingTask
from backend.app.workers.task_store import recover_expired
from backend.app.db.session import SessionLocal
from backend.app.workers.image_worker import enqueue_document_index, enqueue_next_face_batch, enqueue_next_photo_batch


SETTINGS_PATH = DATA_DIR / "queue_settings.json"
WORKER_STAGES = ["index_photos", "index_documents", "index_faces", "refresh_person_vectors"]


@dataclass
class QueueSettings:
    enabled: bool = field(default_factory=lambda: env_bool("LENTICA_QUEUE_ENABLED", True))
    auto_enqueue_photos: bool = True
    auto_enqueue_faces: bool = field(default_factory=lambda: env_bool("LENTICA_ENABLE_FACES", False))
    auto_enqueue_documents: bool = True
    auto_adjust_concurrency: bool = True
    worker_mode: str = field(default_factory=lambda: os.environ.get("LENTICA_WORKER_MODE", "gpu"))
    max_concurrency: int = 4
    photo_limit: int = 100
    photo_max_indexed: int = 10
    photo_probe_limit: int = 1000
    face_limit: int = 20
    document_max_files: int = 100
    document_max_mb: int = 20
    document_interval_seconds: int = 600
    vision_backend: str = "vllm"
    vision_endpoint: str = VISION_ENDPOINT
    vision_model: str = VISION_MODEL
    vision_timeout_seconds: int = 180
    vision_fallback: str = "basic"


class QueueController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._workers: list[subprocess.Popen[bytes]] = []
        self._draining: set[int] = set()
        self._last_document_enqueue = 0.0
        self._last_error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._recover_interrupted_tasks()
            self._stop.clear()
            self._thread = threading.Thread(target=self._run_loop, name="queue-controller", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        with self._lock:
            workers = list(self._workers)
        for worker in workers:
            if worker.poll() is None:
                worker.send_signal(signal.SIGUSR1)
        for worker in workers:
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)

    def settings(self) -> QueueSettings:
        return load_queue_settings()

    def update_settings(self, patch: dict[str, Any]) -> QueueSettings:
        settings = load_queue_settings()
        values = asdict(settings)
        for key, value in patch.items():
            if key in values and value is not None:
                values[key] = value
        values["max_concurrency"] = max(0, min(int(values["max_concurrency"]), 4))
        values["face_limit"] = max(1, min(int(values["face_limit"]), 200))
        values["worker_mode"] = "gpu" if values["worker_mode"] == "gpu" else "smoke"
        values["vision_backend"] = str(values["vision_backend"]).lower()
        if values["vision_backend"] not in {"vllm", "transformers", "basic"}:
            values["vision_backend"] = "vllm"
        values["vision_fallback"] = str(values["vision_fallback"]).lower()
        if values["vision_fallback"] not in {"transformers", "basic", "none"}:
            values["vision_fallback"] = "transformers"
        values["vision_timeout_seconds"] = max(5, int(values["vision_timeout_seconds"]))
        updated = QueueSettings(**values)
        save_queue_settings(updated)
        return updated

    def status(self) -> dict[str, Any]:
        settings = load_queue_settings()
        gpu = read_gpu_status()
        with self._lock:
            workers = self._reap_workers_locked()
            worker_status = [
                {
                    "pid": worker.pid,
                    "running": worker.poll() is None,
                    "returncode": worker.poll(),
                    "draining": worker.pid in self._draining,
                }
                for worker in workers
            ]
        task_summary = self._task_summary()
        return {
            "settings": asdict(settings),
            "gpu": gpu,
            "desired_concurrency": self._desired_concurrency(settings, gpu),
            "workers": worker_status,
            "task_summary": task_summary,
            "last_error": self._last_error,
            "last_document_enqueue": self._last_document_enqueue or None,
        }

    def _recover_interrupted_tasks(self) -> None:
        with SessionLocal() as db:
            recover_expired(db)

    def _task_summary(self) -> dict[str, dict[str, int]]:
        db = SessionLocal()
        try:
            rows = db.execute(
                select(ProcessingTask.pipeline_stage, ProcessingTask.state, func.count(ProcessingTask.id))
                .where(ProcessingTask.pipeline_stage.in_(WORKER_STAGES))
                .group_by(ProcessingTask.pipeline_stage, ProcessingTask.state)
            ).all()
            summary: dict[str, dict[str, int]] = {
                "index_photos": {"queued": 0, "running": 0, "succeeded": 0, "failed": 0, "cancelled": 0},
                "index_documents": {"queued": 0, "running": 0, "succeeded": 0, "failed": 0, "cancelled": 0},
                "index_faces": {"queued": 0, "running": 0, "succeeded": 0, "failed": 0, "cancelled": 0},
                "refresh_person_vectors": {"queued": 0, "running": 0, "succeeded": 0, "failed": 0, "cancelled": 0},
            }
            for stage, state, count in rows:
                summary.setdefault(stage, {})[state] = int(count)
            return summary
        finally:
            db.close()

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._recover_interrupted_tasks()
                settings = load_queue_settings()
                errors = []
                if settings.enabled:
                    errors = self._enqueue_work(settings)
                    self._reconcile_workers(settings)
                else:
                    self._reconcile_workers(QueueSettings(enabled=False, max_concurrency=0))
                self._last_error = "; ".join(errors) or None
            except Exception as exc:  # noqa: BLE001
                self._last_error = repr(exc)
            self._stop.wait(5)

    def _enqueue_work(self, settings: QueueSettings) -> list[str]:
        db = SessionLocal()
        errors = []

        def enqueue(stage, handler, payload):
            try:
                handler(db, payload)
            except Exception as exc:
                db.rollback()
                errors.append(f"{stage}: {exc}")

        try:
            if settings.auto_enqueue_photos:
                enqueue(
                    "photos",
                    enqueue_next_photo_batch,
                    {
                        "root": str(get_settings().photos_source),
                        "limit": settings.photo_limit,
                        "max_indexed": settings.photo_max_indexed,
                        "probe_limit": settings.photo_probe_limit,
                        "captioner_kind": "qwen-vl" if settings.worker_mode == "gpu" else "basic",
                        "vision_backend": settings.vision_backend if settings.worker_mode == "gpu" else "basic",
                        "vision_endpoint": settings.vision_endpoint,
                        "vision_model": settings.vision_model,
                        "vision_timeout_seconds": settings.vision_timeout_seconds,
                        "vision_fallback": settings.vision_fallback,
                        "embedder_kind": "clip" if settings.worker_mode == "gpu" else "hash",
                        "text_embedder_kind": "bge" if settings.worker_mode == "gpu" else "hash",
                        "device": "cuda" if settings.worker_mode == "gpu" else "cpu",
                        "embedding_device": (
                            "cpu"
                            if settings.worker_mode == "gpu" and settings.vision_backend == "vllm"
                            else ("cuda" if settings.worker_mode == "gpu" else "cpu")
                        ),
                        "unindexed_only": True,
                    },
                )
            if settings.auto_enqueue_faces:
                enqueue("faces", enqueue_next_face_batch, {"limit": settings.face_limit, "priority": 90})
            now = time.monotonic()
            if (
                settings.auto_enqueue_documents
                and now - self._last_document_enqueue >= settings.document_interval_seconds
            ):
                if not _has_active_task(db, "index_documents"):
                    enqueue(
                        "documents",
                        enqueue_document_index,
                        {
                            "max_files": settings.document_max_files,
                            "max_mb": settings.document_max_mb,
                        },
                    )
                    self._last_document_enqueue = now
        finally:
            db.close()
        return errors

    def _reconcile_workers(self, settings: QueueSettings) -> None:
        gpu = read_gpu_status()
        desired = self._desired_concurrency(settings, gpu)
        with self._lock:
            workers = self._reap_workers_locked()
            active = [worker for worker in workers if worker.pid not in self._draining]
            for worker in active[desired:]:
                if worker.poll() is None:
                    worker.send_signal(signal.SIGUSR1)
                    self._draining.add(worker.pid)
            # Draining processes still consume memory; wait for their exit before
            # replacing them, including a rapid pause/resume toggle.
            while len(workers) < desired and not self._stop.is_set():
                workers.append(self._spawn_worker())
            self._workers = workers

    def _desired_concurrency(self, settings: QueueSettings, gpu: dict[str, Any]) -> int:
        if not settings.enabled:
            return 0
        if not settings.auto_adjust_concurrency or settings.worker_mode != "gpu":
            return settings.max_concurrency
        free_mb = int(gpu.get("memory_free_mb") or 0)
        if free_mb >= 24000:
            return min(settings.max_concurrency, 2)
        if free_mb >= 8000:
            return min(settings.max_concurrency, 1)
        return min(settings.max_concurrency, 1) if settings.auto_enqueue_documents else 0

    def _spawn_worker(self) -> subprocess.Popen[bytes]:
        python = WORKER_PYTHON
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        return subprocess.Popen(
            [str(python), "-m", "backend.app.workers.image_worker"],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _reap_workers_locked(self) -> list[subprocess.Popen[bytes]]:
        self._workers = [worker for worker in self._workers if worker.poll() is None]
        self._draining.intersection_update(worker.pid for worker in self._workers)
        return self._workers


def load_queue_settings() -> QueueSettings:
    if not SETTINGS_PATH.exists():
        save_queue_settings(QueueSettings())
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return QueueSettings(**{**asdict(QueueSettings()), **data})
    except Exception:
        return QueueSettings()


def save_queue_settings(settings: QueueSettings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=SETTINGS_PATH.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(asdict(settings), stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, SETTINGS_PATH)
        finally:
            temporary.unlink(missing_ok=True)


def read_gpu_status() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        first = result.stdout.strip().splitlines()[0]
        total, used, free, util = [int(part.strip()) for part in first.split(",")]
        return {
            "available": True,
            "memory_total_mb": total,
            "memory_used_mb": used,
            "memory_free_mb": free,
            "utilization_percent": util,
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}


def _has_active_task(db, stage: str) -> bool:
    return (
        db.scalar(
            select(ProcessingTask.id).where(
                ProcessingTask.pipeline_stage == stage,
                ProcessingTask.state.in_(["queued", "running"]),
            )
        )
        is not None
    )


queue_controller = QueueController()
