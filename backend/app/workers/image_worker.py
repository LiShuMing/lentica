from __future__ import annotations

import argparse
import itertools
import json
import hashlib
import logging
import socket
import signal
import threading
import uuid
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, exists, select, update, or_

from backend.app.core.config import (
    CLIP_MODEL,
    DATA_DIR,
    PHOTOS_SOURCE,
    QWEN_MODEL,
    TEXT_MODEL,
    VISION_ENDPOINT,
    VISION_MODEL,
    get_settings,
)
from backend.app.core.paths import to_windows_path
from backend.app.db.models import (
    FaceDetection,
    File,
    Image,
    PhotoVector,
    ProcessingRun,
    ProcessingTask,
    Source,
    TaskPayload,
)
from backend.app.db.session import SessionLocal, init_db
from backend.app.ingestion.faces import FaceIndexer, people_for_file, refresh_people_context_for_file
from backend.app.ingestion.photo_metadata import (
    candidates_text,
    ensure_local_geocode_cache,
    upsert_photo_metadata_candidates,
)
from backend.app.ingestion.images import inspect_image, iter_image_files, make_thumbnail, sha256_file, SUPPORTED_IMAGES
from backend.app.ingestion.traversal import iter_source_files, scan_cursor
from backend.app.workers.task_store import LEASE_SECONDS, LostLease, TaskLease, enqueue_task
from backend.app.embedding.context_text_builder import build_context_vector_texts
from backend.app.ingestion.scanner import scan_documents
from backend.app.models.local_clip import Embedder, build_embedder
from backend.app.models.vision_service import VisionClientConfig, VisionServiceClient, build_vision_client_with_fallback
from backend.app.retrieval.photo_search import upsert_photo_vector


DEFAULT_QWEN_VL_PATH = QWEN_MODEL
DEFAULT_CLIP_MODEL = CLIP_MODEL
DEFAULT_BGE_MODEL = TEXT_MODEL
FALLBACK_BGE_MODEL = "BAAI/bge-small-zh-v1.5"
PHOTO_CONTEXT_VERSION = "photo_context"


@dataclass
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    faces: int = 0
    face_clusters_created: int = 0
    face_unavailable: int = 0
    vectors_written: int = 0
    metadata_backfilled: int = 0


class ModelManager:
    def __init__(self) -> None:
        self._vision_clients: dict[tuple[str, str, str, str, int, str, str, int], VisionServiceClient] = {}
        self._embedders: dict[tuple[str, str, str], Embedder] = {}
        self._face_indexer: FaceIndexer | None = None
        self.logger = logging.getLogger("personal_knowledge_lab.image_worker.models")

    def vision_client(self, config: VisionClientConfig) -> VisionServiceClient:
        key = (
            config.backend,
            config.endpoint,
            config.model,
            config.transformers_model,
            config.timeout_seconds,
            config.fallback,
            config.device,
            config.max_pixels,
        )
        if key not in self._vision_clients:
            self.logger.info(
                "vision_client_create backend=%s endpoint=%s model=%s fallback=%s",
                config.backend,
                config.endpoint,
                config.model,
                config.fallback,
            )
            self._vision_clients[key] = build_vision_client_with_fallback(config)
        return self._vision_clients[key]

    def embedder(self, kind: str, model_name: str, device: str) -> Embedder:
        key = (kind, model_name, device)
        if key not in self._embedders:
            self.logger.info("embedder_load_start kind=%s model=%s device=%s", kind, model_name, device)
            self._embedders[key] = build_embedder(kind, model_name, device)
            self.logger.info("embedder_load_done kind=%s model=%s device=%s", kind, model_name, device)
        return self._embedders[key]

    def face_indexer(self) -> FaceIndexer:
        if self._face_indexer is None:
            self._face_indexer = FaceIndexer()
        return self._face_indexer


class ImageWorker:
    def __init__(self, worker_id: str | None = None) -> None:
        self.worker_id = worker_id or f"{socket.gethostname()}:{time.time_ns()}"
        self.models = ModelManager()
        self.stop_requested = threading.Event()
        self._lease: TaskLease | None = None
        self.logger = logging.getLogger("personal_knowledge_lab.image_worker")

    def run_forever(
        self,
        poll_seconds: float = 2.0,
        auto_enqueue: bool = False,
        auto_payload: dict[str, Any] | None = None,
    ) -> None:
        self.logger.info("worker_start worker_id=%s auto_enqueue=%s", self.worker_id, auto_enqueue)
        while not self.stop_requested.is_set():
            handled = self.run_once()
            if not handled and auto_enqueue:
                handled = self.enqueue_next_batch(auto_payload or {})
            if not handled:
                self.stop_requested.wait(poll_seconds)

    def run_once(self) -> bool:
        db = SessionLocal()
        try:
            task = self._claim_task(db)
            if not task:
                return False
            self._run_task(db, task)
            return True
        finally:
            db.close()

    def enqueue_next_batch(self, payload: dict[str, Any]) -> bool:
        db = SessionLocal()
        try:
            task = enqueue_next_photo_batch(db, payload)
            if not task:
                self.logger.info("auto_enqueue_idle reason=no_unindexed_candidates_or_active_task")
                return False
            self.logger.info("auto_enqueue_task task_id=%s", task.id)
            return True
        finally:
            db.close()

    def _claim_task(self, db) -> ProcessingTask | None:
        task_id = db.scalar(
            select(ProcessingTask.id)
            .where(
                ProcessingTask.pipeline_stage.in_(
                    ["index_photos", "index_documents", "index_faces", "refresh_person_vectors"]
                ),
                ProcessingTask.state == "queued",
                or_(ProcessingTask.next_attempt_at.is_(None), ProcessingTask.next_attempt_at <= datetime.utcnow()),
            )
            .order_by(ProcessingTask.priority.asc(), ProcessingTask.id.asc())
        )
        if task_id is None:
            return None
        result = db.execute(
            update(ProcessingTask)
            .where(ProcessingTask.id == task_id, ProcessingTask.state == "queued")
            .values(
                state="running",
                worker_id=self.worker_id,
                attempt_count=ProcessingTask.attempt_count + 1,
                claim_token=uuid.uuid4().hex,
                lease_expires_at=datetime.utcnow() + timedelta(seconds=LEASE_SECONDS),
            )
        )
        if result.rowcount != 1:
            db.rollback()
            return None
        db.commit()
        task = db.get(ProcessingTask, task_id)
        if not task:
            return None
        db.refresh(task)
        return task

    def _commit(self, db) -> None:
        if self._lease:
            self._lease.commit(db)
        else:
            db.commit()

    def _document_task(self, db, payload):
        job = scan_documents(
            db,
            source_id=payload.get("source_id"),
            max_files=None if int(payload.get("max_files", 100)) == 0 else int(payload.get("max_files", 100)),
            max_bytes=int(payload.get("max_mb", 20)) * 1024 * 1024,
            commit=lambda: self._commit(db),
        )
        return {
            key: getattr(job, key)
            for key in ("status", "total_files", "indexed_files", "skipped_files", "failed_files", "error")
        }

    def _run_task(self, db, task: ProcessingTask) -> None:
        self._lease = TaskLease(task.id, task.claim_token, SessionLocal)
        run = ProcessingRun(task_id=task.id, started_at=datetime.utcnow())
        payload = _task_payload(db, task.id)
        handlers = {
            "index_photos": (self._index_photos, lambda: _photo_model_profile(payload)),
            "index_faces": (self._index_faces, lambda: "insightface+incremental_centroid"),
            "refresh_person_vectors": (self._refresh_person_vectors, lambda: _person_refresh_model_profile(payload)),
            "index_documents": (self._document_task, lambda: "document_parser+local_tfidf"),
        }
        try:
            db.add(run)
            self._commit(db)
            with self._lease.heartbeat():
                handler, profile = handlers[task.pipeline_stage]
                output = handler(db, payload)
                stats = output if isinstance(output, dict) else output.__dict__
                failed = int(stats.get("failed", stats.get("failed_files", 0)))
                error = stats.get("error")
                if failed or stats.get("status") == "failed":
                    error = error or f"{failed} item(s) failed; see run statistics"
                task.state = "failed" if error else "succeeded"
                task.error = str(error) if error else None
                run.error = task.error
                run.finished_at = datetime.utcnow()
                run.model_profile = profile()
                run.output_artifact_ids = json.dumps(stats, ensure_ascii=False)
                self._commit(db)
        except LostLease:
            db.rollback()
            self.logger.warning("task_ownership_lost task_id=%s", task.id)
        except Exception as exc:
            db.rollback()
            try:
                # Verify before recording failure; a stale owner must not overwrite recovery.
                self._lease.renew(db)
                task.state = "failed"
                task.error = repr(exc)
                run.finished_at = datetime.utcnow()
                run.error = repr(exc)
                self._commit(db)
            except LostLease:
                db.rollback()
            self.logger.exception("task_failed task_id=%s", task.id)
        finally:
            self._lease = None

    def _index_photos(self, db, payload: dict[str, Any]) -> IndexStats:
        root = Path(payload.get("root") or get_settings().photos_source)
        if not root.is_dir():
            raise FileNotFoundError(f"Source directory unavailable: {root}")
        limit = payload.get("limit")
        max_indexed = payload.get("max_indexed")
        force = bool(payload.get("force", False))
        captioner_kind = payload.get("captioner_kind", "basic")
        caption_model = payload.get("caption_model") or DEFAULT_QWEN_VL_PATH
        vision_backend = payload.get("vision_backend") or ("vllm" if captioner_kind == "qwen-vl" else "basic")
        vision_endpoint = payload.get("vision_endpoint") or VISION_ENDPOINT
        vision_model = payload.get("vision_model") or VISION_MODEL
        vision_timeout_seconds = int(payload.get("vision_timeout_seconds") or 180)
        vision_fallback = payload.get("vision_fallback") or "transformers"
        embedder_kind = payload.get("embedder_kind", "hash")
        embed_model = payload.get("embed_model") or DEFAULT_CLIP_MODEL
        text_embedder_kind = payload.get("text_embedder_kind") or ("bge" if embedder_kind == "clip" else embedder_kind)
        text_embed_model = payload.get("text_embed_model") or DEFAULT_BGE_MODEL
        device = payload.get("device", "cuda")
        embedding_device = payload.get("embedding_device") or ("cpu" if vision_backend == "vllm" else device)
        max_pixels = int(payload.get("max_pixels", 768 * 768))
        unindexed_only = bool(payload.get("unindexed_only", True))
        target_path = str(payload.get("target_path") or "")

        self._commit(db)  # Release the payload read snapshot before loading models.
        vision_client = self.models.vision_client(
            VisionClientConfig(
                backend=vision_backend,
                endpoint=vision_endpoint,
                model=vision_model,
                timeout_seconds=vision_timeout_seconds,
                fallback=vision_fallback,
                transformers_model=caption_model,
                device=device,
                max_pixels=max_pixels,
            )
        )
        image_embedder = self.models.embedder(embedder_kind, embed_model, embedding_device)
        actual_text_embedder_kind, text_embedder = self._text_embedder(
            text_embedder_kind, text_embed_model, embedding_device, embedder_kind, image_embedder
        )
        source = _ensure_photo_source(db, root)
        thumb_dir = DATA_DIR / "cache" / "thumbs"
        stats = IndexStats()
        started = time.monotonic()
        self.logger.info(
            "index_start root=%s limit=%s max_indexed=%s force=%s captioner=%s embedder=%s embedding_device=%s",
            root,
            limit,
            max_indexed,
            force,
            captioner_kind,
            embedder_kind,
            embedding_device,
        )
        image_paths = iter_image_files(root)
        if target_path:
            image_paths = itertools.dropwhile(lambda path: str(path) != target_path, image_paths)
        for image_path in image_paths:
            if limit is not None and stats.scanned >= int(limit):
                break
            if max_indexed is not None and stats.indexed >= int(max_indexed):
                break
            item_started = time.monotonic()
            stats.scanned += 1
            try:
                existing = db.scalar(select(File).where(File.canonical_path == str(image_path)))
                meta = inspect_image(image_path)
                file = existing
                if (
                    file
                    and file.sha256 == meta.sha256
                    and file.ingestion_status == "indexed"
                    and not force
                    and _image_context_current(db, file)
                    and db.scalar(select(Image.index_profile).where(Image.file_id == file.id))
                    in (None, _index_profile(payload))
                ):
                    if _backfill_metadata_context(db, file, image_path, meta):
                        stats.metadata_backfilled += 1
                        self._commit(db)
                    stats.skipped += 1
                    self.logger.info(
                        "image_skipped scanned=%d indexed=%d skipped=%d failed=%d elapsed=%.2fs path=%s",
                        stats.scanned,
                        stats.indexed,
                        stats.skipped,
                        stats.failed,
                        time.monotonic() - item_started,
                        image_path,
                    )
                    continue

                self._commit(db)  # Keep slow vision requests outside a read transaction.
                thumb = make_thumbnail(image_path, thumb_dir)
                analysis = vision_client.analyze_photo(image_path)
                context = _normalize_context(analysis)
                caption = str(context.get("caption") or "")
                tags = list(context.get("tags") or [])
                ocr_text = str(context.get("ocr_text") or context.get("text_in_image") or "")
                file = _upsert_image_file(db, source, image_path, meta)
                _upsert_image_context(db, file, meta, thumb, context)
                face_stats = self.models.face_indexer().index_photo(db, file, image_path, force=force)
                stats.faces += face_stats.faces
                stats.face_clusters_created += face_stats.clusters_created
                stats.face_unavailable += 1 if face_stats.unavailable else 0
                if face_stats.faces:
                    _attach_people_context(db, file, context)
                _attach_metadata_context(db, file, meta, context)
                image = db.scalar(select(Image).where(Image.file_id == file.id))
                image.index_profile = _index_profile(payload)
                # Replace the complete vector set in the same transaction as the context.
                db.execute(delete(PhotoVector).where(PhotoVector.file_id == file.id))
                image_vector = image_embedder.embed_image(image_path, caption=caption)
                upsert_photo_vector(
                    db,
                    file.id,
                    _vector_namespace(embedder_kind, "image"),
                    image_vector,
                    encoder_id=image_embedder.encoder_id,
                )
                texts = {
                    name: value
                    for name, value in _context_vector_texts(context, image_path, meta).items()
                    if value.strip()
                }
                vectors = text_embedder.embed_texts(list(texts.values()))
                for namespace, vector in zip(texts, vectors):
                    upsert_photo_vector(
                        db,
                        file.id,
                        _vector_namespace(actual_text_embedder_kind, namespace),
                        vector,
                        encoder_id=text_embedder.encoder_id,
                    )
                self._commit(db)
                stats.indexed += 1
                self.logger.info(
                    "image_indexed scanned=%d indexed=%d skipped=%d failed=%d elapsed=%.2fs file_id=%d path=%s",
                    stats.scanned,
                    stats.indexed,
                    stats.skipped,
                    stats.failed,
                    time.monotonic() - item_started,
                    file.id,
                    image_path,
                )
            except LostLease:
                raise
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                stats.failed += 1
                self.logger.exception(
                    "image_failed scanned=%d indexed=%d skipped=%d failed=%d elapsed=%.2fs path=%s error=%r",
                    stats.scanned,
                    stats.indexed,
                    stats.skipped,
                    stats.failed,
                    time.monotonic() - item_started,
                    image_path,
                    exc,
                )
        self.logger.info(
            "index_done scanned=%d indexed=%d skipped=%d failed=%d elapsed=%.2fs",
            stats.scanned,
            stats.indexed,
            stats.skipped,
            stats.failed,
            time.monotonic() - started,
        )
        return stats

    def _index_faces(self, db, payload: dict[str, Any]) -> IndexStats:
        limit = int(payload.get("limit", 500))
        force = bool(payload.get("force", False))
        stats = IndexStats()
        query = select(File).join(Image, Image.file_id == File.id).where(File.ingestion_status == "indexed")
        if not force:
            query = query.where(
                ~exists(select(FaceDetection.id).where(FaceDetection.file_id == File.id)),
                (Image.face_analysis_status.is_(None)) | (Image.face_analysis_status.in_(["pending", "failed"])),
            )
        rows = db.execute(query.order_by(File.id.asc()).limit(limit)).scalars()
        for file in rows:
            stats.scanned += 1
            try:
                path = Path(file.canonical_path)
                face_stats = self.models.face_indexer().index_photo(db, file, path, force=force)
                stats.faces += face_stats.faces
                stats.face_clusters_created += face_stats.clusters_created
                stats.face_unavailable += 1 if face_stats.unavailable else 0
                if face_stats.unavailable:
                    stats.skipped += 1
                elif face_stats.error:
                    stats.failed += 1
                else:
                    stats.indexed += 1
                    image = db.scalar(select(Image).where(Image.file_id == file.id))
                    context = _parse_context_json(image.context_json if image else None)
                    refresh_people_context_for_file(db, file.id)
                self._commit(db)
            except LostLease:
                raise
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                stats.failed += 1
                self.logger.exception("face_index_failed file_id=%s error=%r", file.id, exc)
        return stats

    def _refresh_person_vectors(self, db, payload: dict[str, Any]) -> IndexStats:
        cluster_id = payload.get("cluster_id")
        file_ids = payload.get("file_ids")
        if cluster_id is not None:
            from backend.app.db.models import FaceDetection

            file_ids = db.scalars(
                select(FaceDetection.file_id).where(FaceDetection.cluster_id == int(cluster_id))
            ).all()
        target_file_ids = sorted(set(int(item) for item in (file_ids or [])))
        stats = IndexStats(scanned=len(target_file_ids))
        if not target_file_ids:
            return stats

        text_embedder_kind = payload.get("text_embedder_kind") or "bge"
        text_embed_model = payload.get("text_embed_model") or DEFAULT_BGE_MODEL
        embedding_device = payload.get("embedding_device") or "cpu"
        fallback_embedder = self.models.embedder("hash", "", "cpu")
        actual_text_embedder_kind, text_embedder = self._text_embedder(
            text_embedder_kind,
            text_embed_model,
            embedding_device,
            "hash",
            fallback_embedder,
        )
        rows = db.execute(
            select(File, Image).join(Image, Image.file_id == File.id).where(File.id.in_(target_file_ids))
        ).all()
        for file, image in rows:
            try:
                refresh_people_context_for_file(db, file.id)
                db.flush()
                db.refresh(image)
                context = _parse_context_json(image.context_json)
                meta = _stored_meta_from_image(image)
                texts = _context_vector_texts(context, Path(file.canonical_path), meta)
                for namespace in ["text", "people", "metadata"]:
                    db.execute(
                        delete(PhotoVector).where(
                            PhotoVector.file_id == file.id,
                            (PhotoVector.namespace == namespace) | PhotoVector.namespace.like(f"%:{namespace}"),
                        )
                    )
                    text = texts.get(namespace, "")
                    if not text.strip():
                        continue
                    upsert_photo_vector(
                        db,
                        file.id,
                        _vector_namespace(actual_text_embedder_kind, namespace),
                        text_embedder.embed_text(text),
                        encoder_id=text_embedder.encoder_id,
                    )
                    stats.vectors_written += 1
                self._commit(db)
                stats.indexed += 1
            except LostLease:
                raise
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                stats.failed += 1
                self.logger.exception("refresh_person_vectors_failed file_id=%s error=%r", file.id, exc)
        self._commit(db)
        return stats

    def _text_embedder(
        self, kind: str, model_name: str, device: str, fallback_kind: str, fallback: Embedder
    ) -> tuple[str, Embedder]:
        if kind not in {"bge", "bge-m3"}:
            return kind, self.models.embedder(kind, model_name, device)
        try:
            return "bge", self.models.embedder("bge", model_name, device)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("text_embedder_fallback primary=%s error=%r", model_name, exc)
            try:
                return "bge", self.models.embedder("bge", FALLBACK_BGE_MODEL, device)
            except Exception as fallback_exc:  # noqa: BLE001
                self.logger.warning("text_embedder_fallback_to_image_embedder error=%r", fallback_exc)
                return fallback_kind, fallback


def enqueue_photo_index(db, payload: dict[str, Any]) -> ProcessingTask:
    return enqueue_task(db, "index_photos", payload, priority=int(payload.get("priority", 100)))


def enqueue_document_index(db, payload: dict[str, Any]) -> ProcessingTask:
    return enqueue_task(db, "index_documents", payload, priority=int(payload.get("priority", 100)))


def enqueue_face_index(db, payload: dict[str, Any]) -> ProcessingTask:
    return enqueue_task(db, "index_faces", payload, priority=int(payload.get("priority", 90)))


def enqueue_person_vector_refresh(db, payload: dict[str, Any]) -> ProcessingTask:
    return enqueue_task(db, "refresh_person_vectors", payload, priority=int(payload.get("priority", 120)))


def _index_profile(payload: dict) -> str:
    embedder = payload.get("embedder_kind", "hash")
    captioner = payload.get("captioner_kind", "basic")
    profile = {
        "pipeline": "photo-context-v2-encoder-spaces-v1",
        "captioner": captioner,
        "caption_model": payload.get("caption_model") or DEFAULT_QWEN_VL_PATH,
        "vision_backend": payload.get("vision_backend") or ("vllm" if captioner == "qwen-vl" else "basic"),
        "vision_endpoint": payload.get("vision_endpoint") or VISION_ENDPOINT,
        "vision_model": payload.get("vision_model") or VISION_MODEL,
        "vision_fallback": payload.get("vision_fallback") or "transformers",
        "max_pixels": int(payload.get("max_pixels", 768 * 768)),
        "embedder": embedder,
        "embed_model": payload.get("embed_model") or DEFAULT_CLIP_MODEL,
        "text_embedder": payload.get("text_embedder_kind") or ("bge" if embedder == "clip" else embedder),
        "text_model": payload.get("text_embed_model") or DEFAULT_BGE_MODEL,
    }
    return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()


def _photo_model_profile(payload: dict[str, Any]) -> str:
    captioner_kind = payload.get("captioner_kind") or "basic"
    vision_backend = payload.get("vision_backend") or ("vllm" if captioner_kind == "qwen-vl" else "basic")
    vision_model = payload.get("vision_model") or payload.get("caption_model") or "local"
    image_embedder = payload.get("embedder_kind") or "hash"
    text_embedder = payload.get("text_embedder_kind") or image_embedder
    embedding_device = (
        payload.get("embedding_device") or ("cpu" if vision_backend == "vllm" else payload.get("device")) or "cpu"
    )
    return f"vision:{vision_backend}:{vision_model}+image:{image_embedder}@{embedding_device}+text:{text_embedder}@{embedding_device}"


def _person_refresh_model_profile(payload: dict[str, Any]) -> str:
    text_embedder = payload.get("text_embedder_kind") or "bge"
    model = payload.get("text_embed_model") or DEFAULT_BGE_MODEL
    device = payload.get("embedding_device") or "cpu"
    return f"person_context_refresh+text:{text_embedder}:{model}@{device}"


def enqueue_next_photo_batch(db, payload: dict[str, Any]) -> ProcessingTask | None:
    if _has_active_photo_task(db):
        return None
    root = Path(payload.get("root") or get_settings().photos_source)
    probe_limit = int(payload.get("probe_limit", 1000))
    candidate = _first_unindexed_image(db, root, probe_limit=probe_limit, payload=payload)
    if not candidate:
        return None
    task_payload = {
        "root": str(root),
        "limit": payload.get("limit", probe_limit),
        "max_indexed": payload.get("max_indexed", 10),
        "force": bool(payload.get("force", False)),
        "unindexed_only": bool(payload.get("unindexed_only", True)),
        "captioner_kind": payload.get("captioner_kind", "basic"),
        "caption_model": payload.get("caption_model"),
        "vision_backend": payload.get("vision_backend"),
        "vision_endpoint": payload.get("vision_endpoint"),
        "vision_model": payload.get("vision_model"),
        "vision_timeout_seconds": payload.get("vision_timeout_seconds"),
        "vision_fallback": payload.get("vision_fallback"),
        "embedder_kind": payload.get("embedder_kind", "hash"),
        "embed_model": payload.get("embed_model"),
        "text_embedder_kind": payload.get("text_embedder_kind"),
        "text_embed_model": payload.get("text_embed_model"),
        "device": payload.get("device", "cpu"),
        "embedding_device": payload.get("embedding_device"),
        "max_pixels": int(payload.get("max_pixels", 768 * 768)),
        "priority": int(payload.get("priority", 100)),
        "target_path": str(candidate),
    }
    return enqueue_photo_index(db, task_payload)


def _has_active_photo_task(db) -> bool:
    return (
        db.scalar(
            select(ProcessingTask.id).where(
                ProcessingTask.pipeline_stage == "index_photos",
                ProcessingTask.state.in_(["queued", "running"]),
            )
        )
        is not None
    )


def enqueue_next_face_batch(db, payload: dict[str, Any]) -> ProcessingTask | None:
    if _has_active_face_task(db):
        return None
    limit = max(1, int(payload.get("limit", 20)))
    if not _first_unindexed_face(db):
        return None
    return enqueue_face_index(db, {"limit": limit, "force": False, "priority": int(payload.get("priority", 90))})


def _has_active_face_task(db) -> bool:
    return (
        db.scalar(
            select(ProcessingTask.id).where(
                ProcessingTask.pipeline_stage == "index_faces",
                ProcessingTask.state.in_(["queued", "running"]),
            )
        )
        is not None
    )


def _first_unindexed_face(db) -> int | None:
    return db.scalar(
        select(File.id)
        .join(Image, Image.file_id == File.id)
        .where(
            File.ingestion_status == "indexed",
            ~exists(select(FaceDetection.id).where(FaceDetection.file_id == File.id)),
            (Image.face_analysis_status.is_(None)) | (Image.face_analysis_status.in_(["pending", "failed"])),
        )
        .order_by(File.id.asc())
    )


def _first_unindexed_image(db, root: Path, probe_limit: int = 1000, payload: dict | None = None) -> Path | None:
    source = _ensure_photo_source(db, root)
    cursor = scan_cursor(db, source.id, "photos")
    scanned = 0
    for image_path in iter_source_files(root, SUPPORTED_IMAGES, after=cursor.path):
        if scanned >= max(1, probe_limit):
            db.commit()
            return None
        scanned += 1
        cursor.path = str(image_path)
        file = db.scalar(select(File).where(File.canonical_path == str(image_path)))
        if (
            not file
            or file.ingestion_status != "indexed"
            or file.sha256 != sha256_file(image_path)
            or not _image_context_current(db, file)
            or not _image_metadata_context_current(db, file)
            or (
                payload is not None
                and (stored_profile := db.scalar(select(Image.index_profile).where(Image.file_id == file.id)))
                is not None
                and stored_profile != _index_profile(payload)
            )
        ):
            db.commit()
            return image_path
    cursor.path = ""
    db.commit()
    return None


def _task_payload(db, task_id: int) -> dict[str, Any]:
    row = db.scalar(select(TaskPayload).where(TaskPayload.task_id == task_id))
    if not row:
        return {}
    return json.loads(row.payload_json)


def _ensure_photo_source(db, root: Path) -> Source:
    source = db.scalar(select(Source).where(Source.root_path == str(root)))
    if source:
        return source
    source = Source(name=f"Photos ({root})", root_path=str(root), kind="images", enabled=True)
    db.add(source)
    db.flush()
    return source


def _upsert_image_file(db, source: Source, image_path: Path, meta) -> File:
    file = db.scalar(select(File).where(File.canonical_path == str(image_path)))
    if not file:
        file = File(
            source_id=source.id,
            canonical_path=str(image_path),
            windows_path=to_windows_path(image_path),
            file_type=image_path.suffix.lower().lstrip("."),
        )
        db.add(file)
        db.flush()
    file.source_id = source.id
    file.windows_path = to_windows_path(image_path)
    file.file_type = image_path.suffix.lower().lstrip(".")
    file.byte_size = meta.size_bytes
    file.modified_time = datetime.fromtimestamp(image_path.stat().st_mtime)
    file.sha256 = meta.sha256
    file.ingestion_status = "indexed"
    file.error = None
    file.updated_at = datetime.utcnow()
    return file


def _upsert_image_context(db, file: File, meta, thumb: Path | None, context: dict[str, object]) -> None:
    image = db.scalar(select(Image).where(Image.file_id == file.id))
    if not image:
        image = Image(file_id=file.id)
        db.add(image)
        db.flush()
    image.width = meta.width
    image.height = meta.height
    image.exif_time = meta.exif_taken_at
    image.gps_lat = str(meta.gps_lat) if meta.gps_lat is not None else None
    image.gps_lon = str(meta.gps_lon) if meta.gps_lon is not None else None
    image.thumb_path = str(thumb) if thumb else None
    image.caption = str(context.get("caption") or "")
    image.ocr_text = str(context.get("ocr_text") or context.get("text_in_image") or "")
    image.tags = json.dumps(context.get("tags") or [], ensure_ascii=False)
    image.context_json = json.dumps(context, ensure_ascii=False)
    image.context_version = PHOTO_CONTEXT_VERSION
    image.visual_analysis_status = "captioned" if image.caption else "metadata_only"
    image.indexed_at = datetime.utcnow()


def _attach_people_context(db, file: File, context: dict[str, object]) -> None:
    detected_people = people_for_file(db, file.id)
    if not detected_people:
        return
    existing_people = context.get("people")
    people_text = _as_text_list(existing_people)
    for person in detected_people:
        name = str(person["name"])
        if name not in people_text:
            people_text.append(name)
    context["people"] = people_text
    context["person_clusters"] = detected_people
    metadata = context.get("metadata")
    if isinstance(metadata, dict):
        metadata["person_ids"] = [str(person.get("label") or "") for person in detected_people if person.get("label")]
    image = db.scalar(select(Image).where(Image.file_id == file.id))
    if image:
        image.context_json = json.dumps(context, ensure_ascii=False)


def _attach_metadata_context(db, file: File, meta, context: dict[str, object]) -> None:
    candidates = upsert_photo_metadata_candidates(db, file, meta, context)
    serialized = [
        {
            "kind": candidate.kind,
            "value": candidate.value,
            "source": candidate.source,
            "confidence": candidate.confidence,
        }
        for candidate in candidates
    ]
    context["metadata_candidates"] = serialized
    context["metadata_text"] = candidates_text(candidates)
    metadata = context.get("metadata")
    if isinstance(metadata, dict):
        metadata.update(
            {
                "captured_at": meta.exif_taken_at.isoformat() if meta.exif_taken_at else None,
                "latitude": meta.gps_lat,
                "longitude": meta.gps_lon,
                "width": meta.width,
                "height": meta.height,
                "file_type": file.file_type,
            }
        )
    image = db.scalar(select(Image).where(Image.file_id == file.id))
    if image:
        if image.gps_lat and image.gps_lon:
            image.location = ensure_local_geocode_cache(db, image.gps_lat, image.gps_lon)
        image.context_json = json.dumps(context, ensure_ascii=False)


def _backfill_metadata_context(db, file: File, image_path: Path, meta=None) -> bool:
    image = db.scalar(select(Image).where(Image.file_id == file.id))
    if not image:
        return False
    context = _parse_context_json(image.context_json)
    if "metadata_candidates" in context and context.get("metadata_text") is not None:
        return False
    if meta is None:
        meta = inspect_image(image_path)
    _attach_metadata_context(db, file, meta, context)
    return True


def _image_context_current(db, file: File) -> bool:
    image = db.scalar(select(Image).where(Image.file_id == file.id))
    return bool(image and image.context_version == PHOTO_CONTEXT_VERSION)


def _image_metadata_context_current(db, file: File) -> bool:
    image = db.scalar(select(Image).where(Image.file_id == file.id))
    if not image:
        return False
    context = _parse_context_json(image.context_json)
    return "metadata_candidates" in context


def _parse_context_json(value: str | None) -> dict[str, object]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _normalize_context(analysis: dict[str, object]) -> dict[str, object]:
    context = dict(analysis)
    context.setdefault("caption", "")
    context.setdefault("dense_caption", context.get("caption") or "")
    context.setdefault("objects", [])
    context.setdefault("people", [])
    context.setdefault("scene", "")
    context.setdefault("activities", [])
    context.setdefault("text_in_image", context.get("ocr_text") or "")
    context.setdefault("time_clues", [])
    context.setdefault("location_clues", [])
    context.setdefault("location", "")
    context.setdefault("events", [])
    context.setdefault("event", "")
    context.setdefault("time", "")
    context.setdefault("colors", [])
    context.setdefault("style", "")
    context.setdefault("quality", "")
    context.setdefault("visual_attributes", [])
    context.setdefault("search_phrases", [])
    context.setdefault("negative_clues", [])
    context.setdefault("tags", [])
    context["ocr_text"] = str(context.get("ocr_text") or context.get("text_in_image") or "")
    for key in [
        "objects",
        "people",
        "activities",
        "time_clues",
        "location_clues",
        "events",
        "colors",
        "visual_attributes",
        "search_phrases",
        "negative_clues",
        "tags",
    ]:
        value = context.get(key)
        if isinstance(value, list):
            context[key] = [str(item).strip() for item in value if str(item).strip()][:32]
        elif value:
            context[key] = [str(value).strip()]
        else:
            context[key] = []
    if not context["tags"]:
        derived: list[str] = []
        for key in ["scene", "objects", "activities", "events", "location_clues", "time_clues"]:
            value = context.get(key)
            if isinstance(value, list):
                derived.extend(str(item) for item in value)
            elif value:
                derived.append(str(value))
        context["tags"] = [tag for tag in derived if 1 < len(tag) < 24][:24]
    if not context["search_phrases"]:
        phrases = [str(context.get("caption") or "")[:48]]
        phrases.extend(str(tag) for tag in context["tags"])
        context["search_phrases"] = [phrase for phrase in phrases if 1 < len(phrase) < 48][:12]
    return context


def _context_vector_texts(context: dict[str, object], image_path: Path, meta) -> dict[str, str]:
    if isinstance(context.get("observations"), dict):
        return _v2_context_vector_texts(context, image_path, meta)

    caption = str(context.get("caption") or "")
    dense_caption = str(context.get("dense_caption") or caption)
    ocr = str(context.get("ocr_text") or context.get("text_in_image") or "")
    tags = " ".join(_as_text_list(context.get("tags")))
    objects = " ".join(_as_text_list(context.get("objects")))
    people = " ".join(_as_text_list(context.get("people")))
    person_clusters = " ".join(_person_cluster_terms(context.get("person_clusters")))
    activities = " ".join(_as_text_list(context.get("activities")))
    time_clues = " ".join(_as_text_list(context.get("time_clues")))
    location_clues = " ".join(_as_text_list(context.get("location_clues")))
    events = " ".join(_as_text_list(context.get("events")))
    colors = " ".join(_as_text_list(context.get("colors")))
    search_phrases = " ".join(_as_text_list(context.get("search_phrases")))
    negative_clues = " ".join(_as_text_list(context.get("negative_clues")))
    visual_attributes = " ".join(_as_text_list(context.get("visual_attributes")))
    location = " ".join([str(context.get("location") or ""), location_clues])
    event = " ".join([str(context.get("event") or ""), events])
    time_text = " ".join([str(context.get("time") or ""), time_clues])
    metadata_candidates = str(context.get("metadata_text") or "")
    metadata = " ".join(
        part
        for part in [
            image_path.name,
            str(context.get("scene") or ""),
            str(context.get("style") or ""),
            str(context.get("quality") or ""),
            time_text,
            location,
            event,
            metadata_candidates,
            f"{meta.exif_taken_at:%Y-%m-%d}" if meta.exif_taken_at else "",
            f"gps {meta.gps_lat:.6f} {meta.gps_lon:.6f}"
            if meta.gps_lat is not None and meta.gps_lon is not None
            else "",
        ]
        if part
    )
    aggregate = " ".join(
        [
            caption,
            dense_caption,
            search_phrases,
            tags,
            objects,
            people,
            person_clusters,
            activities,
            ocr,
            metadata,
            colors,
            visual_attributes,
        ]
    )
    return {
        "text": aggregate,
        "caption": caption,
        "dense_caption": dense_caption,
        "ocr": ocr,
        "tags": tags,
        "objects": objects,
        "people": " ".join([people, person_clusters]),
        "activities": activities,
        "location": location,
        "time": time_text,
        "event": event,
        "visual_attributes": visual_attributes,
        "search_phrases": search_phrases,
        "negative_clues": negative_clues,
        "metadata": metadata,
    }


def _v2_context_vector_texts(context: dict[str, object], image_path: Path, meta) -> dict[str, str]:
    vector_texts = build_context_vector_texts(context)
    observations = context.get("observations") if isinstance(context.get("observations"), dict) else {}
    metadata = context.get("metadata") if isinstance(context.get("metadata"), dict) else {}
    inferences = context.get("inferences") if isinstance(context.get("inferences"), dict) else {}
    people = " ".join(_as_text_list(context.get("people")) + _person_cluster_terms(context.get("person_clusters")))
    objects = " ".join(_as_text_list(observations.get("objects")))
    activities = " ".join(_as_text_list(observations.get("actions")))
    high_confidence = _high_confidence_inference_text(inferences)
    deterministic = " ".join(
        part
        for part in [
            image_path.name,
            str(context.get("metadata_text") or ""),
            str(metadata.get("city") or ""),
            str(metadata.get("location_name") or ""),
            str(metadata.get("captured_at") or ""),
            f"{meta.exif_taken_at:%Y-%m-%d}" if meta.exif_taken_at else "",
            f"gps {meta.gps_lat:.6f} {meta.gps_lon:.6f}"
            if meta.gps_lat is not None and meta.gps_lon is not None
            else "",
        ]
        if part
    )
    aggregate = " ".join(
        part
        for part in [vector_texts.caption_text, vector_texts.context_text, vector_texts.ocr_text, people, deterministic]
        if part
    )
    return {
        "text": aggregate,
        "caption": vector_texts.caption_text,
        "dense_caption": vector_texts.caption_text,
        "ocr": vector_texts.ocr_text,
        "tags": " ".join(vector_texts.lexical_terms),
        "objects": objects,
        "people": people,
        "activities": activities,
        "location": high_confidence,
        "time": high_confidence,
        "event": high_confidence,
        "visual_attributes": " ".join(_as_text_list(observations.get("visual_attributes"))),
        "search_phrases": vector_texts.caption_text,
        "negative_clues": "",
        "metadata": deterministic,
    }


def _high_confidence_inference_text(inferences: dict[str, object], threshold: float = 0.65) -> str:
    values: list[str] = []
    for key in ["possible_events", "possible_locations", "possible_time_periods", "possible_relationships"]:
        candidates = inferences.get(key)
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            try:
                confidence = float(candidate.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            value = str(candidate.get("value") or "").strip()
            if value and confidence >= threshold:
                values.append(value)
    return " ".join(dict.fromkeys(values))


def _as_text_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


@dataclass(frozen=True)
class _StoredImageMeta:
    exif_taken_at: datetime | None
    gps_lat: float | None
    gps_lon: float | None


def _stored_meta_from_image(image: Image) -> _StoredImageMeta:
    return _StoredImageMeta(
        exif_taken_at=image.exif_time,
        gps_lat=_parse_float(image.gps_lat),
        gps_lon=_parse_float(image.gps_lon),
    )


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _person_cluster_terms(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    terms: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        for key in ["name", "label"]:
            text = str(item.get(key) or "").strip()
            if text:
                terms.append(text)
        aliases = item.get("aliases")
        if isinstance(aliases, list):
            terms.extend(str(alias).strip() for alias in aliases if str(alias).strip())
    return terms


def _vector_namespace(kind: str, namespace: str) -> str:
    if kind in {"bge", "bge-m3"}:
        return f"bge:{namespace}"
    if kind == "clip":
        return f"clip:{namespace}"
    if kind == "hash":
        return f"hash:{namespace}"
    return f"{kind}:{namespace}"[:40]


def configure_logging() -> None:
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "index.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Process at most one queued task and exit.")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--auto-enqueue", action="store_true", help="Continuously enqueue the next photo batch when idle."
    )
    parser.add_argument("--limit", type=int, default=100, help="Max files scanned per auto-enqueued task.")
    parser.add_argument("--max-indexed", type=int, default=10, help="Max newly indexed photos per auto-enqueued task.")
    parser.add_argument(
        "--probe-limit", type=int, default=1000, help="Max source paths probed while looking for new photos."
    )
    parser.add_argument("--captioner-kind", default="basic", choices=["basic", "qwen-vl"])
    parser.add_argument("--embedder-kind", default="hash", choices=["hash", "clip"])
    parser.add_argument("--text-embedder-kind", default=None, choices=["hash", "clip", "bge"])
    parser.add_argument("--text-embed-model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configure_logging()
    init_db()
    worker = ImageWorker()
    signal.signal(signal.SIGUSR1, lambda *_: worker.stop_requested.set())
    if args.once:
        worker.run_once()
    else:
        worker.run_forever(
            poll_seconds=args.poll_seconds,
            auto_enqueue=args.auto_enqueue,
            auto_payload={
                "limit": args.limit,
                "max_indexed": args.max_indexed,
                "probe_limit": args.probe_limit,
                "captioner_kind": args.captioner_kind,
                "embedder_kind": args.embedder_kind,
                "text_embedder_kind": args.text_embedder_kind,
                "text_embed_model": args.text_embed_model,
                "device": args.device,
                "force": args.force,
            },
        )


if __name__ == "__main__":
    main()
