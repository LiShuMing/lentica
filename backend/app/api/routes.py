from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from backend.app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatSource,
    DocumentIndexRequest,
    FaceIndexRequest,
    FaceSampleInfo,
    FileInfo,
    HealthResponse,
    ModelInfo,
    PersonClusterInfo,
    PersonUpdateRequest,
    PhotoIndexRequest,
    PhotoSearchRequest,
    PhotoSearchResponse,
    PhotoSearchV2Request,
    PhotoSearchV2Response,
    QueueControlResponse,
    QueueNextPhotoRequest,
    QueueSettingsUpdate,
    QueueStatusResponse,
    ScanResponse,
    SettingsResponse,
    SourceInfo,
    TaskInfo,
    VisionStatusResponse,
)
from backend.app.core.config import CLIP_MODEL, DATA_DIR, DB_PATH, TEXT_MODEL, USER_ENV_PATH, get_settings
from backend.app.db.models import (
    Chunk,
    FaceDetection,
    File,
    ModelProfile,
    PersonAlias,
    PersonCluster,
    ProcessingRun,
    ProcessingTask,
    Source,
    TaskPayload,
)
from backend.app.db.session import get_db
from backend.app.ingestion.faces import refresh_people_context_for_cluster
from backend.app.ingestion.scanner import ensure_default_sources, scan_documents
from backend.app.llm.client import LocalAnswerer
from backend.app.models.local_clip import Embedder, build_embedder
from backend.app.models.local_registry import list_local_models
from backend.app.models.vision_service import vision_status as get_vision_service_status
from backend.app.retrieval import asset_search, service
from backend.app.workers.image_worker import (
    enqueue_document_index,
    enqueue_face_index,
    enqueue_next_photo_batch,
    enqueue_person_vector_refresh,
    enqueue_photo_index,
)
from backend.app.workers.queue_controller import queue_controller

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/api/settings", response_model=SettingsResponse)
def settings() -> SettingsResponse:
    loaded = get_settings()
    return SettingsResponse(
        env_file_present=USER_ENV_PATH.exists(),
        runtime_mode=loaded.runtime.mode,
        external_api_enabled=loaded.runtime.external_api_enabled,
        legacy_llm_env_present=loaded.runtime.legacy_llm_env_present,
        documents_source=str(loaded.documents_source),
        documents_source_exists=loaded.documents_source.exists(),
        photos_source=str(loaded.photos_source),
        photos_source_exists=loaded.photos_source.exists(),
        database_path=str(DB_PATH),
    )


@router.get("/api/models", response_model=list[ModelInfo])
async def models(db: Session = Depends(get_db)) -> list[ModelInfo]:
    discovered = [ModelInfo(**model.to_api()) for model in list_local_models()]
    for model in discovered:
        profile = db.scalar(select(ModelProfile).where(ModelProfile.model_id == model.id))
        safe_metadata = {
            "capabilities": model.capabilities,
            "runtime_type": model.runtime_type,
            "model_path": model.model_path,
            "device": model.device,
            "dtype": model.dtype,
            "version": model.version,
            "memory_budget": model.memory_budget,
            "status": model.status,
            "notes": model.notes,
        }
        capabilities = json.dumps(safe_metadata, ensure_ascii=False)
        if profile:
            profile.role = model.role
            profile.capabilities = capabilities
            profile.enabled = model.enabled
            profile.updated_at = datetime.utcnow()
        else:
            db.add(
                ModelProfile(
                    model_id=model.id,
                    role=model.role,
                    capabilities=capabilities,
                    enabled=model.enabled,
                )
            )
    db.commit()
    return discovered


@router.get("/api/library/sources", response_model=list[SourceInfo])
def library_sources(db: Session = Depends(get_db)) -> list[SourceInfo]:
    ensure_default_sources(db)
    rows = db.execute(
        select(Source, func.count(File.id)).outerjoin(File, File.source_id == Source.id).group_by(Source.id)
    ).all()
    return [
        SourceInfo(
            id=source.id,
            name=source.name,
            root_path=source.root_path,
            kind=source.kind,
            enabled=source.enabled,
            last_scan_at=source.last_scan_at,
            file_count=count,
        )
        for source, count in rows
    ]


@router.post("/api/library/scan", response_model=ScanResponse)
def scan_library(
    source_id: int | None = None,
    max_files: int = 100,
    max_mb: int = 20,
    db: Session = Depends(get_db),
) -> ScanResponse:
    job = scan_documents(
        db,
        source_id=source_id,
        max_files=None if max_files == 0 else max_files,
        max_bytes=max(1, max_mb) * 1024 * 1024,
    )
    return ScanResponse(
        id=job.id,
        status=job.status,
        total_files=job.total_files,
        indexed_files=job.indexed_files,
        skipped_files=job.skipped_files,
        failed_files=job.failed_files,
        error=job.error,
    )


@router.post("/api/library/index", response_model=TaskInfo)
def enqueue_library_index(request: DocumentIndexRequest, db: Session = Depends(get_db)) -> TaskInfo:
    task = enqueue_document_index(db, request.model_dump())
    return _task_info(db, task)


@router.get("/api/library/files", response_model=list[FileInfo])
def library_files(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)) -> list[FileInfo]:
    rows = db.execute(
        select(File, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.file_id == File.id)
        .group_by(File.id)
        .order_by(File.updated_at.desc())
        .offset(offset)
        .limit(min(limit, 500))
    ).all()
    return [
        FileInfo(
            id=file.id,
            canonical_path=file.canonical_path,
            windows_path=file.windows_path,
            file_type=file.file_type,
            byte_size=file.byte_size,
            modified_time=file.modified_time,
            ingestion_status=file.ingestion_status,
            error=file.error,
            chunk_count=chunk_count,
            title=file.document.title if file.document else Path(file.canonical_path).stem,
        )
        for file, chunk_count in rows
    ]


@router.post("/api/photos/index", response_model=TaskInfo)
def enqueue_photos(request: PhotoIndexRequest, db: Session = Depends(get_db)) -> TaskInfo:
    payload = request.model_dump()
    payload["root"] = str(get_settings().photos_source)
    task = enqueue_photo_index(db, payload)
    return _task_info(db, task)


@router.post("/api/photos/index-next", response_model=QueueControlResponse)
def enqueue_next_photo(request: QueueNextPhotoRequest, db: Session = Depends(get_db)) -> QueueControlResponse:
    payload = request.model_dump()
    payload["root"] = str(get_settings().photos_source)
    task = enqueue_next_photo_batch(db, payload)
    if not task:
        return QueueControlResponse(
            queued=False,
            message="No unindexed photo found within probe limit, or an active photo task already exists.",
        )
    return QueueControlResponse(queued=True, message="Queued next unindexed photo batch.", task=_task_info(db, task))


@router.post("/api/photos/search", response_model=PhotoSearchResponse)
def search_photos(request: PhotoSearchRequest, db: Session = Depends(get_db)) -> PhotoSearchResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    result = _product_search(db, request, ["photo", "screenshot"])
    return PhotoSearchResponse(answer=result.answer, hits=[asdict(hit) for hit in result.photos])


@router.post("/api/v2/search", response_model=PhotoSearchV2Response)
def search_assets_v2(request: PhotoSearchV2Request, db: Session = Depends(get_db)) -> PhotoSearchV2Response:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    result = _product_search(db, request, request.asset_types)
    return PhotoSearchV2Response(
        answer=result.answer,
        hits=[asdict(hit) for hit in result.photos],
        document_hits=[_document_payload(row) for row in result.documents],
        results=[
            dict(
                file_id=hit.file_id,
                asset_type=hit.asset_type,
                rank=hit.rank,
                score=hit.score,
                unit_ids=list(hit.unit_ids),
                photo=asdict(hit.photo) if hit.photo is not None else None,
                documents=[_document_payload(row) for row in hit.documents],
            )
            for hit in result.results
        ],
        plan=result.plan.debug_payload(),
        debug=result.debug if request.debug else None,
    )


def _product_search(db: Session, request: PhotoSearchRequest, asset_types: list[str]) -> asset_search.AssetSearchResult:
    """Defer image model selection until the shared service chooses an image plan."""
    return asset_search.search_assets(
        db,
        request.query.strip(),
        limit=request.limit,
        asset_types=asset_types,
        embedders=lambda: _photo_search_embedders(request),
    )


def _document_payload(row: service.RetrievedSource) -> dict[str, object]:
    """Project final selected chunk records into the existing citation contract."""
    return dict(
        file_id=row.file_id,
        chunk_id=row.chunk_id,
        path=row.path,
        windows_path=row.windows_path,
        page_or_segment=row.page_or_segment,
        score=row.score,
        excerpt=row.text[:360],
    )


@router.get("/api/photos/thumb/{name}")
def photo_thumbnail(name: str) -> FileResponse:
    if "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid thumbnail name")
    path = DATA_DIR / "cache" / "thumbs" / name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(path)


@router.get("/api/tasks", response_model=list[TaskInfo])
def list_tasks(limit: int = 50, db: Session = Depends(get_db)) -> list[TaskInfo]:
    tasks = db.scalars(select(ProcessingTask).order_by(ProcessingTask.id.desc()).limit(min(limit, 200))).all()
    return [_task_info(db, task) for task in tasks]


@router.get("/api/queue/status", response_model=QueueStatusResponse)
def queue_status() -> QueueStatusResponse:
    return QueueStatusResponse(**queue_controller.status())


@router.put("/api/queue/settings", response_model=QueueStatusResponse)
def update_queue_settings(request: QueueSettingsUpdate) -> QueueStatusResponse:
    queue_controller.update_settings(request.model_dump(exclude_unset=True))
    return QueueStatusResponse(**queue_controller.status())


@router.get("/api/vision/status", response_model=VisionStatusResponse)
def vision_status() -> VisionStatusResponse:
    settings = queue_controller.settings()
    if settings.vision_backend == "vllm":
        status = get_vision_service_status(
            settings.vision_endpoint,
            settings.vision_model,
            timeout_seconds=min(10, settings.vision_timeout_seconds),
        )
    else:
        status = {
            "backend": settings.vision_backend,
            "endpoint": "",
            "model": settings.vision_model,
            "healthy": True,
            "models": [],
            "error": None,
        }
    status["fallback"] = settings.vision_fallback
    return VisionStatusResponse(**status)


@router.get("/api/people", response_model=list[PersonClusterInfo])
def list_people(db: Session = Depends(get_db)) -> list[PersonClusterInfo]:
    clusters = db.scalars(
        select(PersonCluster).order_by(PersonCluster.ignored.asc(), PersonCluster.face_count.desc())
    ).all()
    aliases_by_cluster = _aliases_by_cluster(db, [cluster.id for cluster in clusters])
    result: list[PersonClusterInfo] = []
    for cluster in clusters:
        face_rows = (
            db.execute(
                select(FaceDetection)
                .where(FaceDetection.cluster_id == cluster.id)
                .order_by(FaceDetection.detection_score.desc())
                .limit(8)
            )
            .scalars()
            .all()
        )
        photo_count = db.scalar(
            select(func.count(func.distinct(FaceDetection.file_id))).where(FaceDetection.cluster_id == cluster.id)
        )
        samples = [
            FaceSampleInfo(
                face_id=face.id,
                file_id=face.file_id,
                bbox=_parse_float_list(face.bbox_json),
                detection_score=float(face.detection_score) if face.detection_score else None,
                thumb_url=f"/api/people/faces/{face.id}/thumb",
            )
            for face in face_rows
        ]
        result.append(
            PersonClusterInfo(
                id=cluster.id,
                label=cluster.label,
                display_name=cluster.display_name,
                aliases=aliases_by_cluster.get(cluster.id, []),
                face_count=cluster.face_count,
                photo_count=int(photo_count or 0),
                confidence=float(cluster.confidence) if cluster.confidence else None,
                ignored=cluster.ignored,
                samples=samples,
            )
        )
    return result


@router.post("/api/people/index", response_model=TaskInfo)
def enqueue_people_index(request: FaceIndexRequest, db: Session = Depends(get_db)) -> TaskInfo:
    task = enqueue_face_index(db, request.model_dump())
    return _task_info(db, task)


@router.put("/api/people/{cluster_id}", response_model=PersonClusterInfo)
def update_person(cluster_id: int, request: PersonUpdateRequest, db: Session = Depends(get_db)) -> PersonClusterInfo:
    cluster = db.get(PersonCluster, cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="Person cluster not found")
    if request.display_name is not None:
        cluster.display_name = request.display_name.strip() or None
    if request.aliases is not None:
        db.execute(delete(PersonAlias).where(PersonAlias.cluster_id == cluster.id))
        for alias in _clean_aliases(request.aliases):
            db.add(PersonAlias(cluster_id=cluster.id, alias=alias))
    if request.ignored is not None:
        cluster.ignored = request.ignored
    cluster.updated_at = datetime.utcnow()
    refresh_people_context_for_cluster(db, cluster.id)
    enqueue_person_vector_refresh(
        db,
        {
            "cluster_id": cluster.id,
            "text_embedder_kind": "bge",
            "text_embed_model": "BAAI/bge-m3",
            "embedding_device": "cpu",
            "priority": 120,
        },
    )
    return next(item for item in list_people(db) if item.id == cluster_id)


@router.get("/api/people/faces/{face_id}/thumb")
def face_thumbnail(face_id: int, db: Session = Depends(get_db)) -> StreamingResponse:
    face = db.get(FaceDetection, face_id)
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")
    file = db.get(File, face.file_id)
    if not file:
        raise HTTPException(status_code=404, detail="Source image not found")
    try:
        from PIL import Image as PilImage

        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except Exception:
            pass

        bbox = _parse_float_list(face.bbox_json)
        if len(bbox) != 4:
            raise ValueError("Invalid face bbox")
        with PilImage.open(file.canonical_path) as image:
            left, top, right, bottom = bbox
            pad_x = (right - left) * 0.22
            pad_y = (bottom - top) * 0.30
            crop = image.crop(
                (
                    max(0, int(left - pad_x)),
                    max(0, int(top - pad_y)),
                    min(image.width, int(right + pad_x)),
                    min(image.height, int(bottom + pad_y)),
                )
            )
            crop.thumbnail((256, 256))
            output = BytesIO()
            crop.convert("RGB").save(output, format="JPEG", quality=88)
            output.seek(0)
        return StreamingResponse(output, media_type="image/jpeg")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"Could not render face thumbnail: {exc}") from exc


@router.post("/api/tasks/{task_id}/cancel", response_model=TaskInfo)
def cancel_task(task_id: int, db: Session = Depends(get_db)) -> TaskInfo:
    task = db.get(ProcessingTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.state != "queued":
        raise HTTPException(
            status_code=409, detail=f"Only queued tasks can be cancelled; current state is {task.state}"
        )
    task.state = "cancelled"
    task.error = "Cancelled by user"
    db.commit()
    db.refresh(task)
    return _task_info(db, task)


@router.post("/api/tasks/{task_id}/retry", response_model=TaskInfo)
def retry_task(task_id: int, db: Session = Depends(get_db)) -> TaskInfo:
    task = db.get(ProcessingTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.state not in {"failed", "cancelled"}:
        raise HTTPException(
            status_code=409, detail=f"Only failed or cancelled tasks can be retried; current state is {task.state}"
        )
    payload_row = db.scalar(select(TaskPayload).where(TaskPayload.task_id == task.id))
    payload = json.loads(payload_row.payload_json) if payload_row else {}
    if task.pipeline_stage == "index_photos":
        new_task = enqueue_photo_index(db, payload)
    elif task.pipeline_stage == "index_documents":
        new_task = enqueue_document_index(db, payload)
    elif task.pipeline_stage == "index_faces":
        new_task = enqueue_face_index(db, payload)
    elif task.pipeline_stage == "refresh_person_vectors":
        new_task = enqueue_person_vector_refresh(db, payload)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported task stage: {task.pipeline_stage}")
    return _task_info(db, new_task)


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    retrieved = asset_search.search_assets(db, question, limit=request.limit, asset_types=["document"]).documents
    evidence = [
        {
            "file_id": item.file_id,
            "chunk_id": item.chunk_id,
            "path": item.path,
            "windows_path": item.windows_path,
            "page_or_segment": item.page_or_segment,
            "text": item.text,
            "score": item.score,
        }
        for item in retrieved
    ]
    answer = await LocalAnswerer().answer(question, evidence)
    sources = [
        ChatSource(
            file_id=item.file_id,
            chunk_id=item.chunk_id,
            path=item.path,
            windows_path=item.windows_path,
            page_or_segment=item.page_or_segment,
            score=item.score,
            excerpt=item.text[:360],
        )
        for item in retrieved
    ]
    confidence = "low" if not sources else ("high" if sources[0].score > 0.35 else "medium")
    return ChatResponse(
        answer=answer,
        confidence=confidence,
        answer_mode="extractive_local",
        sources=sources,
        trace=[
            "classify: document_search",
            "rewrite: deterministic",
            "retrieve: staged_hybrid_document_tfidf",
            "answer: extractive_local",
        ],
    )


def _task_info(db: Session, task: ProcessingTask) -> TaskInfo:
    payload_row = db.scalar(select(TaskPayload).where(TaskPayload.task_id == task.id))
    latest_run = db.scalar(
        select(ProcessingRun).where(ProcessingRun.task_id == task.id).order_by(ProcessingRun.id.desc())
    )
    latest_run_payload = None
    if latest_run:
        stats = _parse_json_object(latest_run.output_artifact_ids)
        duration_seconds = None
        throughput = {}
        if latest_run.started_at:
            end_time = latest_run.finished_at or datetime.utcnow()
            duration_seconds = max(0.0, (end_time - latest_run.started_at).total_seconds())
            if duration_seconds > 0 and stats:
                for key in ["scanned", "indexed", "skipped", "failed", "total_files", "indexed_files"]:
                    value = stats.get(key)
                    if isinstance(value, (int, float)):
                        throughput[f"{key}_per_second"] = round(float(value) / duration_seconds, 3)
        computational_load = _computational_load(task.pipeline_stage, stats)
        latest_run_payload = {
            "id": latest_run.id,
            "started_at": latest_run.started_at,
            "finished_at": latest_run.finished_at,
            "duration_seconds": duration_seconds,
            "model_profile": latest_run.model_profile,
            "stats": stats,
            "throughput": throughput,
            "computational_load": computational_load,
            "raw_output": latest_run.output_artifact_ids,
            "error": latest_run.error,
        }
    return TaskInfo(
        id=task.id,
        pipeline_stage=task.pipeline_stage,
        state=task.state,
        priority=task.priority,
        attempt_count=task.attempt_count,
        worker_id=task.worker_id,
        error=task.error,
        payload=json.loads(payload_row.payload_json) if payload_row else {},
        latest_run=latest_run_payload,
    )


def _photo_search_embedders(request: PhotoSearchRequest) -> tuple[Embedder, Embedder]:
    if request.embedder_kind == "hash":
        embedder = _cached_embedder("hash", request.embed_model or "", request.device)
        return embedder, embedder
    visual = _cached_embedder("clip", request.embed_model or CLIP_MODEL, request.device)
    if request.embedder_kind in {"hybrid", "bge", "clip"}:
        try:
            text = _cached_embedder(
                "bge",
                request.text_embed_model or TEXT_MODEL,
                request.device,
            )
        except Exception:
            try:
                text = _cached_embedder("bge", "BAAI/bge-small-zh-v1.5", request.device)
            except Exception:
                text = visual
        return visual, text
    raise HTTPException(status_code=400, detail=f"Unsupported photo embedder: {request.embedder_kind}")


@lru_cache(maxsize=6)
def _cached_embedder(kind: str, model_name: str, device: str) -> Embedder:
    """Keep retrieval models warm; callers select CPU when vLLM owns GPU memory."""
    return build_embedder(kind, model_name, device)


def _parse_json_object(value: str | None) -> dict[str, object]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _parse_float_list(value: str | None) -> list[float]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return []


def _aliases_by_cluster(db: Session, cluster_ids: list[int]) -> dict[int, list[str]]:
    if not cluster_ids:
        return {}
    rows = db.execute(
        select(PersonAlias.cluster_id, PersonAlias.alias).where(PersonAlias.cluster_id.in_(cluster_ids))
    ).all()
    result: dict[int, list[str]] = {}
    for cluster_id, alias in rows:
        text = str(alias or "").strip()
        if text:
            result.setdefault(int(cluster_id), []).append(text)
    return result


def _clean_aliases(aliases: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        text = str(alias).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text[:120])
    return result[:24]


def _computational_load(stage: str, stats: dict[str, object]) -> dict[str, object]:
    if stage == "index_photos":
        indexed = int(stats.get("indexed", 0) or 0)
        scanned = int(stats.get("scanned", 0) or 0)
        return {
            "source_items_scanned": scanned,
            "images_captioned": indexed,
            "metadata_backfilled": int(stats.get("metadata_backfilled", 0) or 0),
            "image_vectors_written": indexed,
            "text_vectors_written": indexed,
            "total_vectors_written": indexed * 2,
        }
    if stage == "index_documents":
        total = int(stats.get("total_files", 0) or 0)
        indexed = int(stats.get("indexed_files", 0) or 0)
        skipped = int(stats.get("skipped_files", 0) or 0)
        return {
            "source_files_seen": total,
            "documents_indexed": indexed,
            "documents_skipped": skipped,
        }
    if stage == "index_faces":
        return {
            "photos_seen": int(stats.get("scanned", 0) or 0),
            "photos_processed": int(stats.get("indexed", 0) or 0),
            "faces_detected": int(stats.get("faces", 0) or 0),
            "clusters_created": int(stats.get("face_clusters_created", 0) or 0),
            "face_unavailable": int(stats.get("face_unavailable", 0) or 0),
        }
    if stage == "refresh_person_vectors":
        return {
            "photos_seen": int(stats.get("scanned", 0) or 0),
            "photos_refreshed": int(stats.get("indexed", 0) or 0),
            "text_vectors_written": int(stats.get("vectors_written", 0) or 0),
            "failed": int(stats.get("failed", 0) or 0),
        }
    return {}
