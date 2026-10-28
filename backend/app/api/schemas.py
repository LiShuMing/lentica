from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str


class SettingsResponse(BaseModel):
    env_file_present: bool
    runtime_mode: str
    external_api_enabled: bool
    legacy_llm_env_present: bool
    documents_source: str
    documents_source_exists: bool
    photos_source: str
    photos_source_exists: bool
    database_path: str


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    role: str
    runtime_type: str
    model_path: str | None
    device: str
    dtype: str | None
    version: str
    memory_budget: str | None
    enabled: bool
    capabilities: dict[str, bool]
    status: str
    notes: str


class SourceInfo(BaseModel):
    id: int
    name: str
    root_path: str
    kind: str
    enabled: bool
    last_scan_at: datetime | None
    file_count: int


class FileInfo(BaseModel):
    id: int
    canonical_path: str
    windows_path: str
    file_type: str
    byte_size: int
    modified_time: datetime | None
    ingestion_status: str
    error: str | None
    chunk_count: int
    title: str | None


class ScanResponse(BaseModel):
    id: int
    status: str
    total_files: int
    indexed_files: int
    skipped_files: int
    failed_files: int
    error: str | None


class PhotoIndexRequest(BaseModel):
    limit: int | None = 50
    max_indexed: int | None = 10
    force: bool = False
    captioner_kind: str = "basic"
    vision_backend: str | None = None
    vision_endpoint: str | None = None
    vision_model: str | None = None
    vision_timeout_seconds: int | None = None
    vision_fallback: str | None = None
    embedder_kind: str = "hash"
    caption_model: str | None = None
    embed_model: str | None = None
    text_embedder_kind: str | None = None
    text_embed_model: str | None = None
    device: str = "cuda"
    embedding_device: str | None = None
    max_pixels: int = 768 * 768
    priority: int = 100


class DocumentIndexRequest(BaseModel):
    source_id: int | None = None
    max_files: int = 100
    max_mb: int = 20
    priority: int = 100


class FaceIndexRequest(BaseModel):
    limit: int = 500
    force: bool = False
    priority: int = 90


class QueueNextPhotoRequest(PhotoIndexRequest):
    probe_limit: int = 1000
    unindexed_only: bool = True


class TaskInfo(BaseModel):
    id: int
    pipeline_stage: str
    state: str
    priority: int
    attempt_count: int
    worker_id: str | None
    error: str | None
    payload: dict[str, Any] = {}
    latest_run: dict[str, Any] | None = None


class QueueControlResponse(BaseModel):
    queued: bool
    message: str
    task: TaskInfo | None = None


class QueueSettingsUpdate(BaseModel):
    enabled: bool | None = None
    auto_enqueue_photos: bool | None = None
    auto_enqueue_faces: bool | None = None
    auto_enqueue_documents: bool | None = None
    face_limit: int | None = None
    auto_adjust_concurrency: bool | None = None
    worker_mode: str | None = None
    max_concurrency: int | None = None
    photo_limit: int | None = None
    photo_max_indexed: int | None = None
    photo_probe_limit: int | None = None
    document_max_files: int | None = None
    document_max_mb: int | None = None
    document_interval_seconds: int | None = None
    vision_backend: str | None = None
    vision_endpoint: str | None = None
    vision_model: str | None = None
    vision_timeout_seconds: int | None = None
    vision_fallback: str | None = None


class QueueStatusResponse(BaseModel):
    settings: dict[str, Any]
    gpu: dict[str, Any]
    desired_concurrency: int
    workers: list[dict[str, Any]]
    task_summary: dict[str, dict[str, int]] = {}
    last_error: str | None
    last_document_enqueue: float | None


class VisionStatusResponse(BaseModel):
    backend: str
    endpoint: str
    model: str
    healthy: bool
    models: list[str] = []
    error: str | None = None
    fallback: str


class FaceSampleInfo(BaseModel):
    face_id: int
    file_id: int
    bbox: list[float]
    detection_score: float | None
    thumb_url: str


class PersonClusterInfo(BaseModel):
    id: int
    label: str
    display_name: str | None
    aliases: list[str] = []
    face_count: int
    photo_count: int
    confidence: float | None
    ignored: bool
    samples: list[FaceSampleInfo] = []


class PersonUpdateRequest(BaseModel):
    display_name: str | None = None
    aliases: list[str] | None = None
    ignored: bool | None = None


class PhotoSearchRequest(BaseModel):
    query: str
    limit: int = 12
    embedder_kind: str = "hash"
    embed_model: str | None = None
    text_embed_model: str | None = None
    device: str = "cpu"


class PhotoHitInfo(BaseModel):
    file_id: int
    score: float
    confidence: float
    image_score: float
    text_score: float
    metadata_score: float
    namespace_scores: dict[str, float]
    matched_terms: list[str]
    match_reason: str
    path: str
    windows_path: str
    file_name: str
    thumb_path: str | None
    caption: str | None
    dense_caption: str | None
    ocr_text: str | None
    context: dict[str, Any]
    tags: list[str]
    exif_taken_at: str | None


class PhotoSearchResponse(BaseModel):
    answer: str
    hits: list[PhotoHitInfo]


class PhotoSearchV2Request(PhotoSearchRequest):
    asset_types: list[str] = []
    debug: bool = False


class PhotoSearchV2Response(PhotoSearchResponse):
    plan: dict[str, Any]
    debug: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    question: str
    limit: int = 8


class ChatSource(BaseModel):
    file_id: int
    chunk_id: int
    path: str
    windows_path: str
    page_or_segment: str
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    confidence: str
    answer_mode: str
    sources: list[ChatSource]
    trace: list[str]
    raw: dict[str, Any] = {}
