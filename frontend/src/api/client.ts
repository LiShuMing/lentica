export type Settings = {
  env_file_present: boolean;
  runtime_mode: string;
  external_api_enabled: boolean;
  legacy_llm_env_present: boolean;
  documents_source: string;
  documents_source_exists: boolean;
  photos_source: string;
  photos_source_exists: boolean;
  database_path: string;
};

export type Source = {
  id: number;
  name: string;
  root_path: string;
  kind: string;
  enabled: boolean;
  last_scan_at: string | null;
  file_count: number;
};

export type FileInfo = {
  id: number;
  canonical_path: string;
  windows_path: string;
  file_type: string;
  byte_size: number;
  modified_time: string | null;
  ingestion_status: string;
  error: string | null;
  chunk_count: number;
  title: string | null;
};

export type ModelInfo = {
  id: string;
  role: string;
  runtime_type: string;
  model_path: string | null;
  device: string;
  dtype: string | null;
  version: string;
  memory_budget: string | null;
  enabled: boolean;
  capabilities: Record<string, boolean>;
  status: string;
  notes: string;
};

export type ScanResponse = {
  id: number;
  status: string;
  total_files: number;
  indexed_files: number;
  skipped_files: number;
  failed_files: number;
  error: string | null;
};

export type ChatResponse = {
  answer: string;
  confidence: string;
  answer_mode: string;
  trace: string[];
  sources: Array<{
    file_id: number;
    chunk_id: number;
    path: string;
    windows_path: string;
    page_or_segment: string;
    score: number;
    excerpt: string;
  }>;
};

export type TaskInfo = {
  id: number;
  pipeline_stage: string;
  state: string;
  priority: number;
  attempt_count: number;
  worker_id: string | null;
  error: string | null;
  payload: Record<string, unknown>;
  latest_run: {
    id: number;
    started_at: string | null;
    finished_at: string | null;
    duration_seconds: number | null;
    model_profile: string | null;
    stats: Record<string, unknown>;
    throughput: Record<string, unknown>;
    computational_load: Record<string, unknown>;
    raw_output: string | null;
    error: string | null;
  } | null;
};

export type PhotoIndexRequest = {
  limit?: number | null;
  max_indexed?: number | null;
  force?: boolean;
  captioner_kind?: string;
  vision_backend?: string | null;
  vision_endpoint?: string | null;
  vision_model?: string | null;
  vision_timeout_seconds?: number | null;
  vision_fallback?: string | null;
  embedder_kind?: string;
  text_embedder_kind?: string;
  text_embed_model?: string | null;
  device?: string;
  embedding_device?: string | null;
  probe_limit?: number;
  unindexed_only?: boolean;
};

export type QueueControlResponse = {
  queued: boolean;
  message: string;
  task: TaskInfo | null;
};

export type QueueStatus = {
  settings: {
    enabled: boolean;
    auto_enqueue_photos: boolean;
    auto_enqueue_documents: boolean;
    auto_adjust_concurrency: boolean;
    worker_mode: string;
    max_concurrency: number;
    photo_limit: number;
    photo_max_indexed: number;
    photo_probe_limit: number;
    document_max_files: number;
    document_max_mb: number;
    document_interval_seconds: number;
    vision_backend: string;
    vision_endpoint: string;
    vision_model: string;
    vision_timeout_seconds: number;
    vision_fallback: string;
  };
  gpu: Record<string, unknown>;
  desired_concurrency: number;
  workers: Array<Record<string, unknown>>;
  task_summary: Record<string, Record<string, number>>;
  last_error: string | null;
  last_document_enqueue: number | null;
};

export type VisionStatus = {
  backend: string;
  endpoint: string;
  model: string;
  healthy: boolean;
  models: string[];
  error: string | null;
  fallback: string;
};

export type PersonCluster = {
  id: number;
  label: string;
  display_name: string | null;
  aliases: string[];
  face_count: number;
  photo_count: number;
  confidence: number | null;
  ignored: boolean;
  samples: Array<{
    face_id: number;
    file_id: number;
    bbox: number[];
    detection_score: number | null;
    thumb_url: string;
  }>;
};

export type PhotoSearchResponse = {
  answer: string;
  hits: Array<{
    file_id: number;
    score: number;
    confidence: number;
    image_score: number;
    text_score: number;
    metadata_score: number;
    namespace_scores: Record<string, number>;
    matched_terms: string[];
    match_reason: string;
    path: string;
    windows_path: string;
    file_name: string;
    thumb_path: string | null;
    caption: string | null;
    dense_caption: string | null;
    ocr_text: string | null;
    context: Record<string, unknown>;
    tags: string[];
    exif_taken_at: string | null;
  }>;
};

export type AssetSearchResponse = PhotoSearchResponse & {
  plan: Record<string, unknown>;
  debug: Record<string, unknown> | null;
  document_hits: ChatResponse["sources"];
  results: Array<{
    file_id: number;
    asset_type: "photo" | "screenshot" | "document";
    rank: number;
    score: number;
    unit_ids: number[];
    photo: PhotoSearchResponse["hits"][number] | null;
    documents: ChatResponse["sources"];
  }>;
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    ...options
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json() as Promise<T>;
}

export const api = {
  searchAssets: (
    query: string,
    asset_types: Array<"photo" | "screenshot" | "document"> = [],
    embedder_kind = "hash"
  ) =>
    request<AssetSearchResponse>("/api/v2/search", {
      method: "POST", body: JSON.stringify({ query, asset_types, embedder_kind, device: "cpu" })
    }),
  settings: () => request<Settings>("/api/settings"),
  models: () => request<ModelInfo[]>("/api/models"),
  sources: () => request<Source[]>("/api/library/sources"),
  files: () => request<FileInfo[]>("/api/library/files?limit=200"),
  scan: () => request<ScanResponse>("/api/library/scan?max_files=100", { method: "POST" }),
  tasks: () => request<TaskInfo[]>("/api/tasks"),
  queueStatus: () => request<QueueStatus>("/api/queue/status"),
  visionStatus: () => request<VisionStatus>("/api/vision/status"),
  people: () => request<PersonCluster[]>("/api/people"),
  indexPeople: (payload: { limit?: number; force?: boolean; priority?: number }) =>
    request<TaskInfo>("/api/people/index", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  updatePerson: (clusterId: number, payload: { display_name?: string | null; aliases?: string[] | null; ignored?: boolean | null }) =>
    request<PersonCluster>(`/api/people/${clusterId}`, {
      method: "PUT",
      body: JSON.stringify(payload)
    }),
  updateQueueSettings: (payload: Partial<QueueStatus["settings"]>) =>
    request<QueueStatus>("/api/queue/settings", {
      method: "PUT",
      body: JSON.stringify(payload)
    }),
  indexDocuments: () =>
    request<TaskInfo>("/api/library/index", {
      method: "POST",
      body: JSON.stringify({ max_files: 100, max_mb: 20 })
    }),
  indexPhotos: (payload: PhotoIndexRequest) =>
    request<TaskInfo>("/api/photos/index", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  indexNextPhotoBatch: (payload: PhotoIndexRequest) =>
    request<QueueControlResponse>("/api/photos/index-next", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  cancelTask: (taskId: number) => request<TaskInfo>(`/api/tasks/${taskId}/cancel`, { method: "POST" }),
  retryTask: (taskId: number) => request<TaskInfo>(`/api/tasks/${taskId}/retry`, { method: "POST" }),
  searchPhotos: (query: string, embedder_kind = "hash") =>
    request<PhotoSearchResponse>("/api/photos/search", {
      method: "POST",
      body: JSON.stringify({ query, embedder_kind, device: "cpu" })
    }),
  chat: (question: string) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ question })
    })
};
