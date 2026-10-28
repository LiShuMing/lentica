# Architecture and Current Limits

The API and background controller are a modular monolith. Subprocess workers keep
inference models warm; a separate vLLM service owns Qwen-VL's GPU runtime.

## Data Flow

- Document scanner: configured roots -> SHA-256 -> native parsers -> page/segment
  chunks -> SQLite -> local TF-IDF/extractive answer with citations.
- Photo worker: configured root -> hash/EXIF/thumbnail -> Qwen context -> CLIP image
  embedding plus BGE text embeddings -> SQLite photo vectors.
- Optional faces: InsightFace observations -> incremental centroid clusters ->
  user names/aliases -> person context refresh jobs.
- Search: rule-based query plan -> independent namespace rankings -> weighted RRF
  -> candidate evidence and heuristic confidence. Confidence is not a calibrated
  probability.

The `observations`, `inferences`, and `metadata` sections distinguish visible
facts, uncertain model output, and deterministic/user-confirmed data. Legacy
payload readers remain available during schema rollout.

## Durable Work

`ProcessingTask` stores stage, idempotency key, state, priority, and attempts.
`TaskPayload` stores inputs; `ProcessingRun` records each attempt and output stats.
Workers claim queued rows using conditional updates. The backend requeues
interrupted running tasks when it restarts. Run one API/controller process;
multiple Uvicorn workers are not supported by this recovery scheme.

Cancellation currently targets queued tasks. Pausing the controller may terminate
workers; inspect interrupted state and restart/retry intentionally. This is not a
distributed queue and does not offer exactly-once execution.

## Boundaries

Original files are read-only. Application state stays in the configured data
directory. APIs are for one trusted local user and are not authenticated.

SQLite JSON vector scoring and character n-gram TF-IDF are intentionally simple
baselines. ANN indexing, document layout/OCR, calibrated confidence, hard-filter
completeness, rerankers, and large-scale quality evaluation remain future work.
See the [roadmap](roadmap/README.md) for the separate Context Plane proposal.
