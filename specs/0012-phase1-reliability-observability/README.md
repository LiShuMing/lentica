# Phase 1: Ingestion Reliability, Real Observability, and Staged Performance Harness

## Problem and Scope

After specs `0002`–`0011` established resumable cursors, task leases/draining,
vector provenance, unified hybrid search, and strict queue settings validation,
four Phase 1 reliability and observability gaps remain from
`docs/roadmap/reliability.md` and `docs/project-status.md`:

1. `REL-4`: Standard JPEGs store `DateTimeOriginal` and `GPSInfo` in nested EXIF
   IFDs and encode rotation via EXIF `Orientation`. Current inspection misses
   nested IFDs and drops orientation, causing thumbnails and face crops to
   disagree with displayed images.
2. `REL-5`: `ImageWorker._index_photos` does not count actual written vectors,
   model calls, vision fallbacks, or stage timings, while `_computational_load`
   fabricates `indexed * 2`. The Jobs UI also silently hides API/config errors,
   degraded fallback states, and loading/empty states.
3. `REL-6`: Repeatedly failing corrupt documents or photos have no per-file
   retry budget or cooldown backoff, allowing early corrupt inputs or repeated
   model failures to spin indefinitely across scan wraps.
4. Phase 1 ingestion benchmark: We need a reproducible, isolated offline harness
   measuring per-stage ingestion latency (`read_hash`, `inspect_exif`,
   `thumbnail`, `vision`, `image_embed`, `text_embed`, `db_commit`), batch/
   concurrency and bounded-prefetch tradeoffs, input resolution scaling, and a
   fault-injection recovery matrix.

This spec implements all four slices for CPU/mock/synthetic execution without
touching private user libraries, downloading models, or inventing GPU metrics.

## Requirements

- EXIF-1 (`REL-4`): `inspect_image` extracts `DateTimeOriginal`/`DateTimeDigitized`
  from the nested Exif IFD and signed latitude/longitude from the nested GPS IFD
  without crashing on malformed tags, and reports orientation-adjusted display
  dimensions. `make_thumbnail`, `InsightFaceDetector.detect`, `face_thumbnail`,
  and `_image_data_url` apply EXIF orientation transposition consistently.
- OBS-1 (`REL-5`): `IndexStats` and `_computational_load` record actual
  persisted image/text/total vector counts, vision/embedding/face calls,
  fallback/degraded counts, and measured stage timings in milliseconds with no
  invented FLOPs or `indexed * 2` estimates. Queue/vision API endpoints return
  structured HTTP 503 errors on invalid `queue_settings.json`. `JobsPage`
  renders loading, empty, API/config error, degraded-mode, and task transition
  states while guarding stale polling and pending mutations.
- RETRY-1 (`REL-6`): Additive schema migration (version 3) adds
  `files.failure_count` and `files.next_retry_at`. Document and photo ingestion
  enforce configurable retry budgets and exponential cooldowns for unchanged
  failing files so early corrupt files cannot starve healthy later files or spin
  indefinitely, while immediately resetting on content-hash change or `force`.
  `recover_expired` supports optional exponential task backoff.
- INGEST-BENCH-1: Provide an isolated, import/help-safe offline benchmark CLI
  (`backend.app.evaluation.ingestion_benchmark`) measuring staged photo
  ingestion baseline (cold vs warm), batch/concurrency sweep, bounded prefetch
  vs synchronous execution, input resolution scaling, and fault-injection
  recovery (corrupt file, vision OOM/unavailable, partial-batch failure,
  post-commit crash idempotency) on synthetic images.

## Ownership and Compatibility

Backend owns EXIF/orientation helpers, additive migration v3, retry/cooldown
bookkeeping, `IndexStats`/`_computational_load`, and the offline ingestion
benchmark CLI. Frontend owns `JobsPage` error/degraded/loading/empty handling
and its zero-browser contract test (`frontend/scripts/check-jobs.mjs`).

All REST routes, response field names, and SQLite tables remain backward
compatible. Migration v3 is strictly additive (`ALTER TABLE files ADD COLUMN`)
and never rebuilds indexes or modifies original source files.

## Acceptance

- Unit/integration tests in `backend/tests/test_phase1_reliability.py` cover:
  - Synthetic JPEGs with nested Exif/GPS IFDs (including negative S/W coords),
    corrupt EXIF safety, and rotated EXIF orientation across `inspect_image`,
    `make_thumbnail`, `face_thumbnail`, and `InsightFaceDetector`.
  - Exact match between persisted `PhotoVector` rows and `IndexStats` /
    `_computational_load` counts in multi-namespace and vision-fallback runs,
    real `stage_timings_ms`, and HTTP 503 reporting on corrupt queue settings.
  - Early corrupt document/photo with `limit=1` advancing to healthy files,
    skipping under cooldown and retry-budget exhaustion, resetting on hash
    change, additive v3 migration idempotency, and `recover_expired` backoff.
  - Ingestion benchmark `--help` safety under `python -S` and full synthetic
    CPU staged baseline + prefetch + resolution + fault-recovery matrix execution.
- `node frontend/scripts/check-jobs.mjs`, `node frontend/scripts/check-search.mjs`,
  and `npm --prefix frontend run build` pass.
- Root `python scripts/check_workspace.py --base origin/main`, `python -m pytest`,
  `ruff check`, and `ruff format --check` pass. Record exact outputs in
  `verification.md`.

## Non-goals

No real GPU/vLLM/Qwen-VL model downloads or hardware claims, no private library
scans or reindexing, no distributed queue or exactly-once guarantees, and no
automatic activation of spec `0007` native indexes.
