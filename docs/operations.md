# Local Operations Runbook

## Before Starting

Read [configuration](configuration.md) and use the repository root as cwd.
Reuse a selected Python environment; vLLM has a separate compatible runtime.
Choose a fresh `LENTICA_DATA_DIR` and a small synthetic source library first.
Create the configured document/photo directories before starting the queue.

Run exactly one backend/controller per data directory: no `--workers N`, no
second checkout using that database, and no simultaneous standalone consumer.
The API is unauthenticated and intended only for a trusted local user. Bind the
API, frontend, and vision server to loopback; do not publish them on a LAN.

```bash
mkdir -p library/documents library/photos
python -m pytest -q --disable-warnings
python scripts/check_workspace.py
python -m uvicorn backend.app.main:app --env-file .env --host 127.0.0.1 --port 8000
```

In another terminal, use `npm --prefix frontend run dev -- --host 127.0.0.1`.
Check `/health`, then Settings (source paths), Jobs (queue state), and Models.
`/health` is process liveness, not proof that models or indexing work.

## Queue Controls and Known Limits

The controller checks settings every five seconds. Jobs polls tasks and queue
status every 2.5 seconds. Tasks progress through queued -> running -> succeeded
or failed; queued tasks can be cancelled. Retry creates a new task ID and retains
the original run. Batch failures now remain failed even when some files succeed.

Environment variables seed settings on first use. Afterward the persisted
`queue_settings.json` in the selected data directory wins. Maximum concurrency
is a ceiling, not a promise; automatic GPU adjustment can select fewer workers.
Zero concurrency prevents consumers, but enabled auto-enqueue may still create
queued tasks. Disable Auto queue as well when no discovery/enqueue is wanted.

**Pausing or reducing concurrency drains active workers via `SIGUSR1`** so excess
workers finish their in-flight task before exiting without claiming new work.
During shutdown the controller sends `SIGUSR1` and waits for a bounded grace
period before killing stuck processes. Tasks use renewable leases; if a worker
exits abnormally, expired ownership is recovered automatically by the controller
and on backend startup. For routine maintenance on long queues, disable the
per-source auto-enqueue switches first and let active jobs complete.

Document `max_files` bounds the number of files visited per scan call and
persists a per-source scan cursor so repeated bounded scans advance across the
full library and wrap only after finishing a traversal. Full scans do not mark
unavailable sources or size-excluded existing files missing. Repeatedly failing
early files still need bounded retries/backoff (REL-6).

Face indexing needs both `LENTICA_ENABLE_FACES=true` and a provisioned runtime.
No face is a valid completed result; unavailable/failed face processing is not.
Named-person queries use confirmed names/aliases, not identities guessed by VLM.
Multiple recognized people mean co-occurrence, not an automatic OR fallback.

## Vision and Encoder Checks

Use [model serving](model-serving.md) for loopback service configuration.
`python services/vllm/scripts/service.py plan` prints a launch plan without loading
weights; `health` checks `/v1/models` on the selected configured endpoint.
Check that the served model name matches the queue settings. Do not change WSL
network policy just to fix an endpoint; use a tested loopback address consistently.

Test semantic indexing on one explicitly selected synthetic image before a large
build. Smoke/hash mode tests orchestration only. A successful basic fallback is
not evidence of successful Qwen context extraction. Inspect the run's model profile
and worker logs, and avoid comparing quality across silently different encoders.

Changing an encoder/path/revision or pooling requires a separately authorized
reindex into isolated state. Stored vectors record `encoder_id` provenance so
incompatible spaces are excluded from semantic ranking rather than silently
mixed. Do not overwrite a working index or mix smoke and semantic namespaces to
improve apparent recall. Confidence values are ranking heuristics, not
probabilities.

## Backup, Recovery, and Rollback

1. Wait for active work to finish and stop the backend plus its consumers. Stop the
   separate vision service only when GPU/runtime maintenance is needed.
2. Back up the entire selected data directory while stopped: SQLite database,
   possible `-wal`/`-shm` sidecars, settings, caches, and logs. Keep source originals
   and model caches separately. Never copy only a live SQLite file as a backup.
3. Record the source commit, Python environment, model identities, and endpoint
   settings with the private backup; never commit those runtime artifacts.
4. Validate a restored copy using an isolated data directory and synthetic roots.
   Stop both instances before promoting a restored directory. Keep one owner.
5. Roll back code only with stopped services and a compatible state backup;
   verify database schema compatibility with [migration](migration.md) before
   reverting across schema-changing releases.

For stale running tasks after an abnormal worker exit, allow the controller's
lease-expiry recovery to requeue the task (or restart the single backend after
verifying no old consumer remains). Recovery closes the abandoned run and
requeues the task. Do not directly edit running rows or retry while another
process may still own them. Inspect failed jobs and their retained counters
before retrying; a systemic model/config error will not heal by retry.

Settings are replaced atomically. Invalid JSON, unknown keys, wrong field types,
or out-of-range concurrency cause an explicit error, not enabled defaults. Stop
the backend, preserve the bad file for diagnosis, and restore/correct it with
`enabled: false` and `max_concurrency: 0` until verified. Do not delete it while
running: a missing file is initialized from environment defaults, which can enable
work. Invalid settings do not forcibly interrupt already-running consumers.

## Evidence and Privacy

Task output includes counts, start/end times, and duration, not measured FLOPs.
Some computational-load/vector counts are estimates pending REL-5. Logs can
contain private paths and payloads; redact them before opening public issues.
Use the [review report](reviews/2026-10-28.md) for current limitations and the
[numbered specs](../specs/README.md) for changing behavior. Real retrieval quality,
GPU memory, latency percentiles, and cloud actions require separate explicit runs.
