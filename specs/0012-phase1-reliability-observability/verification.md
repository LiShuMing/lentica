# Verification

Date: 2026-11-21. Status: implemented; CPU/mock/synthetic acceptance passed.
Baseline: `codex/project-review` (`bf565e9` on top of `origin/main` `36508c8`).

Reused the available Python 3.12.3 environment, Ruff 0.15.7, and Node 24.21.0.
No new environment or dependency was installed.

| Requirement | Command / evidence | Result |
| --- | --- | --- |
| EXIF-1 (`REL-4`) | `backend/tests/test_phase1_reliability.py`: nested Exif/GPS IFDs (signed S/W coordinates), corrupt JPEG/zero-denominator safety, rotated orientation=6 thumbnail, `InsightFaceDetector`, and `/api/people/faces/{id}/thumb` crop alignment | Passed |
| OBS-1 (`REL-5`) | `backend/tests/test_phase1_reliability.py`: persisted `PhotoVector` count matches `IndexStats` and `_computational_load` (no `indexed * 2`), fallback/degraded flags, `stage_timings_ms`, HTTP 503 on corrupt `queue_settings.json` | Passed |
| OBS-1 (UI) | `node frontend/scripts/check-jobs.mjs`, `node frontend/scripts/check-search.mjs`, `npm --prefix frontend run build` (`1619 modules`, `1.37s`) | Passed (mock hooks/HTTP; no browser) |
| RETRY-1 (`REL-6`) | `backend/tests/test_phase1_reliability.py`: `limit=1` fairness past corrupt files, document/photo cooldown & retry-budget exhaustion, hash-change reset, additive migration v3 idempotency, `recover_expired` exponential backoff | Passed |
| INGEST-BENCH-1 | `python -S -m backend.app.evaluation.ingestion_benchmark --help` and `python -m backend.app.evaluation.ingestion_benchmark --count 12` | Passed (synthetic CPU/hash baseline: cold `0.5407s` / `22.195 img/s`, `84/84` vectors persisted, warm skip `0.0362s`, `256x256` vs `512x512` thumbnail `6.668ms` -> `34.349ms`, simulated OOM fault + cooldown + recovery `3` files / `21` vectors) |
| All | `python scripts/check_workspace.py --base origin/main` | Passed (`439 source files`) |
| All | `ruff check backend scripts tests apps examples services labs learning` | Passed |
| All | `ruff format --check scripts tests apps examples services labs learning backend/tests/test_phase1_reliability.py backend/app/evaluation/ingestion_benchmark.py backend/app/ingestion/images.py` | Passed (`33 files already formatted`) |
| All | `git diff --check` | Passed |
| All | `python -m pytest -q --disable-warnings` | 380 passed, 14 skipped |

Limits: All measurements above are CPU/mock/synthetic runs. Real GPU/vLLM
Qwen-VL/CLIP/BGE throughput curves, GPU VRAM peak sweeps, real browser E2E,
and private-library indexing remain pending an explicitly selected GPU/service
environment.
