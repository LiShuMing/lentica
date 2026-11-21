# Review-Driven Reliability Follow-Ups and Operations Runbook

## Problem and Scope

The 2026-10-28 workspace review (baseline `1821a51`) identified ingestion,
retrieval, task-state, queue-settings, and governance gaps. Most scan cursor,
task lease/drain, vector provenance, and pre-top-k filter contracts were
subsequently implemented on `main` in specs `0002-index-task-consistency`
through `0009-product-hybrid-integration`.

This slice ships the remaining review-driven correctness and governance items on
top of `main` (`36508c8`):

1. Fail explicit `index_faces` tasks when the face runtime is unavailable and
   log detector errors explicitly.
2. Validate persisted `queue_settings.json` strictly so corrupt JSON, unknown
   keys, wrong field types, or out-of-range concurrency fail explicitly instead
   of silently re-enabling defaults, and guard `max_concurrency <= 0` early.
3. Require numbered spec + verification updates when executable `.ipynb`
   notebooks change in governed directories.
4. Publish the local operations runbook, historical review report, and updated
   reliability backlog aligned with current `main` behavior, plus regression
   tests covering both the new fixes and earlier review invariants.

It does not restart services, scan private libraries, modify database schemas,
or claim model accuracy.

## Requirements

- STATE-1: Explicit `index_faces` tasks fail when the face runtime is
  unavailable (`face_unavailable > 0`), while retaining run output statistics
  and logging per-file face detector errors.
- QUEUE-1: Non-positive `max_concurrency` starts no workers regardless of GPU
  memory. Settings are written atomically (closed before `os.replace`); invalid
  persisted settings fail explicitly with `ValueError` instead of silently
  enabling default queue processing.
- GOV-2: Executable `.ipynb` notebook changes under governed directories
  require a numbered spec and verification update in `scripts/check_workspace.py`.
- GUIDE-1: Publish accurate local operations guidance, review findings, and
  prioritized reliability follow-ups aligned with `main` (specs `0002`–`0010`).

## Ownership and Compatibility

Backend owns `image_worker.py` and `queue_controller.py`; workspace governance
owns `scripts/check_workspace.py` and documentation. Keep existing APIs,
database schema, component boundaries, and source files unchanged; no new
dependencies are introduced.

Existing valid `queue_settings.json` files remain compatible. Invalid persisted
settings now raise `ValueError` instead of silently reverting to enabled
defaults; the controller loop records the error in `_last_error` without
enabling work. Fix or restore the file while the backend is stopped per
`docs/operations.md`. Rollback is a source checkout change with stopped
services; no data migration or index rebuild is required.

## Acceptance

CPU tests in `backend/tests/test_review_reliability.py` and
`tests/test_workspace.py` cover:

- explicit `index_faces` failure on unavailable face runtime and detector errors;
- `max_concurrency <= 0` guard, malformed/typed/out-of-range `queue_settings.json`
  rejection, and atomic replacement failure preservation;
- notebook spec-change enforcement in `check_change_spec`;
- regression coverage for `ScanCursor` bounded document scans, unavailable
  roots, oversized existing documents, parser errors, pre-limit person/year
  filtering, co-occurrence, and partial task failures.

Run `python scripts/check_workspace.py --base origin/main`, `python -m pytest`,
`ruff check backend scripts tests apps examples services labs learning`, and
`ruff format --check scripts tests apps examples services labs learning`.
Record actual results in `verification.md`.

## Non-goals

No private-library scans, GPU/vLLM/cloud/GUI execution, schema changes, or
duplicate reimplementation of `ScanCursor` / `PhotoSearch` contracts already
governed by specs `0002`–`0009`.
