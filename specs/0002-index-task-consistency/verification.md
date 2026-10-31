# Verification

Status: implemented. Verified on 2026-10-30 with synthetic temporary sources and
databases. No original user sources or existing indexes were scanned or rebuilt.

## Environment

Selected interpreter: `.venv/bin/python`, Python 3.11.7, SQLite 3.7.17,
SQLAlchemy 2.0.36, NumPy 1.26.4, scikit-learn 1.5.2, pytest 9.2.0.
CPU validation dependencies were installed into the ignored local environment;
no new product dependency was added.

## Results

- `.venv/bin/python -m pytest -q --tb=short`: **100 passed**, 5 PyMuPDF/SWIG
  deprecation warnings, 7.74 seconds. Full run executed outside the sandbox:
  inside it even an isolated `anyio.to_thread.run_sync` call hung. A bounded
  sandbox run excluding the two asynchronous API tests also passed before the
  final supplementary tests (93 passed, 2 deselected). The complete final result
  includes both API tests; no tests were skipped in the final run.
- `.venv/bin/python scripts/check_workspace.py --base HEAD`: passed,
  213 source files.
- `ruff check backend scripts tests apps examples services labs learning`: passed.
- `ruff format --check scripts tests apps examples services labs learning`: passed,
  64 files already formatted.
- `npm --prefix frontend run build`: passed, TypeScript and Vite production build,
  1619 modules transformed.
- `.venv/bin/python -m build --no-isolation`: sdist and wheel build passed.
- `git diff --check`: passed.

## Acceptance evidence

- SCAN-1: 215 documents resume across sessions; photo probes pass 1000 indexed
  entries; deterministic traversal resumes and excludes child symlinks. Same-path
  image changes replace prior vectors. Unavailable sources do not mark files
  missing and do not prevent another source from progressing.
- TASK-1: active request deduplication, direct database constraint enforcement,
  retry history, delayed eligibility, lease recovery/renewal, stale owner fencing,
  bounded crash retries, zero concurrency and pause/drain checks pass. A real
  photo handler batch with an injected file-read failure retains its successful
  output and reports the task failed with per-file counts.
- VECTOR-1: encoder revision/dimension mismatch and unknown legacy semantic
  vectors are excluded; replacement keeps a single current vector set.
- SEARCH-1: pre-window year/person/location and explicit asset filters, query
  plan branches, vectorless OCR exact matches, deterministic larger candidate
  windows, document cache reuse/invalidation/rollback isolation, and V2 document
  citations without image-model loading pass.
- EVAL-1: one retrieved result among four relevant assets yields Recall@5 = 0.25
  and Hit@5 = 1; rank fusion deduplicates and resolves ties deterministically.
- COMPAT-1: repeated additive migration and complete legacy startup preserve
  existing records and install task constraints after columns are available.
  Existing portable CPU/hash pipeline and API regression tests pass.

## Limits

No GPU inference, real model retrieval-quality/performance measurement, cloud
request, browser interaction test, deployment, or user-library acceptance run.
Frontend verification is a build check only. JSON vectors still require linear
scanning; TF-IDF still builds a full corpus matrix, with at most two cached
revisions per engine. One API/controller owns a data directory; distributed
queue ownership and ANN indexing are not implemented. Unknown legacy semantic
vectors remain stored and need an explicit force-index job for semantic retrieval.
