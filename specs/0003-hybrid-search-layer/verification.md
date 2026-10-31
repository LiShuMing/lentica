# Verification

Status: implemented. Verified on 2026-10-30 against the current local working
tree. No commit/push or deployment was performed. See [evaluation.md](evaluation.md)
for actual public-corpus metrics, per-query cases and limits.

## Changes and acceptance evidence

- CORE-1: dependency-free query/filter/retriever/fusion/store contracts; bounded
  named branches; replaceable fusion and adapters. Core import and actual search
  pass with `python -S`, both from source and an extracted built wheel in a
  temporary directory without site packages or the source checkout on its path.
- FUSE-1: 100 deterministic randomized RRF comparisons against an independent
  rank-table reference pass. Duplicates, ties, one-route-only IDs, candidate
  windows, disabled routes, stable repeated results, and constant-dependent
  ordering are explicitly covered. Native scores do not enter the core.
- FILTER-1: real BM25/exact-vector adapters apply shared metadata/allowed-ID
  filters before windowing. A two-document constructed probe returns target ID 2
  with pre-window filters; the post-fusion filtering control loses it.
- FETCH-1: spy-store tests cover one final batch, skipped fetch for empty results
  and failure on missing records. The full public-corpus matrix observed 3300
  fetch calls, 33000 records, maximum 10 records per call.
- FAIL-1: strict defaults, explicit degraded mode, missing adapter configuration,
  invalid IDs/configuration, missing records and invalid custom fusion outputs
  pass their acceptance tests. Disabled routes are not invoked.
- EVAL-1: two independent CPU runs over all 5183 SciFact documents, all 300 judged
  test queries and 339 positive judgments. Eleven single-route/hybrid/weight/
  constant/window configurations. Full per-query metrics and rankings match
  exactly between runs; ranking SHA-256 is
  `922262879a9c677dd0e71d21dbab6ada9ec63849fbefdc0e17f6e36e9af44da7`.
  Corpus-only 128-dimensional latent semantic encoding uses seed 42. Observed
  Recall@10: BM25 0.7843, vector-only 0.4766, equal hybrid 0.6895. The aggregate
  hybrid regression is recorded, not relabeled as a quality improvement.
- COMPAT-1: the legacy photo fusion facade delegates to the new rank primitive.
  Existing photo/search/task/migration/API regressions remain included in the
  complete test result. No application defaults or existing indexes changed.

## Executed checks

- `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q --tb=short`:
  **125 passed**, 5 inherited PyMuPDF/SWIG deprecation warnings, 8.37 seconds.
  This is an actual full run outside the sandbox, where the previously isolated
  AnyIO wakeup issue does not occur. No tests skipped.
- `.venv/bin/python scripts/check_workspace.py --base HEAD`: passed.
- `ruff check backend scripts tests apps examples services labs learning`: passed.
- `ruff format --check scripts tests apps examples services labs learning`: passed.
- Scoped `ruff format --check` for the new hybrid package, benchmark and tests:
  passed after formatting the README's Python example.
- `.venv/bin/python -m build --no-isolation`: sdist/wheel built successfully.
  Extracted-wheel search smoke test passes without NumPy, SQLAlchemy, Torch or
  application configuration imports. This is an import/search check, not a full
  backend dependency installation. The selected local environment has no pip
  module, so a no-dependency pip-install smoke check was not available.
- `git diff --check`: passed.
- Frontend is unchanged by this slice; no new frontend build/browser check.

## Reproduction and artifacts

Run the explicit local archive through
`python -m backend.app.evaluation.hybrid_benchmark DATASET --output REPORT`.
Use `--seed 42 --dimensions 128`, with BLAS/OMP threads set to 2. No import/help
or benchmark command automatically downloads data or loads credentials.
The separately selected public archive and runtime JSON outputs are ignored under
`data/hybrid-benchmark/`; they are not source artifacts to publish.

Archive SHA-256:
`536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
Runtime: Python 3.11.7, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.2.
No new mandatory dependency was added for this slice.

## Limits and next acceptance boundary

The LSA encoder is corpus-fitted statistical retrieval, not pretrained semantic
model proof. Test-set ablations are descriptive and not a held-out model/weight
selection result. Candidate lists are cached across matrix configurations;
fusion timings must not be presented as end-to-end dual-retriever latency.
Vector retrieval is exact and linear; no ANN, persistent index lifecycle,
distributed execution, deadlines or cancellation orchestration was implemented.

Photo APIs use the shared fusion primitive but keep their existing candidate
materialization. One-final-fetch applies to the new coordinator with a store.
There was no GPU/cloud/pretrained-model run, private source access, user-library
reindex or runtime deployment. The next quality gate is a separately selected
stronger semantic encoder and independent validation queries before changing
default routing/weights. The user's TODO.md was used as input and left unchanged.
