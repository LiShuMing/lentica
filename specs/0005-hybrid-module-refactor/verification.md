# Verification

Date: 2026-10-31. Status: implemented and verified in the existing selected
Python environment. No commit, push, deployment, model retuning or default change.

## Policy and module ownership

- STYLE-1: Read the actual Google Python Style Guide sections 2.2, 2.4, 3.8,
  3.16, 3.18 and 3.19 from its public official URL. Root AGENTS.md now records
  required derived practices and explicit local exceptions. Public retrieval and
  evaluation interfaces have annotations and Google-style docstrings. Import
  ordering was checked separately with Ruff I; optional numeric imports stay lazy.
- MODULE-1: Split documents/store, BM25 and exact vectors into documents.py,
  lexical.py and vector.py; adapters.py keeps explicit compatibility exports and
  the cached-rank adapter. Search now composes retrieval, output validation and
  final fetch helpers. Evaluation's hybrid_eval package owns dataset, encoders,
  reporting, execution, workflow and original benchmark logic. CLIs only parse,
  invoke and write; no evaluation library imports either CLI. Route construction
  uses focused builders; the original search space is unchanged.
- API-1: Captured 19 exact pre-refactor source/policy files with SHA-256 in
  `/tmp/lentica-0005-before/manifest.json`, preserving existing dirty work.
  The original-source and refactored modules were run in separate subprocesses
  against the same temporary BEIR-format archive. All eleven benchmark outcomes,
  81 development candidates, five validation finalists, seven evaluation
  comparisons, per-query results, partition identities and encoder IDs match.
  Observed timing values and checksums incorporating those values differ by
  design. Full recursive report keys, list lengths and scalar types also match,
  including timed fields. Legacy facade identities are tested explicitly.
- CORE-1: Real import/search checks pass without site packages, including the
  extracted wheel. New tests cover independent evaluation-library imports and
  chained original retriever/store exception causes. Existing filter, failure,
  one-fetch and randomized rank-fusion tests remain unchanged and pass.

## Actual public-corpus compatibility

Used the existing SciFact archive and existing selection artifact; no selection
or parameter search was rerun on the public corpus for this refactor:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m backend.app.evaluation.hybrid_optimize \
  evaluate data/hybrid-benchmark/scifact.zip \
  --selection data/hybrid-benchmark/scifact-selection-v1.json \
  --output data/hybrid-benchmark/scifact-refactor-0005.json
```

The command exited 0. All seven comparison configurations/metrics, all 300
per-query rankings/metrics, paired bootstrap results, dataset/environment,
selection identity and exposure/latency scope match scifact-optimized-test-1.json
exactly. Rank SHA-256 remains:
`a71015c5c95ef103878374faeacfee60ab95b67c275966b81d2c1a5e24019c5b`.
Existing selection payload checksum remains:
`4685f9137cb273cd65c60153e2fe06b032bf2ef44a38c151371d01a16e26d0c6`.

Actual baseline/selected pipelines also reproduced cached rankings. Each observed
300 final fetch calls, 3000 records, maximum 10 per call. Timings are fresh
observations, not an improvement claim: baseline p50/p95 23.494/40.559 ms;
selected 77.749/103.017 ms. The original quality/statistical limitations from
spec 0004 still apply. No expected outputs were regenerated to conceal a mismatch.

## Executed acceptance checks

- Full `.venv/bin/python -m pytest -q`: **137 passed**, 5 existing PyMuPDF/SWIG
  deprecation warnings, 8.65 seconds. CPU/mock tests only; executed outside the
  sandbox due to the previously isolated AnyIO thread-wakeup limitation. No skip.
- `.venv/bin/python scripts/check_workspace.py --base HEAD`: passed (252 source
  files), including spec ownership and paired change/evidence checks.
- `ruff check backend scripts tests apps examples services labs learning`: passed.
- Additional scoped `ruff check --select I` for hybrid, hybrid_eval, entrypoints
  and the new boundary tests: passed.
- `ruff format --check scripts tests apps examples services labs learning` and
  scoped format checks for all touched/new backend modules/tests: passed.
- `.venv/bin/python -m build --no-isolation --outdir /tmp/lentica-0005-build`:
  sdist and wheel built. The wheel contains the new adapter/evaluation modules.
  Extracted-wheel python -S checks pass for both CLI help commands, independent
  evaluation imports and a real BM25/fusion/final-fetch search. No numeric,
  database, model or application-configuration import occurred in that smoke run.
- `git diff --check`: passed. No frontend behavior changed or frontend build run.

Runtime: Python 3.11.7, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.2; existing
.venv reused, no new dependency installed. Full Google/Pylint certification and
static type checking were not configured/run, and are not claimed. Package smoke
is extracted-wheel evidence, not a fresh dependency-install test.

Local snapshot/fixture comparisons are under `/tmp/lentica-0005-*`; public-corpus
report/stdout are ignored under data/hybrid-benchmark. Previous unrelated source
changes and TODO.md were retained. No private-library reindex, cloud/GPU run,
source publication or user-data migration.

## Comment and code-quality follow-up (COMMENT-1)

Date: 2026-10-31. This section records fresh checks for the comment/quality
follow-up; the public SciFact and package checks above belong to the earlier
module split and were not rerun for this cleanup.

- Explained BM25 saturation, length normalization, repeated query terms and
  global filtered-query corpus statistics; sparse analyzer/feature-budget and
  cosine semantics; dense encoder identity, zero vectors and nonpositive scores.
  Documented pre-window filtering, rank deduplication and deterministic addition,
  partial failures, unchanged custom-fusion exceptions and shallow metadata.
- Clarified corpus-only LSA fitting and dimension caps, grouped query partitions,
  macro metrics, historical MRR@10 naming, cached versus actual pipeline timing,
  order-statistic percentiles and JSON checksum scope. Corrected the bootstrap
  memory claim: the largest draw is 100 * query_count, so memory still grows
  with query count. The interval concerns paired query nDCG, not Recall.
- Added focused local/container annotations and clearer query-frequency naming.
  Replaced repeated example-report relevance scans with one query-ID lookup and
  computed each development-family winner once for both selection and progress.
  No retrieval formula, candidate matrix, JSON format or failure-policy change.
- Captured 19 pre-edit files and hashes at
  `/tmp/lentica-0005-quality-before/manifest.json`. Ran the same existing synthetic
  archive through benchmark, selection and evaluation before/after in separate
  processes with OPENBLAS_NUM_THREADS=2 and OMP_NUM_THREADS=2. Recursively compared
  every report key, scalar type and list length: **2232 deterministic scalar
  values match**, including all rankings, metrics, selected configs, example
  relevance flags, encoder IDs and paired statistics. Excluded 65 observed timing
  values and timing-derived selection-checksum fields from value equality only.
  Reports: `/tmp/lentica-0005-quality-before.json` and
  `/tmp/lentica-0005-quality-after.json`.
- Full local CPU/mock suite: **137 passed**, 5 existing SWIG deprecation warnings,
  8.72 seconds. Used the existing `.venv/bin/python`, outside the sandbox because
  of the previously isolated AnyIO thread-wakeup limitation. Existing tests were
  kept intact; no new test mirroring comments or lookup implementation was added.
- Full configured Ruff check, scoped import-order check, workspace and scoped
  backend format checks, workspace checker with `--base HEAD`, and
  `git diff --check`: passed. No new dependencies, static type checker, package
  rebuild, public-corpus tuning, runtime-service or retrieval-quality claim.

All edits remain local. Existing unrelated changes and TODO.md are preserved.
