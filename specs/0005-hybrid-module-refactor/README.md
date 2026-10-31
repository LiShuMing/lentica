# Hybrid Module Refactor

## Problem and scope

The independent hybrid module and CPU experiments work, but adapters combine
several responsibilities and optimization imports reusable logic from another
CLI module. Important public functions lack annotations/docstrings, and long
orchestration functions mix retrieval, formatting and IO. Apply selected Google
Python engineering conventions to these modules without retuning retrieval.

## Requirements

- STYLE-1: Record the Google Python Style Guide URL and applicable sections in
  root AGENTS.md. Apply focused responsibilities, descriptive names, public type
  annotations, Google-style Args/Returns/Raises docs and narrow exception handling.
  Explicitly retain Ruff's 120-character/four-space style, lazy numeric imports,
  relative hybrid-core imports and compatibility re-exports. No new style dependency.
- MODULE-1: Separate document/store, lexical and dense-vector adapters behind the
  existing adapter facade. Extract shared evaluation data loading, encoding,
  reporting, experiment execution and selection into an importable library.
  Benchmark and optimization entrypoints reuse that library, not each other.
  Separate engine orchestration from branch retrieval, output checks and fetch.
- API-1: Preserve existing public class/function imports, constructor parameters,
  CLI flags, output/report keys, encoder IDs, selection schema/checksum rules,
  fixed search space, float operations, stable ties and strict/partial failures.
  Existing selection artifacts must remain usable; no re-selection or new default.
- CORE-1: Core import/search and CLI --help remain usable with python -S, without
  optional model/numeric/application imports. Preserve pre-window filtering and
  one final batch fetch. No database, frontend or private-library changes.
- COMMENT-1: Document scoring parameters, zero-vector behavior, filter/window
  order, deterministic fusion, shallow metadata snapshots and the exact exception
  boundaries. Explain corpus-only fitting, grouped partitions, checksum limits,
  cached versus actual retrieval timing and paired-bootstrap memory/statistics.
  Replace repeated report-time relevance scans with a query-ID lookup and reuse
  each development family's selected name without changing outputs or selection.
- VERIFY-1: Capture the exact pre-refactor module files locally. Compare old/new
  fixture selection and benchmark deterministic outputs, then run a frozen public
  SciFact evaluation against the existing selection and compare all 300 per-query
  ranks/metrics and seven comparison configurations with the existing report.
  Run focused/full CPU/mock tests, workspace/Ruff/format, package and extracted
  wheel import/help checks. Report actual results and unexecuted checks honestly.

## Ownership, compatibility and failure

Backend owns retrieval/hybrid and evaluation/hybrid_eval. Existing
hybrid.adapters, hybrid_benchmark and hybrid_optimize remain compatibility
facades. No stored-data migration, new dependencies, new environment, automatic
downloads, model changes or output baseline regeneration. Invalid inputs and
provider/store failures retain their existing semantics and chained causes.
Broad exception handling is confined to explicit retriever/store plugin boundaries.

## Acceptance and non-goals

Existing randomized fusion, filtering, provider failure, selection-lock and CLI
tests stay intact. Add only import/dependency and old/new compatibility coverage
needed by the split. Original source snapshot is under /tmp; reports remain in
ignored data storage. This refactor makes no new quality/performance claim and
does not authorize user reindexing, cloud/GPU use, commit, push or deployment.

For COMMENT-1, capture a separate pre-edit snapshot and compare complete fixture
benchmark/selection/evaluation reports, excluding only observed timings and the
selection checksum derived from them. Use the existing tests and required
workspace/style checks; no new test-corpus tuning or public-data rerun is needed
for this documentation and report-lookup cleanup.
