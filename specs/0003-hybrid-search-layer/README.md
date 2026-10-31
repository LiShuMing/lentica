# Independent Hybrid Search Layer

## Problem and scope

Photo retrieval mixes domain assembly and fusion; there is no reusable coordinator
for independent retrievers, bounded candidate windows, shared filters, final record
fetching, or controlled retrieval experiments. Introduce a small local Python
library under `backend/app/retrieval/hybrid/`, based on Lentica's own contracts.
The package imports no database, application settings, model runtime or cloud SDK.
The existing backend owns the component; a separate distribution is not required.

## Requirements

- CORE-1: Typed query, immutable filters, retriever, fusion and document-store
  interfaces. Any number of named branches with separate windows and weights;
  validate finite parameters, bounded work and unique branch names.
- FUSE-1: Weighted reciprocal rank fusion (RRF) accepts ordered IDs, never native
  scores. Deduplicate each list before assigning one-based ranks; preserve single
  branch hits; stable ID tie-breaking; independent windows precede fusion.
- FILTER-1: Every retriever applies the same structured exact-match/ID filters
  before its candidate window. Built-in BM25 and vector adapters implement this;
  unsupported filters fail rather than silently becoming post-fusion filters.
- FETCH-1: Retrieve IDs only, fuse, then fetch final records once in one batch.
  Missing records fail explicitly. Ranking-only callers can omit a store.
- FAIL-1: Branch exceptions fail by default. Explicit partial mode returns a
  degraded status and branch errors. Zero-weight/window branches are not invoked;
  never suppress a failed required fetch or fabricate a complete response.
- EVAL-1: Reproducible CPU evaluation of lexical-only, vector-only and hybrid,
  weight/constant/window ablations with Recall@10, nDCG@10 and MRR. Freeze corpus,
  query judgments, seed and versions; report per-query examples, filtered-search
  counterexamples and actual timings. Distinguish fixture correctness, observed
  corpus retrieval quality and unmeasured pretrained-model quality.
- COMPAT-1: Keep photo API shapes and existing fusion imports. Route legacy
  fusion through the independent core, with focused regression tests; no index
  migration, default model switch, private-library access or GPU execution.

## Ownership and interfaces

`hybrid/contracts.py`: SearchQuery, SearchFilter, Branch, SearchConfig, Retriever,
DocumentStore, FusionStrategy and result records.
`hybrid/fusion.py`: bounded rank-only RRF.
`hybrid/engine.py`: branch orchestration, diagnostics and final fetch.
`hybrid/adapters.py`: local BM25, exact dense-vector and precomputed-rank adapters.
`backend/app/evaluation/hybrid_benchmark.py`: public-corpus/fixture evaluation;
optional corpus-fitted latent semantic encoding uses existing scikit-learn.

Conceptual signatures:
`Retriever.retrieve(query: SearchQuery, limit: int) -> Sequence[int]`;
`FusionStrategy.fuse(rankings, branches, constant, top_k) -> Sequence[FusedHit]`;
`DocumentStore.fetch(ids: Sequence[int]) -> Mapping[int, object]`;
`HybridSearch.search(query, config) -> SearchResult`.

## Compatibility, failure and migration

The core has only standard-library dependencies; concrete adapters can be replaced
without changing fusion or record fetching. Existing `weighted_rrf` remains a
compatibility facade. No stored vectors, source files or task records change.
Adapter contract violations and invalid configuration raise actionable exceptions.
Execution is synchronous and deterministic; deadlines/cancellation, remote ANN,
distributed statistics and parallel scheduling are future adapter concerns.

## Acceptance

Independent reference RRF comparisons over randomized lists; duplicates, ties,
single-route hits, disabled routes, parameter bounds, shared pre-window filters,
explicit branch failures and exactly one final fetch. Backend regression suite,
workspace checks, Ruff, package build and independent import/package smoke checks.
Run the evaluation matrix on a public labeled corpus if download is available;
otherwise run a frozen local labeled corpus and report the missing external run.
No quality gain is assumed: publish measured regressions as well as gains.

## Non-goals

No upstream proprietary design reproduction, mandatory index server, synthetic
vectors presented as model-quality evidence, automatic model/data downloads on
import/help, API redesign, user-library reindex, GPU/cloud execution or deployment.
