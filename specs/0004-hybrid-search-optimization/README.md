# Hybrid Search Optimization

## Problem, scope and ownership

Spec 0003 measured weak corpus-fitted LSA vectors and degraded equal-weight
fusion on SciFact. Improve the local CPU retrieval adapters and add a separate
selection/evaluation workflow under the backend. This is Lentica's own design.
No quality improvement is assumed and no product default changes automatically.

## Requirements

- SPARSE-1: Add an optional sparse TF-IDF cosine retriever supporting word ngrams
  and character-within-word ngrams. Fit corpus only, allow singleton terms within
  the vocabulary budget, apply
  shared filters before top-k, omit zero matches, stable integer-ID ties. Lazy
  numeric imports preserve standard-library-only hybrid core/import/help.
- SPLIT-1: Load explicit BEIR train/test splits; preserve the existing test default.
  Remove training queries whose normalized text occurs in test. Group remaining
  duplicate texts before deterministic seed-42 80/20 development/validation split.
  Log excluded IDs, group/query counts and hashes. Never fit on query texts/labels.
- SELECT-1: Freeze the search space before evaluation: BM25 k1={0.6,1.2,2.0},
  b={0.25,0.5,0.75}; word TF-IDF (1,2), char_wb TF-IDF (3,5), at most 60000
  features each, min_df=1, sublinear_tf=True; existing 128-dimensional LSA.
  Pick one BM25 on development nDCG@10, then evaluate each secondary branch
  fused with that BM25 using secondary weights={0.1,0.25,0.5,1}, c={1,10,60},
  windows={50,200}. Keep at most one finalist per family (lexical, word, char,
  LSA), plus original BM25. Selection on validation maximizes nDCG@10, then
  Recall@10, then MRR@10, then stable configuration name. Accept an optimization
  only when validation nDCG improves by at least 0.005 and Recall drops no more
  than 0.01 against original BM25; otherwise fall back to original BM25.
- LOCK-1: Write a corpus/archive/partition/config-bound selection artifact before
  test execution. A separate evaluate command verifies it and runs only frozen
  comparisons; test metrics cannot alter configuration. Artifact hashes detect
  accidental edits, not malicious tampering. No automatic data/model download.
- REPORT-1: Include every development candidate, validation finalist and fixed test
  comparison, per-query test rankings, metrics, paired nDCG bootstrap interval (10000
  samples, seed 42), improved/regressed/tied counts, reproducibility hashes and
  observed retrieval/build timings. Cached fusion time is not end-to-end latency.
  Prior test inspection means this is a historical test comparison, not a fresh
  unseen holdout. Character/word sparse retrieval is not pretrained semantics.
- COMPAT-1: Preserve core protocols, old benchmark CLI, REST APIs and existing
  application defaults. No new dependencies, DB migrations or private reindex.

## Failure behavior and migration

Invalid/empty partitions, unsupported splits, corpus mismatch, edited selection
artifacts or malformed configurations fail explicitly. No silent partial tuning.
Numeric imports occur only when explicitly constructing an adapter/running an
experiment. Selection is opt-in and local. There is no stored-index migration.

## Acceptance

Test sparse retrieval/filtering/zero queries/ties, duplicate and cross-split leakage
protection, deterministic selection and validation fallback, lock validation and
separate fixture select/evaluate flow. Run actual SciFact selection and two frozen
test evaluations. Record quality regressions as well as gains. Run workspace,
full CPU/mock pytest and Ruff checks. No frontend behavior changes.

## Non-goals

No pretrained-model claim, GPU/cloud execution, ANN or distributed indexing,
universal statistical significance claim, product default switch or publication.
