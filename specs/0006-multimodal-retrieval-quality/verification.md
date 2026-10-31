# Verification

Continuation: spec 0007 now tracks and partially implements M2/M3. Its own
verification/evaluation records cover actual CPU BGE/CLIP/OCR, persistent
adapters, multimodal routing and process deadlines. The M1 evidence below stays
historical; later results do not silently replace or expand its claims.

Date: 2026-10-31. M1 implemented and verified; overall status remains active
because M2/M3 are planned. Design/spec/decision/ownership were written before
implementation. Results and limits are in [evaluation.md](evaluation.md).

## Implemented acceptance

- UNIT-1: Frozen resource-linked units, validated page/bbox/time locators,
  producer/version text evidence, scalar metadata snapshots and unique catalog
  IDs. A UnitStore protocol decouples candidate features from final asset stores.
- STAGE-1: An additive StagedSearch wraps the unchanged HybridSearch. Branch,
  candidate, rerank and final budgets are separate. Per-resource caps, prefix-only
  reranking, complete-permutation validation, stable tails and best-unit grouping
  are implemented. Existing imports and old engine semantics remain compatible.
- EVIDENCE-1: Results retain unit locators and original fusion contributions.
  Candidate-feature unit counts and final asset calls/rows have separate scopes.
  Invalid feature identities and filter leaks fail before final fetching. Missing
  assets chain HydrationError; arbitrary reranker exceptions propagate. The fixed
  identifier baseline ignores generated captions and does not delete candidates.
- DIAG-1: Per-route/union candidate coverage, exclusive relevant pairs, oracle
  candidate upper bound, category metrics and query-level loss counts are present.
  Frozen SciFact baseline/selected rankings match all 300 previous spec-0005
  per-query rankings. Both new runs match all three configurations, diagnostics,
  paired statistics, final-fetch counts and feature counts exactly.
- EXPERIMENT-1: Two generated-media runs each create 216 PNGs and 12 eight-page
  PDFs: 228 resources, 312 units, 48 queries. Pixel histograms and native PDF text
  are actually computed. OCR/captions are explicitly supplied synthetic text.
  All four comparisons' ranks, metrics, locators, counts and generated-file hashes
  match across runs. One-group repeated execution is also covered by tests.
- COMPAT-1: No change to old REST/CLI/selection/default-model/database behavior.
  Existing SciFact archive/selection stay read-only; lock remains
  `4685f9137cb273cd65c60153e2fe06b032bf2ef44a38c151371d01a16e26d0c6`.

## Executed checks

- Full local CPU/mock pytest: **166 passed**, 5 existing SWIG deprecation warnings,
  12.84 seconds; no skipped tests. 29 new cases cover locator/ID/budget boundaries,
  reranker permutations and original causes, feature-store identity, strict filters,
  partial branch failures, final batch fetching, actual pixel colors, page locations,
  loss attribution, frozen-selection parity, repeatability and import/CLI boundaries.
- `ruff check backend scripts tests apps examples services labs learning`: passed.
  Additional E9/F/I checks on all new Python modules/tests passed, as did workspace
  format checks and scoped backend format checks.
- `.venv/bin/python scripts/check_workspace.py --base HEAD`: passed, including
  source hygiene and spec/verification change checks. `git diff --check`: passed.
- `.venv/bin/python -m build --no-isolation --outdir data/hybrid-benchmark/build-0006`:
  sdist and wheel built. New modules are in the wheel; generated data is absent.
  Extracted-wheel `python -S` runs both new CLI help commands and an actual
  BM25 -> identifier rerank -> page evidence -> one final asset-fetch search.
  No NumPy/Pillow/PyMuPDF/model/database/application config import occurs in that
  smoke run. This is extracted-wheel evidence, not a clean dependency install.

Existing `.venv/bin/python` reused: Python 3.11.7, NumPy 1.26.4, SciPy 1.13.1,
scikit-learn 1.5.2, Pillow 10.4.0, PyMuPDF 1.24.14. No dependency installation.
No torch/transformers installation or selected local model directory was available.

The first full public run and media run completed locally. A later sandbox start
failed because the shared overlay `/tmp` was almost full (81 MiB then available;
inodes were not exhausted). Subsequent commands ran outside the sandbox with
TMPDIR pointing at this repository's ignored `data/hybrid-benchmark/tmp-0006`.
Full pytest also needed the previously identified AnyIO sandbox workaround.
No shared cleanup, unrelated process termination or external directory change.

## Artifacts and limits

Ignored `data/hybrid-benchmark/` holds `scifact-quality-0006-run-{1,2}.json`,
`multimodal-0006-run-{1,2}.json`, their fresh media directories and build outputs.
Ignore rules were verified for reports, PNGs and the wheel. Hashes are recorded in
evaluation.md; elapsed-time observations are intentionally excluded from equality.

No product integration/default switch, pretrained inference, real OCR, image-query
API, audio/video extraction, ANN/persistent index, private-library reindex, frontend
change/build, GPU/cloud run, commit, push or deployment. Historical-test exposure
and synthetic controls do not establish generalization. Existing unrelated changes
and the user's TODO.md remain intact. Full Google/Pylint/type-checker certification
is not configured or claimed.
