# Verification

Date: 2026-10-31. Status: implemented and verified with actual CPU/mock checks
and public SciFact selection plus two locked test evaluations. Observed quality
and latency tradeoffs are in [evaluation.md](evaluation.md); no default switch.

## Executed implementation checks

- `.venv/bin/python scripts/check_workspace.py --base HEAD`: passed (236 files).
  Ownership/spec/evidence checks included.
- `ruff check backend scripts tests apps examples services labs learning`: passed.
- `ruff format --check scripts tests apps examples services labs learning`: passed,
  plus scoped checks of the new adapter/optimizer/tests and modified benchmark.
- Full CPU/mock `.venv/bin/python -m pytest -q`: **133 passed**, 5 inherited
  PyMuPDF/SWIG deprecation warnings, 8.53 seconds. Run outside the sandbox due to
  the already isolated sandbox AnyIO thread-wakeup issue. No skipped tests.
- Eight new focused tests cover sparse word/character retrieval, rare terms,
  pre-window filters, deterministic ties, empty/OOV queries, duplicate-text and
  cross-split isolation, deterministic partitioning, validation fallback, lock
  tampering, changed judgments, paired bootstrap and selection/evaluation on a
  temporary BEIR-format fixture. Actual pipelines reproduce cached ranks and
  fetch final records once per nonempty query.
- `.venv/bin/python -m build --no-isolation --outdir /tmp/lentica-0004-build`:
  sdist/wheel built. An extracted-wheel `python -S` smoke check runs the optimizer
  help and a real hybrid core search without NumPy/sklearn/SQLAlchemy/Torch.
  This is package/import evidence, not a fresh dependency-install test.
- `git diff --check`: passed. No frontend behavior changed in this slice.

Selected interpreter: existing `.venv/bin/python`, Python 3.11.7, NumPy 1.26.4,
SciPy 1.13.1, scikit-learn 1.5.2. No dependencies were added or downloaded.

## Actual data partition

The existing selected SciFact archive has SHA-256
`536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
809 train queries, 300 historical test queries, 5183 shared corpus documents.
Excluded training IDs `1291`, `871` match normalized test query text. The remaining
807 queries form 805 normalized-text groups: 644 development queries/groups,
163 validation queries in 161 groups. Duplicate groups remain together; they
are not collapsed into one label set.

Development partition SHA-256:
`c2c105f4ff74aeb679c1660f25d9a18b3ce1e33a73b663258e34e24aa7a0e12e`.
Validation partition SHA-256:
`6abd0fbef1e28a713070a8003f94c67cf61dde414e8b3d5fa8fd58dbe0d40696`.

## Actual selection and frozen evaluation

- SPARSE-1: real corpus-fitted word/char sparse adapters, no SVD projection or
  pretrained runtime. The character matrix has 60000 features and 8278771 nonzeros.
  Shared filters precede limits; zero scores are excluded. Numeric imports stay lazy.
- SPLIT-1: the above train/development/validation partition and overlap exclusions
  were actually checked. Corpus encoding only uses documents, not queries/labels.
- SELECT-1: 81 development configurations and five validation finalists completed.
  Selected BM25 + char_wb(3,5), equal weights, windows 50, c=1, top-k=10; BM25
  k1=1.2/b=.75. Validation nDCG .696522 -> .721375, Recall .803067 -> .857464.
  Original BM25 remains the best development lexical configuration; word/LSA
  finalists fail the predefined validation improvement gate.
- LOCK-1: `scifact-selection-v1.json` was written before separate test commands.
  Canonical payload lock SHA-256:
  `4685f9137cb273cd65c60153e2fe06b032bf2ef44a38c151371d01a16e26d0c6`.
  Selection environment/corpus/partition/config checks passed in both runs.
- REPORT-1: both test commands exited 0. All seven comparisons' 300 per-query
  rankings, metrics and bootstrap results match exactly. Ranking SHA-256:
  `a71015c5c95ef103878374faeacfee60ab95b67c275966b81d2c1a5e24019c5b`.
  Selected Recall .836944 vs baseline .784278; nDCG .676097 vs .662213;
  MRR .629487 vs .630243. Paired nDCG mean difference .013884, 95% bootstrap
  interval [-.003667, .032295], 51 improved/53 regressed/196 tied queries.
  The interval includes zero; no stable generalization claim. Baseline/LSA/equal
  rankings also match all 300 historical spec-0003 controls exactly.
- COMPAT-1: each run additionally executed real baseline/selected pipelines;
  cached top-k rankings matched every query. Each pipeline observed 300 final
  fetches, 3000 records, maximum 10 per fetch. First selected pipeline p50/p95
  76.950/101.842 ms vs baseline 23.164/40.626 ms; second 77.005/101.144 vs
  23.273/40.179 ms. This quality optimization increases local compute cost.

Ignored local artifacts under `data/hybrid-benchmark/`:
`scifact-selection-v1.json`, `scifact-optimized-test-1.json`,
`scifact-optimized-test-2.json`, and their stdout files. Git ignore rules were
checked. The selection artifact is about 61 KiB and test reports about 1.2 MiB
each. Artifacts are not committed/published. Selected runtime/data only; no model
download or hidden test-time parameter search occurred.

## Limits

No pretrained/GPU/cloud run, private library reindex, persistent/ANN index,
frontend/browser check, product default change, commit, push or deployment.
CPU sparse retrieval does not prove pretrained semantic quality. Historical test
inspection is disclosed; bootstrap intervals do not remove that exposure.
User's original TODO.md and previous unrelated tracked/untracked changes retained.
