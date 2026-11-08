# Verification

Execution date: 2026-11-06. Selected interpreter: repository `.venv/bin/python`
3.11.7. At measurement time, the baseline was `fb23448` and the measurement
additions were local uncommitted changes. On 2026-11-08, the baseline commit's
message was simplified to `51b2052`; its file tree is identical. Product retrieval
behavior and model/index defaults were not modified.
Measurement delivery is complete within the scope below. The offline report has
219 metric rows; static validation verified 25 unique anchors, all filter groups,
CSV/JSON numerical agreement, 21 raw JSON hashes and embedded JavaScript syntax.
All owned benchmark supervisors exited; no benchmark process remains running.

## Completed evidence

- BENCH-1: A first 3 x 300 query product run exhausted its 600-second deadline
  (exit 124). Its resource/log record is retained; it produced no final report.
  A diagnostic confirmed one cached index build followed by warm queries, so
  this is not evidence of rebuilding the index for each query. A second fresh
  fixture runs the same complete 300 queries twice with a 900-second deadline.
  Both complete repeats finished with identical rankings and zero failed queries.
  Actual product Recall@10 was .7712, Hit@10 .7867, nDCG@10 .5990 and MRR
  .5507. First-repeat warm p50/p95 was 830.238/910.425 ms; second-repeat
  828.348/909.991 ms. Cold/revision-rebuild queries took 19.985/19.367 seconds.
- BENCH-2: Real product SQL/scoring/fusion was executed with 100, 1,000 and
  5,000 synthetic photos, two 384-dimensional vectors per file, 20 queries per
  scale. Observed p50 was 73.085 / 761.673 / 4,232.744 ms; p95 was
  76.021 / 828.756 / 4,415.234 ms. Repeat rankings matched. Random vectors
  quantify scan cost, not semantic quality or real photo encoder throughput.
- Synthetic mixed controls returned 10 photos and 0 documents in Top10 for
  10 identical-evidence photos and 10 documents; document-only scope returned
  all 10 documents. A 20-file/219-chunk stress fixture (200 chunks in the first
  file) returned 1 file when requesting 20 with branch window 50. These expose
  existing branch-count and chunk-candidate effects, not natural relevance.
- BENCH-3: 100 copied public abstracts produced 149 chunks. Initial ingestion
  indexed 100 files in 4.276 seconds; unchanged re-scan skipped 100 in 4.098
  seconds; one-changed re-scan indexed 1 and skipped 99 in 4.116 seconds.
  All phases reported zero failures and preserved their input bytes. Rates of
  23.38 / 24.41 / 24.30 count visited files, not newly indexed files.
- BENCH-4: Two full native text runs on SciFact produced identical ranking
  hashes. BM25 Recall@10/nDCG@10 were .7843/.6622; BGE .8362/.7127;
  three-route fusion .8504/.7095. Equal BM25+LSA regressed to .6895/.5058.
  Query inference is fresh; the pre-existing corpus vector cache was reused
  after model/hash/ID alignment and encoder endpoint probes. These are independent
  retrieval-layer results, not the product SQL path or a product rollout.
- Persistent FAISS index overlap with exact Top10 was .99933 for 5,183 eligible
  documents and .99700 for 1,296; reload ranks matched and there were no filter
  violations. Exact/ANN p95 were 11.069/5.786 ms for 5,183. A 52-document domain
  uses exact fallback; these timings exclude encoding and product JSON/SQL.
- CLIP repeated identically: 200 image category accuracy .855, no-answer false
  positives 9/80; promotion remains rejected. SigLIP accuracy .875 and false
  positives 7/80; selection and promotion remain rejected. CLAP's 80-clip
  category accuracy was 1.0 and no-answer false positives 0/32; only its frozen
  local gate passed. Whisper tiny.en had 36 edits/308 reference words across
  12 English clips (WER .116883). None certifies open-world multimodal quality.
- PDF layout's 15 anchor-order pairs improved from 12/15 to 15/15 on one fixed
  page. The unruled table control recovered 24/24 reference slots and 7/7 query
  evidence units. The first numeric-table command incorrectly set one header
  row; the existing frozen control requires two. Preserve this failed harness
  attempt and rerun the original two-header configuration without changing code,
  annotations or expected results. The corrected run passed: 64/64 fields and
  8/8 typed row/answer queries, mean answer bounding-box IoU .93445; source
  unchanged. This is one explicit-region development control.
- BENCH-5: The initial root run completed 367 tests with zero failures/skips.
  Workspace check, configured Ruff check/format scope, frontend simulated
  interaction, Vite build, pip dependency check and sdist/wheel build succeeded.
  vLLM `plan` is configuration output only. The final regression after the
  empty-latency reporting fix also passed all 367 tests with zero failures/skips;
  final workspace/Ruff and the new benchmark modules' import/format checks
  passed. No missing timing is represented as zero.

- Actual in-process API search/chat each completed 30 requests with HTTP 200
  and no contract errors. Warm p50/p95 were 813.286/894.237 ms for search and
  811.134/890.407 ms for chat. Invalid-limit/blank-query responses were 422/400.
- A single warmed CPU profile (official query ID 1395) took .991 seconds with
  instrumentation: cosine similarity .666 seconds cumulative, nested sparse
  CSC-to-CSR conversion .467 seconds, exact-match boosts .225 seconds across
  5,183 calls. Cumulative times overlap; this is a hotspot diagnostic, not a
  full-distribution latency or measured speedup.

## Commands and raw artifacts

All generated state is in ignored `data/project-benchmark-20261106/`.
`execution.json`, `completion-execution.json` and `supplement-execution.json`
retain exact commands, UTC times, deadlines and exits. Per-job `.resources.txt`
contains CPU time and peak process RSS when collected. `native-text-1` has a
fresh result and log but no time-v resource record. `summary.json`/`metrics.csv`
and `report.html` summarize the raw results and source/artifact SHA-256 identities.

The reusable measurement entry point is:

```bash
.venv/bin/python -m backend.app.evaluation.product_benchmark documents \
  --dataset data/hybrid-benchmark/scifact.zip --repeats 2 \
  --work-dir data/new-public-product-run --output data/new-public-product-run.json
```

Use fresh work/output names for every run. Additional modes: `scale`, `ingestion`,
`mixed`, `api`. The `documents`, `ingestion` and `api` modes require `--dataset`;
`mixed` and `scale` use explicit synthetic features. Do not substitute a private database.

Engineering checks used the selected interpreter, `python -m pytest -q`,
`python scripts/check_workspace.py --base HEAD`, configured root `ruff check`,
`ruff format --check scripts tests apps examples services labs learning`,
`node frontend/scripts/check-search.mjs`, `npm --prefix frontend run build`,
`python -m pip check`, and `python -m build --outdir <ignored-output>/dist`.

Public source identities are retained in the reports: BEIR SciFact,
CIFAR-10, ESC-50, LibriSpeech and docling public PDF controls. Sources:
[BEIR datasets](https://github.com/beir-cellar/beir#available-datasets),
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html),
[ESC-50](https://github.com/karolpiczak/ESC-50),
[LibriSpeech](https://www.openslr.org/12),
[docling PDF fixtures](https://github.com/docling-project/docling/tree/main/tests/data/pdf).

## Limits

These are historically exposed, frozen public tests, not unseen holdouts. No
threshold selection or default promotion occurred. Latencies are serial shared
CPU-host observations with varying I/O pressure, not an SLA or controlled
before/after product speedup. Synthetic photo/mixed controls are mechanism
measurements, not human relevance judgments. ASGI requests exclude TCP, browser,
lifespan and concurrency. Chat checks cover local extractive routing, not answer
truthfulness. The full regression uses CPU/mock component contracts; GPU/vLLM
inference, Qwen-VL photo context, cloud/desktop actions, real browser/concurrent
load, face quality, OCR/general PDF, query adaptation, joint relation models and
full video quality were not newly benchmarked. Report rendering receives static
HTML/data/script checks only; no browser visual/interaction acceptance is claimed.
