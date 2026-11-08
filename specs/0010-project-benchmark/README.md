# Project benchmark evidence

Status: implemented (measurement tooling and scoped evidence, not whole-product acceptance). Backend owns offline product measurement and report inputs;
frontend and optional components retain their existing acceptance contracts.

## Problem and scope

The product coordinator is integrated, but isolated algorithm experiments and
mock regressions do not quantify the actual product document route or its scale.
Provide reproducible numerical evidence without touching a private library,
changing retrieval defaults, tuning on test labels, or enabling cloud/GUI work.

- BENCH-1: Run the unchanged product document search on all 5,183 BEIR SciFact
  abstracts and all 300 official judged test queries. Materialize an isolated
  SQLite fixture with opaque paths and one abstract per chunk, record source and
  query identities, cold/revision rebuild costs, warm latency distributions,
  resource-level Recall/Hit/nDCG/MRR and repeat ranking identities. This measures
  the product TF-IDF route, not source parsing or generative answer accuracy.
- BENCH-2: Record bounded synthetic photo-vector scale controls and mixed
  photo/document ranking controls separately from public model-quality evidence.
  Exercise real product scoring/fusion/final SQL reads with explicit deterministic
  encoders. Do not call hash/random vectors semantic models. Report dataset sizes,
  dimensions, candidate counts, scanned vectors and actual elapsed time.
  Add actual in-process ASGI V2/chat measurements over the same public fixture;
  keep HTTP routing/serialization separate from TCP/browser or concurrent load.
- BENCH-3: Measure actual document parsing/indexing on a bounded public-text
  sample, unchanged re-scan and changed-file replacement. Verify counts and
  source byte preservation. Isolate all generated sources, state and outputs.
- BENCH-4: Re-execute selected existing frozen text/index/image/audio/PDF
  benchmarks with explicit local public inputs/models. Distinguish reused corpus
  features from fresh inference, historical exposed tests from unseen holdouts,
  independent layers from product activation, failures from missing coverage.
- BENCH-5: Deliver machine-readable fresh reports, command/exit/environment
  records and an offline numerical overview. Include full CPU/mock regression,
  frontend checks/build and static checks. No aggregate product PASS from optional
  model controls. Record CPU/shared-host latency as observations, not SLA evidence.

## Compatibility, failures and non-goals

Existing APIs, defaults, queues, model identities and stored user data remain
unchanged. CLI requires explicit source and fresh output/work paths, performs no
downloads, and never uses the user's database. Missing records, invalid outputs
and failed checks remain visible; never exclude failed queries from denominators.
No new GPU, cloud, desktop, private-library or running-user-service operation.
No parameter selection or model promotion is authorized by benchmark execution.
Committing or publishing results requires a separate explicit user request.

## Acceptance

Validate metric/ID mapping and isolation on small fixtures, then execute the full
fixed corpus and scale matrix. Retain repeat hashes and all individual queries.
Run root checks and record actual commands/results in verification.md. Generated
databases, logs and numerical reports live only in the selected ignored data
directory; source and evidence protocol remain reviewable.
