# Specification Workflow

`AGENTS.md` is the agent contract; `workspace.toml` maps components to their specs.
Create a numbered kebab-case directory for each substantial change and copy the
files from `specs/template/`. Choose the next unused number; do not renumber
published specs. Requirement IDs remain stable when implementation details change.

Lifecycle: `draft` -> `active` -> `implemented`. A spec may remain `active` while
shipping a subset; list missing acceptance checks explicitly. Use a new spec when
the product contract changes, not just a new implementation of the same contract.

Every spec contains `README.md`, `spec.toml`, and `verification.md`. The latter
records commands, environment, result, and checks not run. Link a decision record
when changing component boundaries, storage/runtime dependencies, or API contracts.

`python scripts/check_workspace.py` checks structure, manifests, source exclusions,
and output-free notebooks. `--base <commit>` also rejects behavior-changing PRs
without a spec update. This is a mechanical minimum; reviewers must still assess
whether acceptance criteria actually prove the requested behavior.

For behavior changes, update the numbered spec's README/spec.toml and its
verification.md together. An evidence-only or template-only edit does not pass.

## Specs Index

- [0001: Agentic workspace](0001-agentic-workspace/README.md): ownership, import
  boundaries, source hygiene, and paired spec/verification checks.
- [0002: Index and task consistency](0002-index-task-consistency/README.md):
  resumable scan cursors, task leases/draining, encoder provenance, and pre-top-k filters.
- [0003: Hybrid search layer](0003-hybrid-search-layer/README.md): independent
  hybrid retrieval contracts and reproducible evaluation.
- [0004: Hybrid search optimization](0004-hybrid-search-optimization/README.md):
  leakage-aware CPU retrieval optimization and locked evaluation.
- [0005: Hybrid module refactor](0005-hybrid-module-refactor/README.md):
  Google-derived Python conventions and cohesive hybrid modules.
- [0006: Multimodal retrieval quality](0006-multimodal-retrieval-quality/README.md):
  evidence-based retrieval quality and multimodal search.
- [0007: Retrieval execution](0007-retrieval-execution/README.md): versioned
  multimodal query execution and honest evaluation.
- [0008: Video temporal quality](0008-video-temporal-quality/README.md):
  source-grouped public multi-action video retrieval and temporal order.
- [0009: Product hybrid integration](0009-product-hybrid-integration/README.md):
  unified product search over the shared hybrid coordinator.
- [0010: Project benchmark](0010-project-benchmark/README.md): project-wide
  numerical benchmark evidence.
- [0011: Review reliability](0011-review-reliability/README.md): strict queue
  settings validation, explicit face-job unavailability failure, notebook spec
  governance, and operational guidance.
- [0012: Phase 1 reliability and observability](0012-phase1-reliability-observability/README.md):
  nested EXIF/GPS and orientation consistency, actual vector/call/timing
  reporting, UI error/degraded states, retry cooldowns, and offline ingestion
  benchmark harness.
