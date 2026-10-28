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
