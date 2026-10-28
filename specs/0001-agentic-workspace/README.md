# Spec-Driven AI Workspace

## Problem and Scope

Five related source trees are unversioned or have no initial commit. Consolidate
their implementation, tests, and learning notes into Lentica with consistent
ownership, naming, and verifiable development rules. The knowledge product stays
at `backend/` + `frontend/`; this is not a rewrite of its retrieval architecture.

## Requirements

- GOV-1: Every component has a README, scoped AGENTS.md, and a manifest entry
  linked to a spec. A PR changing behavior updates a numbered spec and evidence.
- IMPORT-1: Copy source/tests/docs from the five selected roots. Preserve original
  directories and Git state. Exclude environments, caches, binaries, personal
  task records, media, model weights, signed URLs, and executed notebook outputs.
- BOUNDARY-1: The knowledge product remains local-first. Media requests and GUI
  capture/actions stay optional and explicit. Import/help/mock tests have no
  credential loading, network, screen capture, or GUI execution side effects.
- PORTABLE-1: Runtime paths and interpreters use arguments/environment or relative
  discovery. vLLM keeps its own environment, weights, and logs outside Git. A
  service `plan` command validates config and prints argv without loading vLLM.
- VERIFY-1: Root CI covers governance, existing backend tests, media mock tests,
  GUI example contracts, service config/argv, notebook hygiene, and CPU learning
  math checks. GPU/CUDA/Windows checks are reported separately when not run.

## Layout and Naming

| Previous project | Lentica component | Dependency boundary |
| --- | --- | --- |
| bailian-media-workflow | apps/media-workflow | Token Plan CLI; explicit cloud use |
| multimodal-agent | examples/multimodal-agent | Optional GUI and API SDK |
| pkl-vllm-service | services/vllm | Independent CUDA/vLLM environment |
| gpu-lab | labs/gpu | Rust/wgpu, HIP, DirectML, CUDA experiments |
| gpu-llm-roadmap | learning/gpu-llm | Learning dependencies and notebooks |

Directory names are kebab-case. Python entrypoints retain snake_case filenames.
The historical `CHNAGES_LOG.md` and private generation IDs are not imported;
component change history starts with a sanitized `CHANGELOG.md`.

## Design and Compatibility

One source repository, separately activated runtimes. No submodules or nested Git
repositories; a reproducible import script records included paths and hashes.
The original source trees remain available for rollback and existing local
services. API paths and persisted Lentica indexes are unaffected by consolidation.

vLLM uses a TOML config example consumed by its launcher, avoiding stale duplicated
YAML/shell configuration. CLI flags retain a conservative loopback default.
Notebook code remains intact except portable path edits; stored outputs and
machine metadata are removed. Unfinished learning work stays unfinished.

See [ADR 0001](../../docs/decisions/0001-workspace-boundaries.md).

## Acceptance

- GOV-1 / IMPORT-1: workspace checker and its rejection tests; import manifest
  lists source files and excludes disallowed runtime artifacts.
- BOUNDARY-1: media mock suite; GUI import/help and explicit-execution tests;
  backend regression suite without optional cloud/GUI dependencies.
- PORTABLE-1: copied service config generates the expected loopback argv from a
  temporary path, with no vLLM imports; shell syntax checks.
- VERIFY-1: root pytest, ruff, package build, frontend build; NumPy loss/gradient
  tests use synthetic arrays. Notebook output checks run without executing cells.
