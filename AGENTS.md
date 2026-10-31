# Agent Development Rules

Lentica is a spec-driven workspace. Read this file, `workspace.toml`, the applicable
component README/AGENTS.md, and its linked spec before editing behavior.

## Spec Before Implementation

1. For a feature, behavior change, dependency/runtime change, or cross-component
   refactor, create or update `specs/NNNN-kebab-title/README.md` and `spec.toml`.
2. State the problem, scope, requirements with stable IDs, component ownership,
   compatibility/migration, failure behavior, acceptance tests, and non-goals.
3. Use `draft`, `active`, or `implemented` honestly. A draft is not working code.
   Small typo/format-only fixes can cite an existing spec without changing it.
4. Implement a small vertical slice, run its acceptance checks, and record actual
   results in `verification.md`. Never invent GPU, cloud, or accuracy results.
5. A cross-component decision also updates `docs/decisions/` and `workspace.toml`.

## Boundaries

- `backend/` + `frontend/`: the local-first product. Never import cloud or desktop
  execution code from `apps/`, `examples/`, `labs/`, or `learning/` into this path.
- `apps/media-workflow/`: explicit cloud media requests; dry-run by default.
- `examples/multimodal-agent/`: optional GUI-learning example; imports, tests, and
  `--help` must not read credentials, capture a screen, call models, or act on GUI.
- `services/vllm/`: independent CUDA/vLLM runtime. Code/config examples only.
- `labs/gpu/` and `learning/gpu-llm/`: optional experiments, not product dependencies.
- Share source control and specs, not a mandatory all-in-one Python environment.
  Reuse the user's selected interpreter; never inject another venv into sys.path.

## Code and Naming

- Top-level component/spec directories use kebab-case; Python modules/functions
  use snake_case; classes PascalCase; constants UPPER_SNAKE_CASE.
- Keep the existing `backend.app.*` namespace and REST APIs compatible. Prefer
  narrow adapters over broad rewrites. Record intentional breaking changes.
- Use `LENTICA_*` for new configuration. Provider `LLM_*` keys belong only to
  explicitly invoked optional cloud components, never the backend's default path.
- Derive filesystem paths from script locations, selected environment variables,
  or explicit arguments. Never hard-code a developer's home or another checkout.
- Keep original sources read-only. No migrations may move/delete private files or
  rebuild a user's index without a separately authorized operation.
- Do not commit credentials, env files, model weights, virtualenvs, screenshots,
  generated media, task records, notebook outputs, caches, binaries, or logs.

## Python Engineering Standard (Google-derived)

New and refactored Python code must apply the relevant guidance in the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html),
especially imports (2.2), exceptions (2.4), docstrings (3.8), naming (3.16),
function length (3.18), and type annotations (3.19).

- Give each module one cohesive responsibility. Separate reusable library logic
  from CLI parsing, filesystem output, model construction and report formatting.
  Prefer explicit interfaces and composition; do not introduce abstractions
  without a concrete consumer. Review functions around 40 lines for useful
  extraction; this is guidance, not a mechanical line-count limit.
- Use descriptive names and annotate public inputs/returns. Prefer concrete
  domain types; use explicit `Any` only at genuinely dynamic JSON/numeric boundaries.
  Keep optional numeric imports lazy; type-only imports belong under TYPE_CHECKING.
- Write Google-style docstrings for public APIs and non-obvious logic. Describe
  semantics, side effects, Args, Returns and Raises where useful; do not repeat
  types or obvious code. Add comments for rationale and constraints.
- Prefer importing modules to individual implementation symbols. Explicit public
  compatibility re-exports and typing imports are allowed. Keep import groups
  ordered; do not use wildcard imports or create circular dependencies.
- Use built-in exceptions for invalid inputs, preserve exception causes, and
  keep try blocks narrow. Broad catches are allowed only at documented extension
  boundaries that translate or explicitly report arbitrary provider failures.
  Never use production assertions as input validation or silently swallow errors.
- Keep mutable defaults and hidden runtime side effects out of public APIs.
  A refactor preserves public imports, CLI/API contracts, ranking/tie-breaking,
  stored formats and failure policy unless a spec explicitly changes them.
- Apply these rules to the touched module; avoid unrelated repository-wide style
  churn. Match checks to risk and compare deterministic outputs for retrieval
  refactors. Formatting alone does not demonstrate behavior compatibility.

Project-specific adaptations take precedence: retain four-space indentation,
Ruff's configured 120-character line width, component directory naming, and
relative imports within the standard-library-only hybrid core. Do not add or
claim a full Google/Pylint/type-checker gate without configuring and running it.

## Verification and Delivery

Run `python scripts/check_workspace.py`, `python -m pytest`, and
`ruff check backend scripts tests apps examples services labs learning` for
Python/spec changes. Build `frontend/` when its behavior or dependencies change.
Component-specific checks are listed in `workspace.toml` and the relevant spec.
New workspace Python code is formatted with Ruff; run
`ruff format --check scripts tests apps examples services labs learning`.
The change checker requires both spec content/metadata and its verification record
to change together for a behavior-changing PR; a template-only edit is insufficient.
CI defaults to CPU/mock checks. Real GUI/cloud/GPU runs require an explicitly
selected task, input, and environment; migration itself does not authorize them.

Summarize the spec IDs, code changes, checks, and limits. Commit/push when the user
asks or existing task authorization covers it; never infer permission to publish
private artifacts from permission to reorganize source code.
