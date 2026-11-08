# Contributing

Start with a small, reproducible issue or a focused pull request. Work from a
branch, keep existing APIs compatible, and describe behavior plus validation.

Read [AGENTS.md](AGENTS.md), [workspace.toml](workspace.toml), and the component's
scoped rules. Create/update a numbered [spec](specs/README.md) before changing
behavior. Link requirement IDs and record acceptance results in verification.md.
Run `python scripts/check_workspace.py`; behavior-changing PRs also need a spec
update, checked in CI against their base revision.

Read and follow [RULES.md](RULES.md) for every contribution. Both Git author and
committer must use the specified personal Gmail identity. Non-public organizational
material, internal APIs/design mappings and real credentials must never be published;
verifiable public documentation, public APIs and open-source material are allowed.
Run the staged-content, stored-commit and outgoing-history reviews described there.
Keep commit messages to a clear subject and, only when needed, a short body.
Omit enterprise templates and Gerrit `Change-Id`; keep detailed validation in the
relevant spec or review description. Check the stored message after committing.

Install `.[dev]`, run `python -m pytest`, `ruff check backend scripts tests apps examples services labs learning`, and the frontend
build. Add tests for cross-module behavior; avoid tests that only restate constants.
GPU/model tests are optional locally but must be reported explicitly when relevant.
Format the added workspace components with Ruff; CI checks
`ruff format --check scripts tests apps examples services labs learning`.

Use synthetic fixtures. Do not contribute personal files, model weights, index
databases, home-directory paths, credentials, or unredacted task payloads.

Keep large model inference in adapters and workers. Configuration belongs in
`backend/app/core/config.py`; use documented `LENTICA_*` environment variables.
Avoid injecting another project's virtualenv into `sys.path`. Add dependencies to
the appropriate base/optional group in `pyproject.toml`.

Documentation under `docs/roadmap/` and `docs/legacy-design.md` records design
history and planned scope, not a promise that all described features exist.

By contributing, you agree to license your contributions under AGPL-3.0-only.
