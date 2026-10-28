# Contributing

Start with a small, reproducible issue or a focused pull request. Work from a
branch, keep existing APIs compatible, and describe behavior plus validation.

Install `.[dev]`, run `python -m pytest`, `ruff check backend`, and the frontend
build. Add tests for cross-module behavior; avoid tests that only restate constants.
GPU/model tests are optional locally but must be reported explicitly when relevant.

Use synthetic fixtures. Do not contribute personal files, model weights, index
databases, home-directory paths, credentials, or unredacted task payloads.

Keep large model inference in adapters and workers. Configuration belongs in
`backend/app/core/config.py`; use documented `LENTICA_*` environment variables.
Avoid injecting another project's virtualenv into `sys.path`. Add dependencies to
the appropriate base/optional group in `pyproject.toml`.

Documentation under `docs/roadmap/` and `docs/legacy-design.md` records design
history and planned scope, not a promise that all described features exist.

By contributing, you agree to license your contributions under AGPL-3.0-only.
