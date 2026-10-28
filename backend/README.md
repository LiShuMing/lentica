# Lentica Backend

FastAPI, SQLite metadata/queue/vector baseline, document parsing, and local model
adapters. See the [root quick start](../README.md), [configuration](../docs/configuration.md),
and [workspace spec](../specs/0001-agentic-workspace/README.md).

From the root: `python -m pytest backend/tests`. Start with the selected Python:
`python -m uvicorn backend.app.main:app --env-file .env --host 127.0.0.1 --port 8000`.
