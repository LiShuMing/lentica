# Configuration

Pass `--env-file .env` to Uvicorn or export environment variables before starting
the process. Python module imports read configuration once; restart after changes.
No configuration is read from `~/.env` or from legacy `LLM_*` credentials.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LENTICA_DOCUMENTS_SOURCE` | `./library/documents` | Read-only document root |
| `LENTICA_PHOTOS_SOURCE` | `./library/photos` | Read-only image root |
| `LENTICA_DATA_DIR` | repository `data/` | SQLite, caches, logs, queue settings |
| `LENTICA_QUEUE_ENABLED` | `true` | Initial automatic consumption setting |
| `LENTICA_WORKER_MODE` | `gpu` | Initial queue mode: `gpu` or `smoke` |
| `LENTICA_ENABLE_FACES` | `false` | Enable face runtime and initial face auto-queue |
| `LENTICA_VISION_ENDPOINT` | `http://127.0.0.1:8100` | Loopback vLLM endpoint |
| `LENTICA_VISION_MODEL` | `qwen-vl-photo-context` | Served model name |
| `LENTICA_QWEN_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Transformers fallback path/ID |
| `LENTICA_CLIP_MODEL` | `openai/clip-vit-base-patch32` | Visual encoder path/ID |
| `LENTICA_TEXT_MODEL` | `BAAI/bge-m3` | Text encoder path/ID |
| `LENTICA_WORKER_PYTHON` | backend `sys.executable` | Optional worker interpreter |
| `LENTICA_API_PROXY` | `http://127.0.0.1:8000` | Frontend dev-server backend target |

Relative paths resolve from the process working directory. Keep the repository
root as cwd. The example `.env` uses smoke mode for a CPU-first initial run.

Queue options are persisted in `data/queue_settings.json` and then take precedence
over the initial environment defaults. Change them on Jobs after first startup.
Face runtime also requires `LENTICA_ENABLE_FACES=true` even for manual jobs.

Maximum concurrency defaults to 4. Automatic GPU adjustment can choose fewer
workers; absence of `nvidia-smi` is reported in queue status. Smoke mode can run
without GPU tooling. Hash embeddings only test storage/queue flow.

Changing an encoder requires rebuilding matching namespaces. Do not mix vectors
from different encoders merely because they have the same dimension.
