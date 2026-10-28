# Local vLLM Service

Source and deployment configuration for a separate model runtime, consolidated
from pkl-vllm-service. Spec: [0001](../../specs/0001-agentic-workspace/README.md).
Lentica's API does not import vLLM or share its dependency installation.

```bash
python services/vllm/scripts/service.py plan
```

`plan` reads `configs/qwen-vl.example.toml`, validates loopback/configuration, and
prints the exact argv without importing Torch/vLLM, loading weights, or starting
a service. The launcher consumes this TOML rather than duplicating knobs in shell.

Provision weights and reuse the already validated external service environment:

```bash
export LENTICA_VLLM_PYTHON=/path/to/selected-vllm-env/bin/python
export LENTICA_VLLM_EXECUTABLE=/path/to/selected-vllm-env/bin/vllm
export LENTICA_QWEN_MODEL=/path/to/qwen2.5-vl-3b-instruct
# Optional on hosts with special localhost policy routes:
export LENTICA_VLLM_HOST=127.0.0.2
services/vllm/scripts/check_env.sh
services/vllm/scripts/start_qwen_vl.sh
# In another shell with the same selected config/environment:
services/vllm/scripts/health_check.sh
```

Use `--config /path/to/private.toml` for an instance config. Model-relative paths
resolve from this service directory. `LENTICA_VLLM_PORT` and
`LENTICA_VLLM_LOG_DIR` override port/log location. If a validated environment needs
FFmpeg library search paths, set `LENTICA_VLLM_LIBRARY_PATH` explicitly.

The original environment, model links, and logs are not moved. No weights, virtual
environment, benchmark results, or logs are committed. Historical environment
notes are in `docs/install-notes.md`; their compatibility switches and measurements
are not universal claims about current vLLM versions. Start remains an explicit
foreground operation. `benchmark_qwen_vl.py` sends real requests when invoked;
CI only tests config/argv construction with mocks.
