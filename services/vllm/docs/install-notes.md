# Installation Notes

> Historical workstation measurements from the original service. No environment
> was recreated or validated by the workspace import. Follow the service README
> and the TOML launcher for current path/configuration conventions.

Date: 2026-07-12

## Official References Checked

- Qwen vLLM deployment docs recommend installing vLLM in a clean environment with `pip install "vllm>=0.8.5"` and note that vLLM wheels have strict Torch/CUDA compatibility.
- vLLM multimodal docs show OpenAI-compatible chat requests using `image_url` content parts and document `--allowed-local-media-path` for local file URLs. This phase keeps the service as direct OpenAI-compatible vLLM and does not add a wrapper.
- vLLM 0.25 CLI help was checked locally. `--limit-mm-per-prompt` now accepts JSON or JSON-key syntax, so the startup script uses `--limit-mm-per-prompt '{"image":1,"video":0}'`.

## Local Environment

- Python: `/usr/bin/python3.12` in `~/work/lentica/services/vllm/.venv`
- vLLM: `0.25.0`
- Torch: `2.11.0+cu130`
- Torch CUDA runtime: `13.0`
- Transformers: `5.13.1`
- Torchvision: `0.26.0+cu130`
- GPU: NVIDIA GeForce RTX 5070 Ti
- GPU capability: `(12, 0)`
- Driver from `nvidia-smi`: `610.43.02`
- CUDA UMD from `nvidia-smi`: `13.3`
- GPU memory: `16303 MiB`

The pyenv Python at `/path/to/python` was not used because it lacks the standard `_bz2` module, which breaks vLLM/torchvision imports. The project `.venv` was rebuilt with `/usr/bin/python3.12`.

## Installed In This Directory

```bash
cd ~/work/lentica/services/vllm
/usr/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "vllm>=0.8.5" av
```

No dependency was installed into `~/.venv` or `personal-knowledge-lab`.

## Runtime Notes

vLLM 0.25 imports TorchCodec from its multimodal stack. This machine does not expose system FFmpeg libraries, so `scripts/start_qwen_vl.sh` sets `LD_LIBRARY_PATH` to FFmpeg libraries installed inside `.venv`:

- `.venv/lib/python3.12/site-packages/PyNvVideoCodec`
- `.venv/lib/python3.12/site-packages/av.libs`

Two vLLM compatibility switches are required on this RTX 5070 Ti / CUDA 13 setup:

- `VLLM_USE_V2_MODEL_RUNNER=0`: avoids `RuntimeError: UVA is not available` during V2 model runner initialization.
- `VLLM_USE_FLASHINFER_SAMPLER=0`: avoids FlashInfer sampler SM 12.x capability detection failure during profiling.

The service is configured for image-only Qwen-VL requests:

- host: `127.0.0.2`
- port: `8100`
- served model name: `qwen-vl-photo-context`
- max model length: `4096`
- max sequences: `8`
- max batched tokens: `4096`
- GPU memory utilization: `0.82`
- dtype: `bfloat16`
- multimodal limit: `{"image":1,"video":0}`
- image processor maximum: `2007040` pixels

## Validation Commands

```bash
scripts/check_env.sh
scripts/start_qwen_vl.sh
scripts/health_check.sh
curl --noproxy '*' http://127.0.0.2:8100/v1/models
```

The service runs in the foreground. Stop it with `Ctrl-C`.

## Localhost Routing Note

This host has WSL policy routing for TCP/UDP:

```text
from all ipproto tcp lookup 127
127.0.0.1 via 169.254.73.152 dev loopback0 table 127
```

As a result, a process can appear to listen on `127.0.0.1:8100`, while client connections to `127.0.0.1:8100` are routed through `loopback0` and return `Connection refused`. A plain Python HTTP server reproduced the same behavior.

`127.0.0.2` routes to the local `lo` table and was verified with a plain Python HTTP server. The vLLM service therefore defaults to `127.0.0.2:8100` on this machine. It remains loopback-only.

## Validation Result

Completed:

- `.venv` import checks passed.
- Torch sees CUDA and the RTX 5070 Ti.
- vLLM CLI starts with `.venv` FFmpeg library paths.
- Qwen2.5-VL architecture is recognized as `Qwen2_5_VLForConditionalGeneration`.
- Model weights load from `models/qwen2.5-vl-3b-instruct`.
- With `VLLM_USE_V2_MODEL_RUNNER=0` and `VLLM_USE_FLASHINFER_SAMPLER=0`, vLLM reaches API startup. Use `127.0.0.2:8100` for access on this host.
- `scripts/health_check.sh` passed on `127.0.0.2:8100`.
- Short validation runs were stopped with `timeout`/`Ctrl-C`; no long-running vLLM or GPU process was left active.

Remaining blocker:

- `127.0.0.1:8100` is not usable on this WSL host because of the policy route above. Use `127.0.0.2:8100` unless the host routing is changed.

Observed non-fatal warnings:

- `Failed to get device capability: SM 12.x requires CUDA >= 12.9` appears even though Torch reports CUDA 13.0 and the process continues.
- vLLM uses the model `generation_config.json`, including `temperature: 1e-06`; pass request-level sampling parameters from the client if different behavior is needed.
