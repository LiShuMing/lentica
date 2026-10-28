# Local Model Serving

Install vLLM in a separate environment appropriate to your GPU/CUDA stack. The
application communicates through an OpenAI-compatible multimodal localhost API.
Do not install a second vLLM stack into the application's lightweight environment.

Example after provisioning Qwen2.5-VL weights:

```bash
vllm serve /path/to/Qwen2.5-VL-3B-Instruct \
  --host 127.0.0.1 --port 8100 \
  --served-model-name qwen-vl-photo-context \
  --max-model-len 4096 --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 \
  --limit-mm-per-prompt '{"image":1,"video":0}'
```

CLI options vary by vLLM release; inspect your installed version's help. One
process normally serves one base model. Benchmark distinct photos before raising
concurrency, image resolution, or GPU memory allocation. Client requests cap
image resolution and send data URLs; no server-side arbitrary media path is needed.

```bash
curl --noproxy '*' http://127.0.0.1:8100/v1/models
```

Configure the matching loopback endpoint and model name in Lentica. On WSL hosts
with unusual localhost routing, another loopback address such as `127.0.0.2` may
be needed; diagnose routing independently before changing inference settings.

CLIP/BGE embeddings run in application workers. The vLLM queue mode currently
defaults their device to CPU to reserve GPU memory for Qwen. This can limit total
throughput; do not infer application concurrency from vLLM's sequence count.

Missing vision service can degrade to basic metadata context. Hash/basic output
is a development fallback, not equivalent to semantic image understanding.
