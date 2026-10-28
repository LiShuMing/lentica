#!/usr/bin/env python3
"""Short, repeatable OpenAI-compatible Qwen2.5-VL benchmark."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import statistics
import time
import urllib.request
from io import BytesIO
from pathlib import Path


METRICS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
)
URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_json(url: str) -> dict:
    with URL_OPENER.open(url, timeout=30) as response:
        return json.loads(response.read())


def fetch_metrics(base_url: str) -> dict[str, float]:
    with URL_OPENER.open(f"{base_url}/metrics", timeout=30) as response:
        lines = response.read().decode("utf-8").splitlines()

    values = dict.fromkeys(METRICS, 0.0)
    for line in lines:
        for metric in METRICS:
            if line.startswith(f"{metric}{{") or line.startswith(f"{metric} "):
                values[metric] += float(line.rsplit(" ", 1)[1])
    return values


def synthetic_image_data_url(size: int) -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    third = size // 3
    draw.rectangle((0, 0, third, size), fill="#d84a3a")
    draw.rectangle((third, 0, third * 2, size), fill="#45a776")
    draw.rectangle((third * 2, 0, size, size), fill="#377fc6")
    draw.ellipse((size // 4, size // 4, size * 3 // 4, size * 3 // 4), fill="#f1c74f")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_data_url(image_path: Path | None, synthetic_size: int) -> str:
    if image_path is None:
        return synthetic_image_data_url(synthetic_size)

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"unsupported image file: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percent)))
    return ordered[index]


def send_request(base_url: str, model: str, image_url: str, max_tokens: int) -> float:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": "Describe the dominant colors and central shape in one short sentence.",
                    },
                ],
            }
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with URL_OPENER.open(request, timeout=180) as response:
        result = json.loads(response.read())
    elapsed = time.perf_counter() - started
    if not result.get("choices"):
        raise RuntimeError(f"no choices returned: {result}")
    return elapsed


def run_case(args: argparse.Namespace, image_url: str, concurrency: int) -> dict[str, float]:
    before = fetch_metrics(args.base_url)
    count = concurrency * args.requests_per_worker
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                send_request,
                args.base_url,
                args.model,
                image_url,
                args.max_tokens,
            )
            for _ in range(count)
        ]
        latencies = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    after = fetch_metrics(args.base_url)

    prompt_tokens = after[METRICS[0]] - before[METRICS[0]]
    generation_tokens = after[METRICS[1]] - before[METRICS[1]]
    return {
        "concurrency": concurrency,
        "requests": count,
        "elapsed_seconds": elapsed,
        "requests_per_second": count / elapsed,
        "p50_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "total_tokens_per_second": (prompt_tokens + generation_tokens) / elapsed,
        "preemptions": after[METRICS[2]] - before[METRICS[2]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("LENTICA_VISION_ENDPOINT", "http://127.0.0.1:8100"))
    parser.add_argument("--model", default="qwen-vl-photo-context")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--synthetic-size", type=int, default=1024)
    parser.add_argument("--concurrencies", default="1,2,4,8")
    parser.add_argument("--requests-per-worker", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fetch_json(f"{args.base_url}/v1/models")
    image_url = image_data_url(args.image, args.synthetic_size)
    concurrencies = [int(value) for value in args.concurrencies.split(",")]
    results = [run_case(args, image_url, value) for value in concurrencies]

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
