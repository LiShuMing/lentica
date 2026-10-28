"""Plan, start, or inspect a separately provisioned local vLLM service."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/qwen-vl.example.toml"


def load_config(path: Path) -> dict:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    required = {
        "model_path",
        "served_model_name",
        "host",
        "port",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "dtype",
        "max_pixels",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError("Missing service configuration: " + ", ".join(missing))
    for key, name in (("host", "LENTICA_VLLM_HOST"), ("model_path", "LENTICA_QWEN_MODEL")):
        if name in os.environ:
            config[key] = os.environ[name]
    if "LENTICA_VLLM_PORT" in os.environ:
        config["port"] = int(os.environ["LENTICA_VLLM_PORT"])
    if not all(
        isinstance(config[key], str) and config[key] for key in ("host", "model_path", "served_model_name", "dtype")
    ):
        raise ValueError("Host, model path, served name, and dtype must be nonempty strings")
    if config["host"] != "localhost" and not ipaddress.ip_address(config["host"]).is_loopback:
        raise ValueError("Service host must be loopback")
    if type(config["port"]) is not int or not 1 <= config["port"] <= 65535:
        raise ValueError("Invalid service port")
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "max_pixels"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["gpu_memory_utilization"]) not in (int, float) or not 0 < config["gpu_memory_utilization"] < 1:
        raise ValueError("GPU memory utilization must be between 0 and 1")
    if not isinstance(config.get("environment", {}), dict) or any(
        not isinstance(value, str) for value in config.get("environment", {}).values()
    ):
        raise ValueError("Environment entries must be strings")
    model = Path(config["model_path"]).expanduser()
    config["model_path"] = str(model if model.is_absolute() else ROOT / model)
    return config


def command(config: dict) -> list[str]:
    args = [os.environ.get("LENTICA_VLLM_EXECUTABLE", "vllm"), "serve", config["model_path"]]
    for key in (
        "host",
        "port",
        "served_model_name",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "dtype",
    ):
        args += ["--" + key.replace("_", "-"), str(config[key])]
    return args + [
        "--limit-mm-per-prompt",
        '{"image":1,"video":0}',
        "--mm-processor-kwargs",
        json.dumps({"max_pixels": config["max_pixels"]}),
    ]


def health(config: dict) -> dict:
    host = config["host"]
    host = f"[{host}]" if ":" in host else host
    url = f"http://{host}:{config['port']}/v1/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as response:
        payload = json.load(response)
    models = [item["id"] for item in payload.get("data", [])]
    if config["served_model_name"] not in models:
        raise ValueError("Configured model is not served")
    return {"healthy": True, "models": models}


def start(config: dict) -> int:
    if not Path(config["model_path"]).is_dir():
        raise ValueError("Provision the model directory before starting vLLM")
    log_dir = Path(os.environ.get("LENTICA_VLLM_LOG_DIR", str(ROOT / "logs"))).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ | config.get("environment", {})
    if env.get("LENTICA_VLLM_LIBRARY_PATH"):
        env["LD_LIBRARY_PATH"] = env["LENTICA_VLLM_LIBRARY_PATH"] + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    with (log_dir / "qwen-vl.log").open("a", encoding="utf-8") as log:
        with subprocess.Popen(
            command(config), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as proc:
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return 130


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "start", "health", "check-env"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.action == "plan":
            print(json.dumps({"argv": command(config), "environment": config.get("environment", {})}, indent=2))
        elif args.action == "start":
            return start(config)
        elif args.action == "health":
            print(json.dumps(health(config)))
        else:
            python = os.environ.get("LENTICA_VLLM_PYTHON", sys.executable)
            return subprocess.call(
                [
                    python,
                    "-c",
                    "import bz2,torch,vllm; print(torch.__version__, vllm.__version__); print('cuda:',torch.cuda.is_available())",
                ]
            )
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
