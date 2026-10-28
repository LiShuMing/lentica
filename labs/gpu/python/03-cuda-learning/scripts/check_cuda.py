"""Check installed software separately from usable CUDA hardware."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def check_cuda() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cuda_path": os.environ.get("CUDA_PATH"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_verified": False,
    }
    for package in ("numpy", "pandas", "torch", "cupy-cuda13x", "jupyterlab"):
        try:
            report[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report[package] = None
    for command in ("nvcc", "nvidia-smi"):
        executable = shutil.which(command)
        report[command] = {"path": executable}
        if executable:
            try:
                result = subprocess.run(
                    [executable, "--version"] if command == "nvcc" else [executable],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=20,
                )
                report[command].update(
                    exit_code=result.returncode,
                    output=(result.stdout + result.stderr).strip(),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                report[command]["error"] = str(exc)
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int()
        driver.cuDriverGetVersion(ctypes.byref(version))
        init_code = driver.cuInit(0)
        error_name = ctypes.c_char_p()
        driver.cuGetErrorName(init_code, ctypes.byref(error_name))
        report["driver"] = {
            "api_version": version.value,
            "cuInit": init_code,
            "error_name": error_name.value.decode() if error_name.value else None,
        }
    except (OSError, AttributeError) as exc:
        report["driver"] = {"error": str(exc)}
    try:
        import torch

        available = torch.cuda.is_available()
        report["pytorch"] = {
            "import_ok": True,
            "build_cuda": torch.version.cuda,
            "cuda_available": available,
            "device_count": torch.cuda.device_count(),
            "gpu_test_passed": False,
        }
        if available:
            a = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
            result = (a @ a.T).cpu().numpy()
            expected = np.arange(12, dtype=np.float32).reshape(3, 4)
            np.testing.assert_allclose(result, expected @ expected.T)
            torch.cuda.synchronize()
            report["pytorch"].update(
                gpu_test_passed=True,
                device_name=torch.cuda.get_device_name(0),
                capability=list(torch.cuda.get_device_capability(0)),
            )
    except Exception as exc:
        report.setdefault("pytorch", {})["error"] = f"{type(exc).__name__}: {exc}"
    try:
        import cupy as cp

        report["cupy"] = {"import_ok": True, "gpu_test_passed": False}
        count = cp.cuda.runtime.getDeviceCount()
        report["cupy"]["device_count"] = count
        if count:
            a = cp.arange(12, dtype=cp.float32).reshape(3, 4)
            result = cp.asnumpy(a @ a.T)
            expected = np.arange(12, dtype=np.float32).reshape(3, 4)
            np.testing.assert_allclose(result, expected @ expected.T)
            cp.cuda.Stream.null.synchronize()
            report["cupy"]["gpu_test_passed"] = True
    except Exception as exc:
        report.setdefault("cupy", {})["error"] = f"{type(exc).__name__}: {exc}"
    report["gpu_verified"] = bool(
        report.get("pytorch", {}).get("gpu_test_passed") and report.get("cupy", {}).get("gpu_test_passed")
    )
    report["environment_ready"] = bool(
        all(report[name] for name in ("numpy", "pandas", "torch", "cupy-cuda13x", "jupyterlab"))
        and report.get("pytorch", {}).get("build_cuda")
        and report.get("pytorch", {}).get("import_ok")
        and report.get("cupy", {}).get("import_ok")
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_cuda()
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["environment_ready"]:
        return 1
    return int(args.require_gpu and not report["gpu_verified"])


if __name__ == "__main__":
    raise SystemExit(main())
