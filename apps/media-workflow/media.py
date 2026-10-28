#!/usr/bin/env python3
"""Interactive Token Plan media jobs backed by the official Bailian CLI."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
ORIGIN = "https://token-plan.cn-beijing.maas.aliyuncs.com"
CATALOG: dict[str, list[str]] = {
    "image": ["wan2.7-image", "wan2.7-image-pro", "qwen-image-3.0-pro", "qwen-image-2.0", "qwen-image-2.0-pro"],
    "t2v": ["happyhorse-1.1-t2v"],
    "i2v": ["happyhorse-1.1-i2v"],
    "r2v": ["happyhorse-1.1-r2v"],
}


class WorkflowError(Exception):
    """A safe, user-facing workflow failure."""


def emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


@dataclass(frozen=True)
class Settings:
    key: str = field(repr=False)
    base_url: str = ORIGIN
    timeout: int = 120
    text_model: str = ""

    @classmethod
    def load(cls, env_file: Path | None = None) -> Settings:
        # Parse, never source a shell file. Only this file supplies credentials.
        config = env_file or Path(os.environ.get("LENTICA_MEDIA_ENV_FILE", str(Path.home() / ".env"))).expanduser()
        values = dotenv_values(config, interpolate=False)
        key = values.get("LLM_API_KEY") or ""
        base = (values.get("LLM_BASE_URL") or "").rstrip("/")
        if base not in (ORIGIN, ORIGIN + "/compatible-mode/v1", ORIGIN + "/api/v1"):
            raise WorkflowError("LLM_BASE_URL is not the approved Beijing Token Plan endpoint.")
        if not key or "\n" in key or "\r" in key:
            raise WorkflowError("LLM_API_KEY is missing or invalid in ~/.env.")
        try:
            timeout = int(values.get("LLM_TIMEOUT") or "120")
        except ValueError:
            raise WorkflowError("LLM_TIMEOUT must be seconds, as an integer.") from None
        if not 10 <= timeout <= 1800:
            raise WorkflowError("LLM_TIMEOUT must be between 10 and 1800 seconds.")
        return cls(key=key, timeout=timeout, text_model=values.get("LLM_MODEL") or "")

    def redact(self, text: str) -> str:
        return re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", text.replace(self.key, "[redacted]"))


class Runner(Protocol):
    def run(self, args: list[str], timeout: int | None = None) -> dict[str, Any]: ...


class BailianCLI:
    def __init__(self, settings: Settings, root: Path = ROOT) -> None:
        self.settings = settings
        self.root = root
        self.entry = Path(
            os.environ.get("LENTICA_MEDIA_CLI", str(root / ".tools/node_modules/bailian-cli/dist/bailian.mjs"))
        )
        # Prefer an existing modern runtime without changing the user's PATH.
        candidates = sorted((Path.home() / ".nvm/versions/node").glob("v24.*/bin/node"))
        self.node = str(candidates[-1]) if candidates else shutil.which("node") or "node"

    def run(self, args: list[str], timeout: int | None = None) -> dict[str, Any]:
        if not self.entry.is_file():
            raise WorkflowError("Project CLI missing; run the installation command in README.md.")
        env = os.environ.copy()
        env.pop("NODE_OPTIONS", None)
        env.pop("NODE_TLS_REJECT_UNAUTHORIZED", None)
        env.update(
            {
                "DASHSCOPE_API_KEY": self.settings.key,
                "DASHSCOPE_BASE_URL": self.settings.base_url,
                "BAILIAN_BASE_URL": self.settings.base_url,
                "BAILIAN_CONFIG_DIR": str(self.root / "data/cli-config"),
                "NO_COLOR": "1",
            }
        )
        config = self.root / "data/cli-config/config.json"
        if not config.exists():
            write_json(config, {"telemetry": False, "language": "en-US"})
        command = [
            self.node,
            str(self.entry),
            *args,
            "--output",
            "json",
            "--base-url",
            self.settings.base_url,
            "--timeout",
            str(self.settings.timeout),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=self.root,
                env=env,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=timeout or self.settings.timeout + 60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise WorkflowError(
                "CLI timed out. Submission may have succeeded; do not resubmit automatically."
            ) from None
        except OSError as error:
            raise WorkflowError(self.settings.redact(str(error))) from None
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            result = None
        if completed.returncode or not isinstance(result, dict):
            detail = json.dumps(result) if result else completed.stderr[-1800:]
            raise WorkflowError(self.settings.redact(f"CLI failed ({completed.returncode}): {detail}"))
        if result.get("error"):
            raise WorkflowError(self.settings.redact(json.dumps(result["error"])))
        return result


def asset_ok(path: Path, video: bool) -> bool:
    if not path.is_file() or path.stat().st_size < 12:
        return False
    with path.open("rb") as stream:
        header = stream.read(32)
    if video:
        return header[4:8] == b"ftyp"
    return (
        header.startswith(b"\x89PNG\r\n\x1a\n")
        or header.startswith(b"\xff\xd8\xff")
        or (header[:4] == b"RIFF" and header[8:12] == b"WEBP")
    )


class Jobs:
    def __init__(self, runner: Runner, root: Path = ROOT) -> None:
        self.runner = runner
        self.directory = root / "data/jobs"

    def path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise WorkflowError("Use the local 32-character job ID, not a path or remote task ID.")
        return self.directory / job_id / "job.json"

    def read(self, job_id: str) -> dict[str, Any]:
        try:
            return json.loads(self.path(job_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise WorkflowError("Local job record is missing or invalid.") from None

    def save(self, job: dict[str, Any]) -> None:
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(self.path(job["id"]), job)

    def submit(
        self, kind: str, model: str, args: list[str], prompt: str, parameters: dict[str, Any], execute: bool
    ) -> dict[str, Any]:
        if not execute:
            return {
                "dry_run": True,
                "kind": kind,
                "model": model,
                "prompt": prompt,
                "parameters": parameters,
                "endpoint": ORIGIN,
                "note": "No API request. Add --execute to consume plan quota.",
            }
        job: dict[str, Any] = {
            "id": uuid4().hex,
            "kind": kind,
            "model": model,
            "prompt": prompt,
            "parameters": parameters,
            "state": "SUBMITTING",
            "files": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save(job)
        emit({"job": job["id"], "state": "SUBMITTING", "record": str(self.path(job["id"]))})
        try:
            if kind == "image":
                args += ["--out-dir", str(self.path(job["id"]).parent), "--out-prefix", "image"]
            result = self.runner.run(args)
            job["result"] = result
            if kind == "image":
                files = [str(Path(p).resolve()) for p in result.get("saved", [])]
                if not files or not all(asset_ok(Path(p), False) for p in files):
                    raise WorkflowError("Image response did not contain valid downloaded images.")
                job.update(state="DOWNLOADED", files=files)
            else:
                task = result.get("task_id")
                if not isinstance(task, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", task):
                    raise WorkflowError("No valid task ID returned. Do not resubmit automatically.")
                job.update(state="PENDING", task_id=task)
            self.save(job)
        except (WorkflowError, KeyboardInterrupt) as error:
            # Keep uncertain writes distinct from definite server-side task failure.
            job.update(state="SUBMISSION_ERROR_OR_UNKNOWN", error=str(error) or "Interrupted")
            self.save(job)
            if isinstance(error, KeyboardInterrupt):
                raise
            raise WorkflowError(f"Job {job['id']}: {error}") from None
        return self.summary(job)

    def status(self, job_id: str) -> dict[str, Any]:
        job = self.read(job_id)
        if job["state"] == "DOWNLOADED" or not job.get("task_id"):
            return self.summary(job)
        result = self.runner.run(["video", "task", "get", "--task-id", job["task_id"]])
        job["last_status"] = result
        state = result.get("task_status", "UNKNOWN")
        job["state"] = state
        self.save(job)
        if state == "SUCCEEDED":
            destination = self.path(job_id).parent / "video.mp4"
            temporary = destination.with_suffix(".partial.mp4")
            try:
                self.runner.run(["video", "download", "--task-id", job["task_id"], "--out", str(temporary)])
                if not asset_ok(temporary, True):
                    raise WorkflowError("Downloaded file is not an MP4; retry status, not generation.")
                temporary.replace(destination)
                job.update(state="DOWNLOADED", files=[str(destination)])
                job.pop("download_error", None)
            except WorkflowError as error:
                job["download_error"] = str(error)
                self.save(job)
                raise
            self.save(job)
        return self.summary(job)

    def wait(self, job_id: str, seconds: int, interval: int = 15) -> dict[str, Any]:
        deadline = time.monotonic() + seconds
        while True:
            result = self.status(job_id)
            if result["state"] in {"DOWNLOADED", "FAILED", "CANCELED", "UNKNOWN", "SUBMISSION_ERROR_OR_UNKNOWN"}:
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {**result, "note": "Wait budget reached. Resume with status/wait; no new generation."}
            print(f"{job_id}: {result['state']}", file=sys.stderr)
            time.sleep(min(interval, remaining))

    def summary(self, job: dict[str, Any]) -> dict[str, Any]:
        keys = ("id", "kind", "model", "state", "task_id", "files", "error", "download_error")
        result = {key: job[key] for key in keys if key in job}
        result["record"] = str(self.path(job["id"]))
        if job.get("last_status", {}).get("message"):
            result["message"] = job["last_status"]["message"]
        return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Local configuration check; no model calls")
    commands.add_parser("models", help="Documented candidates, not verified account entitlements")
    commands.add_parser("jobs", help="List local jobs")
    commands.add_parser("agent", help="Start interactive Qwen Code using LLM_* and existing Node 24")
    for name in ("image", "video"):
        command = commands.add_parser(name)
        command.add_argument("--prompt", required=True)
        command.add_argument("--model")
        command.add_argument("--execute", action="store_true", help="Submit ONE quota-consuming request")
        if name == "image":
            command.add_argument("--size", default="1024*1024")
        else:
            source = command.add_mutually_exclusive_group()
            source.add_argument("--image", help="Local image or HTTPS image URL; uploaded to provider")
            source.add_argument("--image-job", help="Use a completed local image job as first frame")
            command.add_argument(
                "--reference", action="store_true", help="Use reference-to-video instead of first frame"
            )
            command.add_argument("--duration", type=int, choices=range(3, 16), default=5)
            command.add_argument("--resolution", choices=["480P", "720P", "1080P"], default="720P")
            command.add_argument(
                "--ratio",
                choices=["16:9", "9:16", "1:1"],
                help="t2v/r2v only (default 16:9); i2v inherits its first frame",
            )
    for name in ("status", "wait"):
        command = commands.add_parser(name)
        command.add_argument("job")
        if name == "wait":
            command.add_argument("--seconds", type=int, choices=range(1, 1801), default=600)
    return result


def main() -> int:
    os.umask(0o077)
    args = parser().parse_args()
    try:
        if args.command == "models":
            emit(
                {
                    "models": CATALOG,
                    "verified_for_account": False,
                    "checked_docs": "2026-10-22",
                    "note": "Availability depends on plan; no automatic fallback.",
                }
            )
            return 0
        settings = Settings.load()
        runner = BailianCLI(settings)
        jobs = Jobs(runner)
        if args.command == "agent":
            entry = os.environ.get("LENTICA_QWEN_CODE") or shutil.which("qwen")
            if not entry or not settings.text_model:
                raise WorkflowError("Qwen Code or LLM_MODEL is missing.")
            env = os.environ.copy()
            env.update(
                {
                    "OPENAI_API_KEY": settings.key,
                    "OPENAI_BASE_URL": ORIGIN + "/compatible-mode/v1",
                    "OPENAI_MODEL": settings.text_model,
                }
            )
            env.pop("NODE_OPTIONS", None)
            env.pop("NODE_TLS_REJECT_UNAUTHORIZED", None)
            env["PATH"] = str(Path(runner.node).parent) + os.pathsep + env.get("PATH", "")
            os.chdir(ROOT)
            os.execvpe(entry, [entry, "--auth-type", "openai", "--model", settings.text_model], env)
        elif args.command == "doctor":
            emit(
                {
                    "config_valid": True,
                    "api_key": "[configured, not displayed]",
                    "origin": settings.base_url,
                    "text_model": settings.text_model,
                    "timeout_seconds": settings.timeout,
                    "node": runner.node,
                    "cli_installed": runner.entry.is_file(),
                    "cli_version": "2.5.0",
                    "account_access_verified": False,
                    "gpu_required": False,
                }
            )
        elif args.command == "jobs":
            emit(
                [
                    jobs.summary(json.loads(p.read_text(encoding="utf-8")))
                    for p in sorted(jobs.directory.glob("*/job.json"))
                ]
            )
        elif args.command in ("status", "wait"):
            value = jobs.status(args.job) if args.command == "status" else jobs.wait(args.job, args.seconds)
            emit(value)
            return 1 if value["state"] in {"FAILED", "CANCELED", "UNKNOWN", "SUBMISSION_ERROR_OR_UNKNOWN"} else 0
        else:
            max_prompt = 6000 if args.command == "image" else 2500
            if not args.prompt.strip() or len(args.prompt) > max_prompt:
                raise WorkflowError(f"Provide a nonempty prompt of at most {max_prompt} characters.")
            kind = "image"
            image = None
            if args.command == "video":
                image = args.image
                if args.image_job:
                    previous = jobs.read(args.image_job)
                    if previous.get("kind") != "image" or previous.get("state") != "DOWNLOADED":
                        raise WorkflowError("--image-job requires a successfully downloaded image job.")
                    # Local file avoids dependence on an expired, signed image URL.
                    image = previous["files"][0]
                if image:
                    if image.startswith("https://"):
                        url = urlsplit(image)
                        if not url.hostname or url.username or url.password or url.fragment:
                            raise WorkflowError("Invalid HTTPS source image URL.")
                    elif not Path(image).expanduser().is_file():
                        raise WorkflowError("Source image must be an existing local file or HTTPS URL.")
                    else:
                        image = str(Path(image).expanduser().resolve())
                    kind = "r2v" if args.reference else "i2v"
                elif args.reference:
                    raise WorkflowError("--reference requires --image or --image-job.")
                else:
                    kind = "t2v"
                if kind == "i2v" and args.ratio is not None:
                    raise WorkflowError(
                        "i2v inherits its first-frame aspect ratio; omit --ratio. Use a matching source image or --reference instead."
                    )
            model = args.model or CATALOG[kind][0]
            if model not in CATALOG[kind]:
                raise WorkflowError(f"Unsupported {kind} model. See models; no silent fallback.")
            if args.command == "image":
                if not re.fullmatch(r"[0-9]{3,4}\*[0-9]{3,4}", args.size):
                    raise WorkflowError("Image size must be W*H pixels, e.g. 1024*1024.")
                parameters = {"size": args.size, "n": 1}
                cli_args = [
                    "image",
                    "generate",
                    "--model",
                    model,
                    "--prompt",
                    args.prompt,
                    "--size",
                    args.size,
                    "--n",
                    "1",
                    "--concurrent",
                    "1",
                ]
            else:
                ratio = None if kind == "i2v" else args.ratio or "16:9"
                parameters = {"duration": args.duration, "resolution": args.resolution, "ratio": ratio, "image": image}
                cli_args = [
                    "video",
                    "ref" if kind == "r2v" else "generate",
                    "--model",
                    model,
                    "--prompt",
                    args.prompt,
                    "--duration",
                    str(args.duration),
                    "--resolution",
                    args.resolution,
                    "--async",
                    "--concurrent",
                    "1",
                ]
                if ratio:
                    cli_args += ["--ratio", ratio]
                if image:
                    cli_args += ["--image", image]
            emit(
                jobs.submit(
                    "image" if kind == "image" else "video", model, cli_args, args.prompt, parameters, args.execute
                )
            )
        return 0
    except WorkflowError as error:
        emit({"error": str(error)})
        return 1
    except KeyboardInterrupt:
        emit({"interrupted": True, "note": "Resume a known task with status/wait. Never blindly resubmit."})
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
