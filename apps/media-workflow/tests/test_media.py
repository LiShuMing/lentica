from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from media import BailianCLI, Jobs, ORIGIN, Settings, WorkflowError, asset_ok, main

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 20


class FakeRunner:
    def __init__(self, replies: list[dict | Exception]) -> None:
        self.replies = iter(replies)
        self.calls: list[list[str]] = []

    def run(self, args: list[str], timeout: int | None = None) -> dict:
        self.calls.append(args.copy())
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        if args[:2] == ["video", "download"]:
            Path(args[args.index("--out") + 1]).write_bytes(MP4)
        if args[:2] == ["image", "generate"]:
            path = Path(args[args.index("--out-dir") + 1]) / "image.png"
            path.write_bytes(PNG)
            return {"urls": ["https://example.com/image.png"], "saved": [str(path)]}
        return reply


@pytest.mark.parametrize("suffix", ["", "/compatible-mode/v1", "/api/v1", "/compatible-mode/v1/"])
def test_load_only_approved_endpoint(tmp_path: Path, suffix: str) -> None:
    env = tmp_path / ".env"
    env.write_text(f"LLM_API_KEY=private-example\nLLM_BASE_URL={ORIGIN}{suffix}\nLLM_TIMEOUT=120\n")
    settings = Settings.load(env)
    assert settings.base_url == ORIGIN
    assert "private-example" not in repr(settings)


@pytest.mark.parametrize(
    "base",
    [
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ORIGIN + ".evil.test",
        ORIGIN + "?x=1",
        "http://localhost:1234",
    ],
)
def test_reject_billing_fallback_and_lookalike(tmp_path: Path, base: str) -> None:
    env = tmp_path / ".env"
    env.write_text(f"LLM_API_KEY=private-example\nLLM_BASE_URL={base}\n")
    with pytest.raises(WorkflowError):
        Settings.load(env)


def test_dry_run_does_not_call_or_create_job(tmp_path: Path) -> None:
    runner = FakeRunner([])
    jobs = Jobs(runner, tmp_path)
    result = jobs.submit("video", "model", [], "prompt", {}, False)
    assert result["dry_run"]
    assert not runner.calls
    assert not jobs.directory.exists()


def test_video_checkpoint_and_resume_never_resubmits(tmp_path: Path) -> None:
    runner = FakeRunner(
        [{"task_id": "remote-task-1"}, {"task_status": "RUNNING"}, {"task_status": "SUCCEEDED"}, {"saved": "ignored"}]
    )
    jobs = Jobs(runner, tmp_path)
    submitted = jobs.submit("video", "model", ["video", "generate", "--async"], "prompt", {}, True)
    record = jobs.read(submitted["id"])
    assert record["task_id"] == "remote-task-1"
    assert record["state"] == "PENDING"
    assert jobs.status(record["id"])["state"] == "RUNNING"
    done = jobs.status(record["id"])
    assert done["state"] == "DOWNLOADED"
    assert asset_ok(Path(done["files"][0]), True)
    assert jobs.status(record["id"])["state"] == "DOWNLOADED"
    assert sum(call[:2] == ["video", "generate"] for call in runner.calls) == 1
    assert len(runner.calls) == 4


def test_failed_task_never_downloads_or_resubmits(tmp_path: Path) -> None:
    runner = FakeRunner([{"task_id": "remote-1"}, {"task_status": "FAILED", "message": "filtered"}])
    jobs = Jobs(runner, tmp_path)
    job = jobs.submit("video", "model", ["video", "generate"], "prompt", {}, True)
    result = jobs.wait(job["id"], 1)
    assert result["state"] == "FAILED"
    assert result["message"] == "filtered"
    assert len(runner.calls) == 2


def test_uncertain_submission_is_not_retried(tmp_path: Path) -> None:
    runner = FakeRunner([WorkflowError("Connection lost")])
    jobs = Jobs(runner, tmp_path)
    with pytest.raises(WorkflowError, match="Connection lost"):
        jobs.submit("video", "model", ["video", "generate"], "prompt", {}, True)
    paths = list(jobs.directory.glob("*/job.json"))
    assert len(paths) == 1
    job = json.loads(paths[0].read_text())
    assert job["state"] == "SUBMISSION_ERROR_OR_UNKNOWN"
    assert jobs.status(job["id"])["state"] == "SUBMISSION_ERROR_OR_UNKNOWN"
    assert len(runner.calls) == 1


def test_download_failure_can_be_retried_without_new_generation(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            {"task_id": "remote-1"},
            {"task_status": "SUCCEEDED"},
            WorkflowError("Download interrupted"),
            {"task_status": "SUCCEEDED"},
            {},
        ]
    )
    jobs = Jobs(runner, tmp_path)
    job = jobs.submit("video", "model", ["video", "generate"], "prompt", {}, True)
    with pytest.raises(WorkflowError, match="Download interrupted"):
        jobs.status(job["id"])
    assert jobs.read(job["id"])["state"] == "SUCCEEDED"
    assert jobs.status(job["id"])["state"] == "DOWNLOADED"
    assert sum(call[:2] == ["video", "generate"] for call in runner.calls) == 1


def test_image_download_and_private_record(tmp_path: Path) -> None:
    jobs = Jobs(FakeRunner([{}]), tmp_path)
    result = jobs.submit("image", "model", ["image", "generate"], "prompt", {}, True)
    assert result["state"] == "DOWNLOADED"
    assert jobs.path(result["id"]).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("value", ["../outside", "remote-task", "x" * 32, "/tmp/job.json"])
def test_local_job_id_cannot_traverse(tmp_path: Path, value: str) -> None:
    with pytest.raises(WorkflowError):
        Jobs(FakeRunner([]), tmp_path).path(value)


def test_cli_secret_in_environment_not_argv_and_endpoint_is_forced(tmp_path: Path) -> None:
    cli = BailianCLI(Settings(key="private-example"), tmp_path)
    cli.entry.parent.mkdir(parents=True)
    cli.entry.touch()
    completed = subprocess.CompletedProcess([], 0, '{"task_id":"remote-1"}', "")
    with patch("media.subprocess.run", return_value=completed) as run:
        assert cli.run(["video", "generate", "--async"])["task_id"] == "remote-1"
        argv = run.call_args.args[0]
        assert "private-example" not in " ".join(argv)
        assert run.call_args.kwargs["env"]["DASHSCOPE_API_KEY"] == "private-example"
        assert argv[argv.index("--base-url") + 1] == ORIGIN
        assert run.call_args.kwargs["env"]["BAILIAN_CONFIG_DIR"].startswith(str(tmp_path))


def test_cli_redacts_error_and_does_not_retry(tmp_path: Path) -> None:
    cli = BailianCLI(Settings(key="private-example"), tmp_path)
    cli.entry.parent.mkdir(parents=True)
    cli.entry.touch()
    completed = subprocess.CompletedProcess([], 1, '{"error":"private-example sk-sp-abcd"}', "")
    with patch("media.subprocess.run", return_value=completed) as run:
        with pytest.raises(WorkflowError) as error:
            cli.run(["video", "generate"])
        assert "private-example" not in str(error.value)
        assert "sk-sp-abcd" not in str(error.value)
        assert run.call_count == 1


def test_cli_timeout_never_prints_process_env_or_resubmits(tmp_path: Path) -> None:
    cli = BailianCLI(Settings(key="private-example"), tmp_path)
    cli.entry.parent.mkdir(parents=True)
    cli.entry.touch()
    with patch("media.subprocess.run", side_effect=subprocess.TimeoutExpired("command", 1)) as run:
        with pytest.raises(WorkflowError, match="do not resubmit"):
            cli.run(["video", "generate"])
        assert run.call_count == 1


def test_bad_download_is_not_completed(tmp_path: Path) -> None:
    path = tmp_path / "bad.mp4"
    path.write_text("<html>not a media file</html>")
    assert not asset_ok(path, True)
    assert not asset_ok(path, False)


@pytest.mark.parametrize("reference,expected", [(False, "happyhorse-1.1-i2v"), (True, "happyhorse-1.1-r2v")])
def test_image_video_dispatch_and_aspect_ratio(tmp_path: Path, reference: bool, expected: str) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(PNG)
    argv = ["media", "video", "--image", str(image), "--prompt", "Slow motion"]
    if reference:
        argv += ["--reference"]
    with (
        patch("media.sys.argv", argv),
        patch("media.Settings.load", return_value=Settings(key="dummy")),
        patch("media.Jobs.submit", return_value={"dry_run": True}) as submit,
    ):
        assert main() == 0
        args = submit.call_args.args
        assert args[1] == expected
        assert ("--ratio" in args[2]) == reference
        assert args[4]["ratio"] == ("16:9" if reference else None)
        assert args[5] is False


def test_i2v_explicit_ratio_rejected_before_submission(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(PNG)
    argv = ["media", "video", "--image", str(image), "--prompt", "Slow motion", "--ratio", "16:9", "--execute"]
    with (
        patch("media.sys.argv", argv),
        patch("media.Settings.load", return_value=Settings(key="dummy")),
        patch("media.Jobs.submit") as submit,
    ):
        assert main() == 1
        submit.assert_not_called()
