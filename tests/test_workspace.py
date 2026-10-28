import json
import shutil
from pathlib import Path

from scripts.check_workspace import check, check_change_spec
from scripts.import_workspace import selected


def fixture_workspace(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ("workspace.toml", "specs"):
        source = root / name
        if source.is_dir():
            shutil.copytree(source, tmp_path / name)
        else:
            shutil.copy2(source, tmp_path / name)
    import tomllib

    for component in tomllib.loads((tmp_path / "workspace.toml").read_text())["components"]:
        directory = tmp_path / component["path"]
        directory.mkdir(parents=True)
        (directory / "README.md").write_text("Component\n")
        (directory / "AGENTS.md").write_text("Follow root rules\n")
    (tmp_path / "specs/0001-agentic-workspace/imports.json").unlink(missing_ok=True)
    return tmp_path


def test_guard_rejects_executed_notebook_and_runtime_artifact(tmp_path):
    root = fixture_workspace(tmp_path)
    path = root / "learning/gpu-llm/test.ipynb"
    path.write_text(
        json.dumps({"cells": [{"cell_type": "code", "execution_count": 1, "outputs": [{"text": "private"}]}]})
    )
    errors = check(root, ["learning/gpu-llm/test.ipynb", "apps/media-workflow/data/job.json"])
    assert any("notebook" in error for error in errors)
    assert any("runtime artifact" in error for error in errors)


def test_guard_accepts_clean_sources_and_rejects_missing_owner_rules(tmp_path):
    root = fixture_workspace(tmp_path)
    assert check(root, []) == []
    (root / "services/vllm/AGENTS.md").unlink()
    assert any("missing AGENTS.md" in error for error in check(root, []))


def test_behavior_change_requires_numbered_spec():
    assert check_change_spec(["backend/app/main.py"])
    assert check_change_spec(["backend/app/main.py", "specs/template/README.md"])
    assert check_change_spec(["backend/app/main.py", "specs/0001-agentic-workspace/verification.md"])
    assert (
        check_change_spec(
            [
                "backend/app/main.py",
                "specs/0001-agentic-workspace/spec.toml",
                "specs/0001-agentic-workspace/verification.md",
            ]
        )
        == []
    )
    assert check_change_spec(["README.md"]) == []


def test_import_filter_excludes_environments_outputs_and_native_binary(tmp_path):
    for name in (
        ".venv/lib/runtime.py",
        "data/private.json",
        "target/debug/tool",
        "experiment",
        "code.py",
        "main.rs",
        "kernel.cu",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example")
        assert selected(path, tmp_path) is (name in {"code.py", "main.rs", "kernel.cu"})
