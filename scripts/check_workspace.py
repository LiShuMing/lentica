"""Check spec ownership, source hygiene, and optional spec coverage for changes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PARTS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".tools",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "data",
    "outputs",
    "logs",
    "target",
    "dist",
    "build",
}
RUNTIME_SUFFIXES = {".pyc", ".db", ".log", ".onnx", ".safetensors", ".pt", ".pth", ".mp4", ".exe", ".swp"}


def git_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return sorted(set(result.stdout.decode().strip("\0").split("\0")) - {""})


def check(root: Path, paths: list[str]) -> list[str]:
    errors = []
    try:
        manifest = tomllib.loads((root / "workspace.toml").read_text())
        components = manifest["components"]
        ids = [component["id"] for component in components]
        if len(ids) != len(set(ids)):
            errors.append("workspace.toml: duplicate component IDs")
        for component in components:
            name, relative = component["id"], Path(component["path"])
            if not re.fullmatch(r"[a-z][a-z0-9-]*", name) or relative.is_absolute() or ".." in relative.parts:
                errors.append(f"Invalid component ID/path: {name}")
                continue
            location = root / relative
            for required in ("README.md", "AGENTS.md"):
                if not (location / required).is_file():
                    errors.append(f"{relative}: missing {required}")
            if not (root / component["spec"] / "spec.toml").is_file():
                errors.append(f"{name}: missing linked specification")
        for metadata in (root / "specs").glob("[0-9]*/spec.toml"):
            spec = tomllib.loads(metadata.read_text())
            if spec.get("id") != metadata.parent.name or spec.get("status") not in {"draft", "active", "implemented"}:
                errors.append(f"{metadata.relative_to(root)}: invalid ID or status")
            if not spec.get("components") or not set(spec["components"]) <= set(ids):
                errors.append(f"{metadata.relative_to(root)}: unknown or absent components")
            for required in ("README.md", "verification.md"):
                if not (metadata.parent / required).is_file():
                    errors.append(f"{metadata.parent.name}: missing {required}")
            text = (metadata.parent / "README.md").read_text()
            if not spec.get("requirements") or any(req not in text for req in spec["requirements"]):
                errors.append(f"{metadata.parent.name}: requirements absent from spec text")
    except (KeyError, OSError, ValueError) as exc:
        errors.append(f"Invalid workspace/spec metadata: {exc}")

    for name in paths:
        path = root / name
        relative = Path(name)
        if any(part in RUNTIME_PARTS for part in relative.parts) or path.suffix in RUNTIME_SUFFIXES:
            errors.append(f"{name}: runtime artifact must not be versioned")
        if path.name.startswith(".env") and path.name != ".env.example":
            errors.append(f"{name}: private environment file")
        if path.is_symlink():
            errors.append(f"{name}: source snapshot must not depend on external symlinks")
        if not path.is_file():
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            errors.append(f"{name}: oversized source artifact")
        if path.suffix == ".ipynb":
            try:
                notebook = json.loads(path.read_text())
                if any(
                    cell.get("outputs") or cell.get("execution_count") is not None or cell.get("attachments")
                    for cell in notebook.get("cells", [])
                ):
                    errors.append(f"{name}: clear notebook outputs/counts/attachments")
            except ValueError:
                errors.append(f"{name}: invalid notebook JSON")
        if path.suffix in {".py", ".sh", ".ps1", ".toml", ".md", ".json", ".ipynb", ".yml", ".yaml"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"/home/" + r"lism(?:/|\\\\)", text):
                errors.append(f"{name}: developer-specific home path")
            if re.search(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}", text):
                errors.append(f"{name}: credential-like content")
    report = root / "specs/0001-agentic-workspace/imports.json"
    if report.is_file():
        for entry in json.loads(report.read_text()):
            if not (root / entry["destination"]).is_file():
                errors.append(f"Import manifest destination missing: {entry['destination']}")
    return errors


def check_change_spec(paths: list[str]) -> list[str]:
    behavior = any(
        (
            name.startswith(
                (
                    "backend/",
                    "frontend/",
                    "apps/",
                    "examples/",
                    "services/",
                    "labs/",
                    "learning/",
                    "scripts/",
                    ".github/workflows/",
                )
            )
            and Path(name).suffix not in {".md", ".ipynb"}
        )
        or name in {"pyproject.toml", "workspace.toml"}
        for name in paths
    )
    directories = {str(Path(name).parent) for name in paths if re.match(r"specs/\d{4}-[^/]+/", name)}
    changed = set(paths)
    spec_changed = any(
        f"{directory}/verification.md" in changed
        and bool({f"{directory}/README.md", f"{directory}/spec.toml"} & changed)
        for directory in directories
    )
    return (
        ["Behavior changes require a numbered spec update AND its verification record"]
        if behavior and not spec_changed
        else []
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Compare working tree/commit against this Git revision")
    args = parser.parse_args()
    paths = git_paths(ROOT)
    errors = check(ROOT, paths)
    if args.base and set(args.base) != {"0"}:
        changed = subprocess.check_output(["git", "diff", "--name-only", args.base], cwd=ROOT, text=True).splitlines()
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True
        ).splitlines()
        errors += check_change_spec(changed + untracked)
    for error in errors:
        print(error)
    if not errors:
        print(f"Workspace checks passed ({len(paths)} source files)")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
