"""Import selected source files; default is a dry-run, never move original data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAPPING = {
    "bailian-media-workflow": "apps/media-workflow",
    "multimodal-agent": "examples/multimodal-agent",
    "pkl-vllm-service": "services/vllm",
    "gpu-lab": "labs/gpu",
    "gpu-llm-roadmap": "learning/gpu-llm",
}
EXCLUDED = {
    ".git",
    ".venv",
    ".tools",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "data",
    "outputs",
    "models",
    "logs",
    "target",
    "node_modules",
    ".ipynb_checkpoints",
    "screenshots",
    "dist",
    "build",
}
SUFFIXES = {".py", ".sh", ".ps1", ".md", ".toml", ".cu", ".cpp", ".c", ".h", ".hpp", ".rs", ".wgsl", ".txt", ".ipynb"}
EXCLUDED_FILES = {"CHNAGES_LOG.md", "TASKS.md", "tags", "requirements-lock.txt", "GUIDE.md"}


def destination_path(relative: Path) -> Path:
    if relative.name == "test_media.py":
        return Path("tests/test_media.py")
    if relative.suffix == ".ipynb":
        names = {
            "Untitled.ipynb": "03-torch-scratch.ipynb",
            "Untitled1.ipynb": "04-scratch.ipynb",
            "example1.ipynb": "05-numpy-experiment-01.ipynb",
            "example2.ipynb": "06-numpy-experiment-02.ipynb",
        }
        return relative.with_name(names.get(relative.name, relative.name.lower().replace("_", "-")))
    return relative


def selected(path: Path, source: Path) -> bool:
    relative = path.relative_to(source)
    return (
        path.is_file()
        and not path.is_symlink()
        and not any(part in EXCLUDED or part.startswith(".") for part in relative.parts)
        and path.name not in EXCLUDED_FILES
        and (path.suffix in SUFFIXES or path.name in {"media", "Makefile", "Cargo.lock"})
    )


def import_sources(source_root: Path, target: Path, execute: bool = False) -> list[dict]:
    entries = []
    for project, destination in MAPPING.items():
        source = source_root / project
        if not source.is_dir():
            raise ValueError(f"Missing source project: {project}")
        paths = []
        for directory, dirs, files in os.walk(source):
            dirs[:] = [
                name
                for name in dirs
                if name not in EXCLUDED and not name.startswith(".") and not (Path(directory) / name).is_symlink()
            ]
            paths.extend(Path(directory) / name for name in files if selected(Path(directory) / name, source))
        paths.sort()
        for path in paths:
            relative = path.relative_to(source)
            output = target / destination / destination_path(relative)
            entries.append(
                {
                    "source_project": project,
                    "source_path": relative.as_posix(),
                    "destination": output.relative_to(target).as_posix(),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "notebook_outputs_removed": path.suffix == ".ipynb",
                }
            )
            if not execute:
                continue
            if output.exists():
                raise ValueError(f"Refusing to overwrite an existing destination: {output.relative_to(target)}")
            output.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".ipynb":
                notebook = json.loads(path.read_text(encoding="utf-8"))
                notebook["metadata"] = {}
                for cell in notebook.get("cells", []):
                    cell["metadata"] = {}
                    cell.pop("attachments", None)
                    if cell["cell_type"] == "code":
                        cell["outputs"] = []
                        cell["execution_count"] = None
                output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            else:
                shutil.copy2(path, output)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.home() / "work")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    entries = import_sources(args.source_root, ROOT, args.execute)
    if args.execute:
        report = ROOT / "specs/0001-agentic-workspace/imports.json"
        report.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"execute": args.execute, "files": len(entries), "projects": list(MAPPING)}))


if __name__ == "__main__":
    main()
