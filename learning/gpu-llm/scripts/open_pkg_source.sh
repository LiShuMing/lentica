#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <package_name>"
  exit 1
fi

PKG_NAME="$1"

# Resolve package source location using the active Python interpreter.
readarray -t RESOLVED < <(python - "$PKG_NAME" <<'PY'
import importlib
import pathlib
import sys

name = sys.argv[1]
mod = importlib.import_module(name)
file = pathlib.Path(mod.__file__).resolve() if getattr(mod, "__file__", None) else None
if file is None:
    print("ERR")
    print(f"Module '{name}' has no __file__. It may be a namespace/built-in module.")
    sys.exit(2)

if file.name == "__init__.py":
    entry = file
    root = file.parent
else:
    entry = file
    root = file.parent

print(str(root))
print(str(entry))
PY
)

if [[ "${#RESOLVED[@]}" -lt 2 ]]; then
  echo "Failed to resolve module: ${PKG_NAME}"
  exit 2
fi

ROOT_DIR="${RESOLVED[0]}"
ENTRY_FILE="${RESOLVED[1]}"

echo "Package root: ${ROOT_DIR}"
echo "Entry file : ${ENTRY_FILE}"

# Open the entry source file in VS Code if the 'code' CLI is available.
if command -v code >/dev/null 2>&1; then
  code -g "${ENTRY_FILE}" >/dev/null 2>&1 || true
else
  echo "Tip: install the 'code' CLI to open files automatically."
fi
