# Verification

Date: 2026-11-21. Status: implemented; CPU/mock acceptance passed.
Baseline: `origin/main` `36508c8`; branch: `codex/project-review` (reconciled
from 2026-10-28 review baseline `1821a51`).

Reused the available Python 3.12.3 environment and Ruff 0.15.7. No new
environment or dependency was installed.

| Requirement | Command / evidence | Result |
| --- | --- | --- |
| STATE-1 | `backend/tests/test_review_reliability.py`: explicit `index_faces` unavailable runtime failure, face detector error accounting, partial photo failure, failed document job | Passed |
| QUEUE-1 | `backend/tests/test_review_reliability.py`: zero concurrency GPU fallback override, invalid JSON/types/out-of-range settings rejection, atomic replacement failure preservation | Passed |
| GOV-2 | `tests/test_workspace.py`: notebook `.ipynb` spec-change enforcement | Passed |
| GUIDE-1 | Operations runbook, reconciled 2026-10-28 review report, updated reliability backlog, and spec index linked from root docs | Added and aligned with `origin/main` (`0002`–`0010`) |
| Regression | `backend/tests/test_review_reliability.py`: `ScanCursor` bounded scans, unavailable root, oversized existing doc, parser error, pre-limit person/year filters, co-occurrence | Passed |
| All | `python -m pytest -q --disable-warnings` | 371 passed, 14 skipped |
| All | `python scripts/check_workspace.py --base origin/main` | Passed |
| All | `ruff check backend scripts tests apps examples services labs learning` | Passed |
| All | `ruff format --check scripts tests apps examples services labs learning backend/tests/test_review_reliability.py` | Passed |
| All | `git diff --check` | Passed |

No real model inference/download, GPU benchmark, vLLM startup, face biometric
evaluation, cloud/GUI action, browser E2E, personal-library scan, runtime
settings change, service restart, or push was performed. Open follow-ups
REL-4 through REL-8 remain tracked in `docs/roadmap/reliability.md`.
