# Verification

Date: 2026-10-28. Status: CPU/mock acceptance passed. Python 3.12.3, Node 24.21.0.
Existing selected environments were reused; no new virtualenv was constructed.

| Requirement | Command / evidence | Result |
| --- | --- | --- |
| GOV-1 | `python scripts/check_workspace.py --base main` | Passed; ownership, specs, source hygiene and paired spec/evidence updates checked |
| IMPORT-1 | Import report plus SHA-256 recheck against all original source files | 73 files imported; all original source hashes unchanged |
| IMPORT-1 | Notebook JSON/output checks, artifact and nested Git exclusions | Passed; no environments, weights, media, data, logs or native binaries imported |
| BOUNDARY-1 | `python -m pytest -q --disable-warnings` | 65 passed; backend, fake media transport, GUI dry-run/confirmation, service and workspace contracts |
| PORTABLE-1 | `python services/vllm/scripts/service.py plan` | Loopback argv emitted; no model runtime imported/started |
| PORTABLE-1 | GUI/image default CLI commands and media `--help` | Dry-run/help returned without provider calls, screen capture or desktop actions |
| PORTABLE-1 | `bash -n` across shell entrypoints | 11 scripts passed syntax checks |
| VERIFY-1 | `ruff check backend scripts tests apps examples services labs learning` | Passed |
| VERIFY-1 | `ruff format --check scripts tests apps examples services labs learning` | Passed |
| VERIFY-1 | `python -m build --no-isolation` | Backend sdist and wheel built |
| VERIFY-1 | `npm --prefix frontend run build` | TypeScript/Vite production build passed |

Not run: real provider generation, cloud entitlement/model freshness checks,
desktop automation, vLLM startup/inference, CUDA/HIP execution, Rust/wgpu builds,
DirectML/PowerShell on Windows, large training runs, or model downloads. The import
does not claim hardware or provider validation. CI is updated locally; this branch
has not been published or run in remote Actions during this task.

Remaining deprecation warnings are inherited UTC datetime use and installed
framework compatibility warnings; they do not indicate GPU/service validation.
