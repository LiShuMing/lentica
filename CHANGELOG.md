# Changelog

## Unreleased

- Add spec-first AGENTS rules, component ownership, workspace checks, and a
  reusable spec/verification template.
- Consolidate media workflow, multimodal GUI example, vLLM service, GPU labs,
  and GPU-to-LLM learning source under explicit component boundaries.
- Keep environments, weights, media, task histories, and notebook outputs private.
- Make GUI/cloud activity explicit and vLLM startup driven by portable TOML config.

## 0.1.0 - 2026-10-28

- Extract the current Personal Knowledge Lab implementation into Lentica.
- Replace machine-specific source/model/interpreter paths with `LENTICA_*` settings.
- Keep SQLite queue, document citations, CLIP/BGE + RRF photo search, and People UI.
- Make model runtimes, HEIF, and face detection optional installation groups.
- Add CPU-only integration fixtures, package metadata, CI, and contributor docs.
- Document experimental limitations and model-weight licensing separately.
