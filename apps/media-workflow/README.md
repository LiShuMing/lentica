# Media Workflow

Optional interactive image/video generation through the official Bailian Token
Plan CLI. This cloud app is separate from Lentica's local-only knowledge backend.
Spec: [0001](../../specs/0001-agentic-workspace/README.md).

Use an existing Python environment with `python-dotenv`; mocked tests use pytest.
Provision the pinned CLI separately when you choose to generate media:

```bash
cd apps/media-workflow
npm install --prefix .tools --ignore-scripts --no-audit --no-fund --save-exact bailian-cli@2.5.0
./media models
./media doctor
./media image --model wan2.7-image --prompt 'A clean database illustration'
```

`models` and `--help` do not load credentials. `doctor` reads config locally but
does not contact the provider. Default image/video commands are dry-runs;
`--execute` submits one quota-consuming request. `status` / `wait` resume a known
video job without resubmitting generation. An uncertain submission must not be
automatically retried. Existing image/video model names are a dated catalog,
not a promise of account entitlement or permanent provider support.

Credentials come from `~/.env` or `LENTICA_MEDIA_ENV_FILE`, parsed without shell
evaluation/interpolation. Only the approved Beijing Token Plan origin is accepted;
there is no pay-as-you-go fallback. Use `LENTICA_MEDIA_PYTHON` for a selected Python,
`LENTICA_MEDIA_CLI` for a preinstalled CLI entry, and `LENTICA_QWEN_CODE` for an
optional Qwen Code executable. No global environment or skill links are modified.

An explicit source image may be uploaded to the provider during generation. Job
records, prompts, signed URLs, media, and CLI config stay under private `data/`;
`.tools/` is also excluded. No historical generation records are included here.

```bash
python -m pytest apps/media-workflow/tests  # from repository root; mocked only
```

See [ARCHITECTURE.md](ARCHITECTURE.md), [CHANGELOG.md](CHANGELOG.md), and the
optional [skill instructions](skills/bailian-media/SKILL.md).
