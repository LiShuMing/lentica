# Workspace Migration

Spec: [0001-agentic-workspace](../specs/0001-agentic-workspace/README.md).
The import records 73 selected source files from five projects; file-level source
SHA-256 values and new destinations are in that spec's `imports.json`. They record
provenance, not checksums that prevent normal future edits to the imported code.

| Old directory under ~/work | New directory under Lentica |
| --- | --- |
| bailian-media-workflow | apps/media-workflow |
| multimodal-agent | examples/multimodal-agent |
| pkl-vllm-service | services/vllm |
| gpu-lab | labs/gpu |
| gpu-llm-roadmap | learning/gpu-llm |

Original directories, their Git state, and existing environments/services remain
in place. Switch working directories when adopting the new source. Do not run
both old and new model servers on the same port or let two product backends own
the same data directory. There is no automatic symlink, service restart, or
destructive move. Existing media Skill links were not changed.

The importer is dry-run by default: `python scripts/import_workspace.py`.
`--execute` is for an empty destination and refuses existing files. Importing source
is not a release/build pipeline; edits made during normalization are recorded by
Git and the migration spec. It excludes nested Git repositories, environments,
models, data, generated outputs, tool caches, executables, and runtime logs.

Notebooks are copied without outputs, execution counts, attachments, or machine
metadata. Names use lowercase kebab-case; ambiguous historical scratch filenames
are renamed but their source cells are retained. Private historical generation
logs and task IDs were intentionally not imported into the media README.

Use component requirements/selected interpreters independently. The root package
still installs only the knowledge product plus explicit extras. Default root tests
run mocks and tiny CPU fixtures; they do not start vLLM, use a provider, capture a
screen, build GPU drivers, or download training data.
