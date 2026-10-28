# Media Workflow

Follow root AGENTS.md and spec 0001. This optional app uses a cloud Token Plan and
does not belong to the backend local-only runtime. Generation remains dry-run
unless explicitly invoked with `--execute`; never retry an uncertain submission
or silently switch models/billing endpoints. Tests use fake runners only.
Credentials stay in a selected dotenv file and subprocess environments, never
argv/output/job records. Keep `.tools/` and `data/` private. Use the active Python
or `LENTICA_MEDIA_PYTHON`; do not install another environment as a migration step.
