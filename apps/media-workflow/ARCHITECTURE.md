# Implementation

## Boundaries

`Settings` parses the existing home dotenv without shell evaluation or variable
interpolation. It validates an exact Token Plan endpoint and normalizes it to an
origin for the native CLI. It never chooses an ordinary DashScope endpoint.

`Runner` is the transport boundary. `BailianCLI` invokes the pinned official CLI
with argument arrays and credentials in a child environment, a forced base URL,
an isolated configuration directory, and a bounded subprocess timeout. The CLI
owns native request schemas, uploads, model dispatch, and downloads. No generation
retry or model fallback is implemented in this wrapper.

`Jobs` owns private local state, submission checkpoints, status polling and asset
verification. A video uses asynchronous submission even when the caller wants to
wait. Submission and polling cannot be confused because they are separate methods
and CLI commands. Completed downloads are reused, and interrupted downloads can
be retried without another generation request. Failed submissions without a task
ID are explicitly uncertain, not assumed safe to retry.

`media agent` starts existing Qwen Code with the current text model, credentials
and endpoint. The `bailian-media` Skill coordinates prompts and human approval;
it is not an autonomous scheduler, a publishing bot, or a multi-shot video editor.

## State

Image: `SUBMITTING -> DOWNLOADED` or `SUBMISSION_ERROR_OR_UNKNOWN`.

Video: `SUBMITTING -> PENDING -> RUNNING -> SUCCEEDED -> DOWNLOADED`.
`FAILED`, `CANCELED`, and `UNKNOWN` stop bounded waiting. A download failure leaves
`SUCCEEDED` plus `download_error` so the next status request can retry retrieval.
Atomic JSON replacement checkpoints state. A crash before receipt of a remote ID
cannot prove whether the server accepted a request; the workflow never silently
replays such a request. Do not run concurrent status/wait processes for the same
job: this small local implementation does not provide a per-job lock.

## Native APIs

- Image: `/api/v1/services/aigc/multimodal-generation/generation`.
- Video submission: `/api/v1/services/aigc/video-generation/video-synthesis`.
- Video status: `/api/v1/tasks/<remote-id>`.

OpenAI-compatible chat completions are not used for generation because those
endpoints do not provide the required native multimodal task contract. All
credentials still come from the requested `LLM_*` configuration.

## Limits

This is a single-user interactive CLI workflow, not a production service. Header
checks detect HTML/error files masquerading as PNG/JPEG/WebP/MP4, not full media
decoding or visual quality. Signed URLs can expire; local images are re-uploaded
when used as video inputs. Model catalog entries are curated from official docs,
not discovered entitlements. Remote quota, moderation, limits, queues and model
availability remain server-controlled. Local state is private but not encrypted;
any process running as the same OS user can inspect it and process environments.

Tests exercise the state machine and transport boundary with mocks, including
the no-automatic-resubmission invariant. CLI dry-run checks verify real installed
model request formats without consuming generation quota. Real smoke-test counts
and outcomes are recorded separately in the change log.
