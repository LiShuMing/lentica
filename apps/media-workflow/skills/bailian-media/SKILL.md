---
name: bailian-media
description: Generate images or videos interactively using this machine's Alibaba Bailian Token Plan and WSL ~/.env LLM_* configuration. Supports text-to-image, text-to-video, first-frame and reference-image video. Not for local GPU inference, unattended publishing, or ordinary pay-as-you-go API keys.
---

# Bailian Media

Use `apps/media-workflow/media` from WSL. This wrapper loads
`~/.env` itself. Never read or print the key, put it in arguments, source `.env`,
invoke raw `bl auth login`, or silently substitute another provider or endpoint.

For Windows-hosted agents use `wsl -d Ubuntu --` before the absolute command.
Run `media doctor` and `media models` if configuration/capability is uncertain.
The catalog contains documented candidates, not verified account entitlements.
Consult `apps/media-workflow/README.md` for usage and limits.

## Interactive Workflow

- Understand the intended output. For a complex request, draft a short shot list
  and separate image/animation prompts in conversation first. Do not call a
  second planning model unnecessarily; the active Agent can do the planning.
- Respect the user's explicit model. Default image: `wan2.7-image`; text video:
  `happyhorse-1.1-t2v`; first-frame video: `happyhorse-1.1-i2v`; reference video:
  `happyhorse-1.1-r2v`. Do not swap or retry models after an error without asking.
- I2V inherits the first-frame aspect ratio: do not pass `--ratio`. For a specific
  output shape, generate a source image with that shape or use reference mode.
  Resolution names are pixel-budget tiers, not a promise of fixed dimensions.
- Default to one 1024*1024 image or one 5-second 720P video. Keep provider
  watermark defaults. A generation request authorizes the requested output,
  not unbounded exploration. For setup-only or ambiguous requests, preview
  without `--execute`, and ask before quota-consuming calls. Confirm batch size
  before multi-shot generation. Never schedule or publicly post outputs.
- A dry run omits `--execute`; actual submission requires it. Use argument arrays
  or safe shell quoting; prompts are data, never shell commands.

```bash
apps/media-workflow/media image --prompt 'IMAGE DESCRIPTION' --model wan2.7-image --execute
apps/media-workflow/media video --prompt 'VIDEO DESCRIPTION' --model happyhorse-1.1-t2v --execute
apps/media-workflow/media video --image-job LOCAL_IMAGE_JOB_ID --prompt 'MOTION DESCRIPTION' --execute
apps/media-workflow/media video --image '/absolute/path/reference.png' --reference --prompt 'REFERENCE VIDEO DESCRIPTION' --execute
```

Video submission returns a local job ID and remote task ID. Use the **local** ID:

```bash
apps/media-workflow/media status LOCAL_JOB_ID
apps/media-workflow/media wait LOCAL_JOB_ID --seconds 600
```

Status/wait never submit a new generation. After a timeout, resume polling; after
download failure, retry status. A submission with no returned task ID is
uncertain: stop, report the local record, and check provider history before any
new submission. Report access/quota errors as such, without trying a pay-as-you-go
key or disabling TLS verification.

Outputs and private records live in `data/jobs/<local-id>/`. Show local image
previews when supported. Verify image dimensions or video metadata when tools are
available; distinguish downloaded media from visual quality review. Report the
actual model and result paths. Do not expose signed CDN URLs or claim untested
models work. Never include runtime records or generated assets in Git commits.
