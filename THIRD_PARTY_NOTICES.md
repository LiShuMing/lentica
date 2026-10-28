# Third-Party Notices

Lentica was extracted from Personal Knowledge Lab by LiShuMing. The local photo
context adapter and batching approach were informed by the author's photo_rag
application under LLaMA-Factory/apps. This repository does not vendor that runtime,
its virtualenv, or its model weights.

Dependencies retain their own license terms. The following deserve explicit review:

- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/about.html) is offered under
  AGPL or a commercial license. Lentica is distributed under AGPL-3.0-only.
- [InsightFace](https://github.com/deepinsight/insightface) code and pretrained
  model weights have separate terms. Its published pretrained models are for
  non-commercial research by default. Face indexing is opt-in; no weights are
  included and the Lentica license does not relicense them.
- Qwen, CLIP, BGE, and all other checkpoints retain the terms attached to their
  exact model/revision. Obtain them separately and check model cards before use.
- vLLM is an independently installed service; it is not vendored or required for
  document retrieval and smoke tests.

These notes are not a complete license inventory of transitive dependencies.
