from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from backend.app.core.config import CLIP_MODEL, QWEN_MODEL, TEXT_MODEL


QWEN_VL_PATH = Path(QWEN_MODEL).expanduser()


@dataclass(frozen=True)
class LocalModelProfile:
    id: str
    role: str
    runtime_type: str
    model_path: str | None
    device: str
    dtype: str | None
    version: str
    memory_budget: str | None
    enabled: bool
    capabilities: dict[str, bool]
    status: str
    notes: str

    def to_api(self) -> dict[str, Any]:
        return asdict(self)


def list_local_models() -> list[LocalModelProfile]:
    clip_ready = _all_modules_present("torch", "transformers", "PIL")
    bge_ready = _all_modules_present("torch", "transformers")
    qwen_deps_ready = _all_modules_present("qwen_vl_utils", "torch", "transformers")
    qwen_status = "ready" if QWEN_VL_PATH.exists() and qwen_deps_ready else (
        "dependency_missing" if QWEN_VL_PATH.exists() else "path_missing"
    )
    return [
        LocalModelProfile(
            id="extractive-local-answerer",
            role="chat",
            runtime_type="deterministic",
            model_path=None,
            device="cpu",
            dtype=None,
            version="extractive-v1",
            memory_budget=None,
            enabled=True,
            capabilities={
                "chat": True,
                "embedding_text": False,
                "embedding_image": False,
                "vision_caption": False,
                "vision_ocr": False,
                "rerank": False,
            },
            status="ready",
            notes="Default local answer mode; assembles answers from retrieved evidence without generation.",
        ),
        LocalModelProfile(
            id=TEXT_MODEL,
            role="embedding_text",
            runtime_type="local_transformers",
            model_path=None,
            device="auto",
            dtype="auto",
            version="local_transformers",
            memory_budget="configurable",
            enabled=bge_ready,
            capabilities={
                "chat": False,
                "embedding_text": True,
                "embedding_image": False,
                "vision_caption": False,
                "vision_ocr": False,
                "rerank": False,
            },
            status="ready" if bge_ready else "dependency_missing",
            notes="Preferred local multilingual text/context embedding model for Chinese photo context and document chunks.",
        ),
        LocalModelProfile(
            id=CLIP_MODEL,
            role="embedding_image",
            runtime_type="local_transformers",
            model_path=None,
            device="auto",
            dtype="auto",
            version="local_transformers",
            memory_budget="configurable",
            enabled=clip_ready,
            capabilities={
                "chat": False,
                "embedding_text": False,
                "embedding_image": True,
                "vision_caption": False,
                "vision_ocr": False,
                "rerank": False,
            },
            status="ready" if clip_ready else "dependency_missing",
            notes="Local text-image retrieval baseline. Search uses this vector model; Qwen is not in the search path.",
        ),
        LocalModelProfile(
            id=QWEN_MODEL,
            role="vision_caption",
            runtime_type="local_transformers",
            model_path=QWEN_MODEL,
            device="cuda_if_available",
            dtype="auto",
            version="planned",
            memory_budget="one_large_model_at_a_time",
            enabled=False,
            capabilities={
                "chat": False,
                "embedding_text": False,
                "embedding_image": False,
                "vision_caption": True,
                "vision_ocr": True,
                "rerank": False,
            },
            status=qwen_status,
            notes="Local vision model for batch caption/context collection only. Requires qwen-vl-utils in the worker environment.",
        ),
        LocalModelProfile(
            id="external-api",
            role="disabled",
            runtime_type="external_disabled",
            model_path=None,
            device="none",
            dtype=None,
            version="disabled",
            memory_budget=None,
            enabled=False,
            capabilities={
                "chat": False,
                "embedding_text": False,
                "embedding_image": False,
                "vision_caption": False,
                "vision_ocr": False,
                "rerank": False,
            },
            status="disabled",
            notes="External LLM APIs are disabled by the current design guide.",
        ),
    ]


def _all_modules_present(*names: str) -> bool:
    return all(find_spec(name) is not None for name in names)
