from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name) or default).expanduser().resolve()


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


DATA_DIR = env_path("LENTICA_DATA_DIR", PROJECT_ROOT / "data")
DB_PATH = DATA_DIR / "knowledge.db"
DOCUMENTS_SOURCE = env_path("LENTICA_DOCUMENTS_SOURCE", PROJECT_ROOT / "library" / "documents")
PHOTOS_SOURCE = env_path("LENTICA_PHOTOS_SOURCE", PROJECT_ROOT / "library" / "photos")
USER_ENV_PATH = PROJECT_ROOT / ".env"
VISION_ENDPOINT = os.environ.get("LENTICA_VISION_ENDPOINT", "http://127.0.0.1:8100")
VISION_MODEL = os.environ.get("LENTICA_VISION_MODEL", "qwen-vl-photo-context")
QWEN_MODEL = os.environ.get("LENTICA_QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
CLIP_MODEL = os.environ.get("LENTICA_CLIP_MODEL", "openai/clip-vit-base-patch32")
TEXT_MODEL = os.environ.get("LENTICA_TEXT_MODEL", "BAAI/bge-m3")
WORKER_PYTHON = os.environ.get("LENTICA_WORKER_PYTHON") or sys.executable


class RuntimeSettings(BaseModel):
    mode: str = "local_only"
    external_api_enabled: bool = False
    legacy_llm_env_present: bool = False


class AppSettings(BaseModel):
    runtime: RuntimeSettings
    database_url: str
    documents_source: Path = DOCUMENTS_SOURCE
    photos_source: Path = PHOTOS_SOURCE

    @classmethod
    def load(cls) -> Self:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return cls(
            runtime=RuntimeSettings(),
            database_url=f"sqlite:///{DB_PATH}",
        )


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings.load()
