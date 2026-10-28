"""Explicitly invoked vision transport; importing this module has no side effects."""

from __future__ import annotations

import os
from pathlib import Path


def create_client(env_file: str | None = None):
    values = {}
    if env_file:
        from dotenv import dotenv_values

        values = dotenv_values(Path(env_file).expanduser(), interpolate=False)
    endpoint = os.environ.get("LLM_BASE_URL") or values.get("LLM_BASE_URL")
    key = os.environ.get("LLM_API_KEY") or values.get("LLM_API_KEY")
    model = os.environ.get("VL_MODEL") or values.get("VL_MODEL") or values.get("LLM_MODEL") or "qwen-vl-photo-context"
    if not endpoint:
        raise ValueError("Set LLM_BASE_URL explicitly or pass --env-file before using --execute")
    from openai import OpenAI

    return OpenAI(base_url=endpoint, api_key=key or "local-only"), model
