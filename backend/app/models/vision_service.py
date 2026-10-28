from __future__ import annotations

import base64
import io
import ipaddress
import json
import mimetypes
import urllib.error
import urllib.request
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.models.local_vision import BasicCaptioner, ModelUnavailable, build_captioner, normalize_vision_context
from backend.app.models.photo_context_prompt import PHOTO_CONTEXT_PROMPT


class VisionServiceError(RuntimeError):
    pass


class VisionServiceUnavailable(VisionServiceError):
    pass


class VisionServiceClient:
    def analyze_photo(self, image_path: Path) -> dict[str, object]:
        raise NotImplementedError


class BasicVisionClient(VisionServiceClient):
    def __init__(self) -> None:
        self.captioner = BasicCaptioner()

    def analyze_photo(self, image_path: Path) -> dict[str, object]:
        return self.captioner.caption(image_path)


class TransformersQwenClient(VisionServiceClient):
    def __init__(self, model_name: str, device: str, max_pixels: int) -> None:
        self.captioner = build_captioner("qwen-vl", model_name, device, max_pixels)

    def analyze_photo(self, image_path: Path) -> dict[str, object]:
        return self.captioner.caption(image_path)


class VllmQwenClient(VisionServiceClient):
    def __init__(self, endpoint: str, model: str, timeout_seconds: int = 180, max_pixels: int = 768 * 768) -> None:
        self.endpoint = validate_local_endpoint(endpoint)
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_pixels = max_pixels

    def analyze_photo(self, image_path: Path) -> dict[str, object]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PHOTO_CONTEXT_PROMPT},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_path, self.max_pixels)}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 512,
        }
        request = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _urlopen_no_proxy(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise VisionServiceUnavailable(f"vLLM vision service unavailable: {exc}") from exc
        except TimeoutError as exc:
            raise VisionServiceUnavailable("vLLM vision service timed out") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionServiceError(f"Unexpected vLLM response: {data}") from exc
        return normalize_vision_context(str(content))


class FallbackVisionClient(VisionServiceClient):
    def __init__(self, primary: VisionServiceClient, fallback_factory) -> None:
        self.primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: VisionServiceClient | None = None

    def analyze_photo(self, image_path: Path) -> dict[str, object]:
        try:
            return self.primary.analyze_photo(image_path)
        except VisionServiceUnavailable:
            if self._fallback is None:
                self._fallback = self._fallback_factory()
            return self._fallback.analyze_photo(image_path)


@dataclass(frozen=True)
class VisionClientConfig:
    backend: str
    endpoint: str
    model: str
    timeout_seconds: int
    fallback: str
    transformers_model: str
    device: str
    max_pixels: int


def build_vision_client(config: VisionClientConfig) -> VisionServiceClient:
    backend = config.backend.lower()
    if backend == "basic":
        return BasicVisionClient()
    if backend == "transformers":
        return TransformersQwenClient(config.transformers_model, config.device, config.max_pixels)
    if backend == "vllm":
        return VllmQwenClient(config.endpoint, config.model, config.timeout_seconds, config.max_pixels)
    raise ValueError(f"Unknown vision backend: {config.backend}")


def build_vision_client_with_fallback(config: VisionClientConfig) -> VisionServiceClient:
    primary = build_vision_client(config)
    fallback_factory = _fallback_factory(config)
    if fallback_factory is None:
        return primary
    return FallbackVisionClient(primary, fallback_factory)


def _fallback_factory(config: VisionClientConfig):
    if config.fallback == "none" or config.fallback == config.backend:
        return None
    if config.fallback == "basic":
        return BasicVisionClient
    if config.fallback == "transformers":
        return lambda: TransformersQwenClient(config.transformers_model, config.device, config.max_pixels)
    return None


def vision_status(endpoint: str, model: str, timeout_seconds: int = 5) -> dict[str, Any]:
    try:
        url = f"{validate_local_endpoint(endpoint)}/v1/models"
        with _urlopen_no_proxy(url, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = [item.get("id") for item in data.get("data", []) if isinstance(item, dict)]
        return {
            "backend": "vllm",
            "endpoint": endpoint,
            "model": model,
            "healthy": True,
            "models": models,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "backend": "vllm",
            "endpoint": endpoint,
            "model": model,
            "healthy": False,
            "models": [],
            "error": repr(exc),
        }


def validate_local_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Vision endpoint must be an HTTP(S) loopback URL without credentials")
    if parsed.hostname != "localhost":
        try:
            local = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            local = False
        if not local:
            raise ValueError("Vision endpoint must use localhost or a loopback IP address")
    return endpoint.rstrip("/")


def _image_data_url(image_path: Path, max_pixels: int) -> str:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image.load()
            if image.width * image.height <= max_pixels:
                mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
                encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                return f"data:{mime};base64,{encoded}"

            resized = image.copy()
            resized.thumbnail(_max_size(image.width, image.height, max_pixels))
            if resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=88, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"


def _max_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    scale = (max_pixels / max(1, width * height)) ** 0.5
    return max(1, int(width * scale)), max(1, int(height * scale))


def _urlopen_no_proxy(request: urllib.request.Request | str, timeout: int):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)
