from __future__ import annotations

import hashlib
from pathlib import Path

from backend.app.models.local_vision import ModelUnavailable
from backend.app.models.encoder_space import space_id


class Embedder:
    dims: int

    @property
    def encoder_id(self) -> str:
        return getattr(self, "_encoder_id", f"{type(self).__module__}.{type(self).__qualname__}:{self.dims}")

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_image(self, image_path: Path, caption: str | None = None) -> list[float]:
        raise NotImplementedError


class HashEmbedder(Embedder):
    dims = 384
    _encoder_id = "hash:sha256-expand-v1:384"

    def _embed(self, payload: bytes) -> list[float]:
        values: list[float] = []
        seed = payload
        while len(values) < self.dims:
            seed = hashlib.sha256(seed).digest()
            values.extend((byte / 127.5) - 1.0 for byte in seed)
        return values[: self.dims]

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text.encode("utf-8"))

    def embed_image(self, image_path: Path, caption: str | None = None) -> list[float]:
        payload = image_path.read_bytes()[: 1024 * 1024]
        if caption:
            payload += caption.encode("utf-8")
        return self._embed(payload)


class ClipEmbedder(Embedder):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:
            raise ModelUnavailable("Install torch, pillow and transformers to use local CLIP embeddings.") from exc

        self.torch = torch
        self.Image = Image
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.dims = int(self.model.config.projection_dim)
        self._encoder_id = space_id(
            "clip",
            model_name,
            getattr(self.model.config, "_commit_hash", None),
            dimensions=self.dims,
            encoding="clip-processor-text77-image-v1",
        )

    def embed_text(self, text: str) -> list[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(
            self.device
        )
        with self.torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return features[0].detach().cpu().float().tolist()

    def embed_image(self, image_path: Path, caption: str | None = None) -> list[float]:
        del caption
        image = self.Image.open(image_path).convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return features[0].detach().cpu().float().tolist()


class BgeTextEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        try:
            import torch
            import torch.nn.functional as functional
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise ModelUnavailable("Install torch and transformers to use local BGE text embeddings.") from exc

        model_name = _resolve_local_bge_model(model_name)
        self.torch = torch
        self.functional = functional
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.dims = int(getattr(self.model.config, "hidden_size", 0) or 0)
        self._encoder_id = space_id(
            "bge",
            model_name,
            getattr(self.model.config, "_commit_hash", None),
            dimensions=self.dims,
            encoding="mean-pool-mask-l2-text512-v1",
        )

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            features = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            features = self.functional.normalize(features, p=2, dim=1)
        return features.detach().cpu().float().tolist()

    def embed_image(self, image_path: Path, caption: str | None = None) -> list[float]:
        del image_path, caption
        raise ModelUnavailable("BGE is a text embedding model and cannot embed images.")


def _mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _resolve_local_bge_model(model_name: str) -> str:
    if model_name != "BAAI/bge-m3":
        return model_name
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    bge_m3 = cache_root / "models--BAAI--bge-m3"
    if bge_m3.exists():
        return model_name
    bge_small = cache_root / "models--BAAI--bge-small-zh-v1.5"
    if bge_small.exists():
        return "BAAI/bge-small-zh-v1.5"
    return model_name


def build_embedder(kind: str, model_name: str, device: str) -> Embedder:
    if kind == "clip":
        return ClipEmbedder(model_name=model_name, device=device)
    if kind in {"bge", "bge-m3"}:
        return BgeTextEmbedder(model_name=model_name, device=device)
    if kind == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedder: {kind}")
