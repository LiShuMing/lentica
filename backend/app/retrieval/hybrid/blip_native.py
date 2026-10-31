"""Verified local CPU BLIP image/text matching with actual cross attention."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

from . import artifacts


class BlipPairScorer:
    """Native ITM logits; reuse vision only within one explicit bounded call.

    The selected publisher provides a bin checkpoint. It is accepted only after
    artifact verification, with Torch>=2.6 and explicit weights_only=True. No
    unrestricted pickle, remote code, downloads or GPU device selection occurs.
    Instances are intended for serial calls within their owning worker.
    """

    def __init__(
        self, directory: Path, *, threads: int = 2, max_pairs: int = 4, max_tokens: int = 128, image_size: int = 384
    ) -> None:
        """Load explicit local native artifacts; fail before inference on budgets.

        Args:
            directory: Caller-selected directory with verified acquisition manifest.
            threads: CPU worker thread budget; recorded in scorer identity.
            max_pairs: Maximum aligned pairs in one call, including repeated images.
            max_tokens: Native token budget; excess text fails without truncation.
            image_size: Explicit square model input. The original384 preserves the
                publisher processor and existing identity. Smaller fixed profiles
                use public native position interpolation and distinct identities;
                accepting a size is not proof of its retrieval accuracy.

        Raises:
            ValueError: Invalid budgets, mismatched artifacts or unsupported runtime.
        """
        if (
            type(threads) is not int
            or not 1 <= threads <= 32
            or type(max_pairs) is not int
            or not 1 <= max_pairs <= 4
            or type(max_tokens) is not int
            or not 2 <= max_tokens <= 128
            or type(image_size) is not int
            or image_size not in (224, 256, 288, 320, 384)
        ):
            raise ValueError("BLIP CPU/pair/token budgets differ from bounded native profile")
        names = (
            "config.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "pytorch_model.bin",
        )
        files = artifacts.verify(directory, names)
        config = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
        if (
            config.get("architectures") != ["BlipForImageTextRetrieval"]
            or config.get("model_type") != "blip"
            or config["vision_config"]["image_size"] != 384
            or config["vision_config"]["hidden_size"] != 768
            or config["text_config"]["max_position_embeddings"] < max_tokens
        ):
            raise ValueError("Unsupported BLIP native image/text matching configuration")
        import torch
        from transformers import BertTokenizerFast, BlipForImageTextRetrieval, BlipImageProcessor

        if tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
            raise ValueError("Verified BLIP bin loading requires Torch>=2.6 weights_only")
        if tuple(int(part) for part in metadata.version("transformers").split(".")[:2]) < (4, 57):
            raise ValueError("BLIP native profile requires Transformers>=4.57")
        torch.set_num_threads(threads)
        self.tokenizer = BertTokenizerFast.from_pretrained(str(directory), local_files_only=True)
        self.processor = BlipImageProcessor.from_pretrained(str(directory), local_files_only=True)
        model, loading = BlipForImageTextRetrieval.from_pretrained(
            str(directory),
            local_files_only=True,
            use_safetensors=False,
            weights_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("BLIP checkpoint does not exactly match the native model state")
        self.model = model.to("cpu").eval()
        self.max_pairs, self.max_tokens = max_pairs, max_tokens
        self.image_size = image_size
        self.last_unique_images = 0
        self.encoding = {
            "model": "Salesforce/blip-itm-base-coco",
            "head": "native-itm-positive-minus-negative-logit",
            "vision_reuse": "unique-resolved-path-within-call-only",
            "text": "native-bert-batch-padding-no-truncation",
            "max_tokens": max_tokens,
            "max_pairs": max_pairs,
            "max_image_pixels": 16_000_000,
            "max_image_bytes": 4 * 1024**2,
            "weights_only": True,
            "attention": "eager",
            "dtype": "float32",
            "device": "cpu",
            "threads": threads,
            **{name: metadata.version(name) for name in ("torch", "transformers", "Pillow")},
        }
        if image_size != 384:
            self.encoding.update(image_size=image_size, interpolate_pos_encoding=True)
        self.scorer_id = artifacts.space_id("blip-native-itm-v1", files, self.encoding)

    def _inputs(self, texts: Sequence[str], paths: Sequence[Path]):
        """Prepare only explicit bytes/text; reject before allocation/inference when possible."""
        if (
            isinstance(texts, str)
            or isinstance(paths, (str, Path))
            or len(texts) != len(paths)
            or len(texts) > self.max_pairs
            or any(not isinstance(text, str) or not text.strip() or len(text) > 4096 for text in texts)
        ):
            raise ValueError("BLIP requires aligned bounded nonempty text/image pairs")
        tokens = self.tokenizer(list(texts), padding=True, truncation=False, return_tensors="pt")
        if tokens["input_ids"].shape[1] > self.max_tokens:
            raise ValueError("BLIP text exceeds token budget; no silent truncation")
        unique = tuple(dict.fromkeys(Path(path).resolve() for path in paths))
        positions = [unique.index(Path(path).resolve()) for path in paths]
        from PIL import Image

        images = []
        try:
            for path in unique:
                if path.stat().st_size > 4 * 1024**2:
                    raise ValueError("BLIP image exceeds byte budget")
                with Image.open(path) as image:
                    if image.width * image.height > 16_000_000:
                        raise ValueError("BLIP image exceeds pixel budget")
                    images.append(image.convert("RGB"))
            options = {"size": {"height": self.image_size, "width": self.image_size}} if self.image_size != 384 else {}
            pixels = self.processor(images=images, return_tensors="pt", **options)["pixel_values"]
        finally:
            for image in images:
                image.close()
        return pixels, tokens, positions

    def score_pairs(self, texts: Sequence[str], paths: Sequence[Path]) -> tuple[float, ...]:
        """Run unique vision features and genuine pair-conditioned text cross attention.

        Scores are ITM logit differences, not calibrated probabilities or native
        fusion inputs. No features survive the call as a query/gallery cache.
        """
        if (
            not isinstance(texts, Sequence)
            or isinstance(texts, (str, bytes))
            or not isinstance(paths, Sequence)
            or isinstance(paths, (str, bytes))
        ):
            raise ValueError("BLIP requires aligned bounded nonempty text/image pairs")
        if not texts and not paths:
            self.last_unique_images = 0
            return ()
        pixels, tokens, positions = self._inputs(texts, paths)
        import torch

        with torch.inference_mode():
            vision = self.model.vision_model(
                pixel_values=pixels, interpolate_pos_encoding=self.image_size != 384
            ).last_hidden_state
            aligned = vision.index_select(0, torch.tensor(positions, dtype=torch.long))
            encoded = self.model.text_encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                encoder_hidden_states=aligned,
                encoder_attention_mask=torch.ones(aligned.shape[:-1], dtype=torch.long),
            ).last_hidden_state
            logits = self.model.itm_head(encoded[:, 0, :])
        if tuple(logits.shape) != (len(texts), 2):
            raise ValueError("BLIP returned misaligned native ITM logits")
        scores = tuple(float(value) for value in (logits[:, 1] - logits[:, 0]).tolist())
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("BLIP returned nonfinite matching scores")
        self.last_unique_images = len(pixels)
        return scores

    def verify_native_equivalence(self, texts: Sequence[str], paths: Sequence[Path]) -> dict[str, object]:
        """Actually compare shared vision with unmodified native full-pair forward.

        Both computations are fresh inference, outside benchmark query costs.
        The tolerance permits ordinary float32 batching roundoff only.
        """
        if not texts:
            raise ValueError("Native equivalence needs explicit pairs")
        shared = self.score_pairs(texts, paths)
        pixels, tokens, positions = self._inputs(texts, paths)
        import torch

        with torch.inference_mode():
            logits = self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                pixel_values=pixels.index_select(0, torch.tensor(positions, dtype=torch.long)),
                use_itm_head=True,
                interpolate_pos_encoding=self.image_size != 384,
            ).itm_score
        native = tuple(float(value) for value in (logits[:, 1] - logits[:, 0]).tolist())
        error = max(abs(first - second) for first, second in zip(shared, native))
        if len(native) != len(shared) or not math.isfinite(error) or error > 1e-4:
            raise ValueError("Shared BLIP vision differs from native full-pair forward")
        return {"shared_scores": shared, "native_scores": native, "max_absolute_error": error, "pairs": len(texts)}
