from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.app.core.config import env_bool

class FaceRuntimeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FaceObservation:
    bbox: list[float]
    embedding: list[float]
    score: float


class InsightFaceDetector:
    def __init__(self, device_id: int = 0, det_size: tuple[int, int] = (640, 640)) -> None:
        if not env_bool("LENTICA_ENABLE_FACES", False):
            raise FaceRuntimeUnavailable("Face indexing is opt-in. Set LENTICA_ENABLE_FACES=true after reviewing model terms.")
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:  # noqa: BLE001
            raise FaceRuntimeUnavailable("Install insightface and onnxruntime-gpu to enable face indexing.") from exc

        providers = _available_providers()
        ctx_id = device_id if "CUDAExecutionProvider" in providers else -1
        self.providers = providers
        self.device = "cuda" if ctx_id >= 0 else "cpu"
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, image_path: Path) -> list[FaceObservation]:
        try:
            import numpy as np
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            raise FaceRuntimeUnavailable("Install numpy and pillow to enable face indexing.") from exc

        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except Exception:
            pass

        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb)[:, :, ::-1]
        faces = self.app.get(array)
        observations: list[FaceObservation] = []
        for face in faces:
            embedding = [float(value) for value in getattr(face, "embedding", [])]
            if not embedding:
                continue
            observations.append(
                FaceObservation(
                    bbox=[float(value) for value in face.bbox],
                    embedding=embedding,
                    score=float(getattr(face, "det_score", 0.0)),
                )
            )
        return observations


def _available_providers() -> list[str]:
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        available = set()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    raise FaceRuntimeUnavailable("onnxruntime has no usable execution provider for face indexing.")
