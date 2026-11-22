"""Image discovery, hashing, EXIF/GPS inspection, and orientation-aware thumbnails."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from backend.app.ingestion.traversal import iter_source_files


SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
_ORIENTATION_TAG = 274
_SWAPPED_ORIENTATIONS = {5, 6, 7, 8}


@dataclass(frozen=True)
class ImageMeta:
    """Deterministic file and EXIF metadata extracted from a local image."""

    path: Path
    sha256: str
    size_bytes: int
    width: int | None = None
    height: int | None = None
    exif_taken_at: datetime | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    orientation: int = 1


def iter_image_files(root: Path) -> Iterable[Path]:
    yield from iter_source_files(root, SUPPORTED_IMAGES)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _register_heif_opener() -> None:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except Exception:  # noqa: BLE001
        pass


def open_oriented_image(path: Path):
    """Open an image and return a copy transposed to its EXIF display orientation."""
    from PIL import Image, ImageOps

    _register_heif_opener()
    with Image.open(path) as raw:
        raw.load()
        oriented = ImageOps.exif_transpose(raw)
        if oriented is raw:
            return raw.copy()
        return oriented


def inspect_image(path: Path) -> ImageMeta:
    """Inspect hash, size, display dimensions, nested EXIF time, and signed GPS."""
    sha = sha256_file(path)
    size = path.stat().st_size
    width: int | None = None
    height: int | None = None
    taken_at: datetime | None = None
    lat: float | None = None
    lon: float | None = None
    orientation = 1

    try:
        from PIL import ExifTags, Image

        _register_heif_opener()
        with Image.open(path) as img:
            width, height = img.size
            exif = img.getexif()
            if exif:
                orientation = _extract_orientation(exif)
                if orientation in _SWAPPED_ORIENTATIONS and width is not None and height is not None:
                    width, height = height, width
                taken_at = _extract_taken_at(exif, ExifTags)
                lat, lon = _extract_gps(exif, ExifTags)
    except Exception:  # noqa: BLE001
        pass

    return ImageMeta(
        path=path,
        sha256=sha,
        size_bytes=size,
        width=width,
        height=height,
        exif_taken_at=taken_at,
        gps_lat=lat,
        gps_lon=lon,
        orientation=orientation,
    )


def make_thumbnail(path: Path, output_dir: Path, size: int = 384) -> Path | None:
    """Create a cached JPEG thumbnail transposed to the image's EXIF orientation."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{sha256_file(path)[:24]}.jpg"
        if out.exists():
            return out
        with open_oriented_image(path) as img:
            img.thumbnail((size, size))
            img.convert("RGB").save(out, "JPEG", quality=86, optimize=True)
        return out
    except Exception:  # noqa: BLE001
        return None


def _extract_orientation(exif) -> int:
    try:
        value = int(exif.get(_ORIENTATION_TAG, 1))
        return value if 1 <= value <= 8 else 1
    except (TypeError, ValueError):
        return 1


def _extract_taken_at(exif, exif_tags) -> datetime | None:
    nested: dict[str, object] = {}
    try:
        exif_ifd = exif.get_ifd(exif_tags.IFD.Exif)
        if isinstance(exif_ifd, dict):
            nested = {str(exif_tags.TAGS.get(key, key)): value for key, value in exif_ifd.items()}
    except Exception:  # noqa: BLE001
        nested = {}

    top_level: dict[str, object] = {}
    try:
        top_level = {str(exif_tags.TAGS.get(key, key)): value for key, value in exif.items()}
    except Exception:  # noqa: BLE001
        top_level = {}

    for raw in (
        nested.get("DateTimeOriginal"),
        nested.get("DateTimeDigitized"),
        top_level.get("DateTimeOriginal"),
        top_level.get("DateTime"),
    ):
        if raw is None:
            continue
        parsed = _parse_exif_time(str(raw).strip().rstrip("\x00"))
        if parsed is not None:
            return parsed
    return None


def _extract_gps(exif, exif_tags) -> tuple[float | None, float | None]:
    gps_raw: object = None
    try:
        gps_raw = exif.get_ifd(exif_tags.IFD.GPSInfo)
    except Exception:  # noqa: BLE001
        gps_raw = None

    if not isinstance(gps_raw, dict) or not gps_raw:
        try:
            top_gps = exif.get(exif_tags.IFD.GPSInfo)
            if isinstance(top_gps, dict):
                gps_raw = top_gps
        except Exception:  # noqa: BLE001
            gps_raw = None

    if not isinstance(gps_raw, dict) or not gps_raw:
        return None, None

    gps = {str(exif_tags.GPSTAGS.get(key, key)): value for key, value in gps_raw.items()}
    lat = _gps_to_degrees(gps.get("GPSLatitude"), _decode_ref(gps.get("GPSLatitudeRef"), "N"))
    lon = _gps_to_degrees(gps.get("GPSLongitude"), _decode_ref(gps.get("GPSLongitudeRef"), "E"))
    return lat, lon


def _decode_ref(value: object, default: str) -> str:
    if isinstance(value, bytes):
        text = value.decode("ascii", errors="ignore")
    elif value is None:
        text = default
    else:
        text = str(value)
    cleaned = text.strip().rstrip("\x00").upper()
    return cleaned or default


def _parse_exif_time(value: str) -> datetime | None:
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _ratio_to_float(value: object) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        if float(den) == 0.0:
            raise ValueError("Zero denominator in EXIF rational")
        return float(num) / float(den)
    try:
        return float(value)
    except TypeError:
        den = value.denominator  # type: ignore[attr-defined]
        if den == 0:
            raise ValueError("Zero denominator in EXIF rational")
        return value.numerator / den  # type: ignore[attr-defined]


def _gps_to_degrees(values: object, ref: str) -> float | None:
    try:
        deg, minutes, seconds = values  # type: ignore[misc]
        result = _ratio_to_float(deg) + _ratio_to_float(minutes) / 60.0 + _ratio_to_float(seconds) / 3600.0
        if ref in {"S", "W"}:
            result = -abs(result)
        return result
    except Exception:  # noqa: BLE001
        return None
