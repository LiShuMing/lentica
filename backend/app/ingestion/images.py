from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}


@dataclass(frozen=True)
class ImageMeta:
    path: Path
    sha256: str
    size_bytes: int
    width: int | None = None
    height: int | None = None
    exif_taken_at: datetime | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGES and not path.name.startswith("._"):
            yield path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def inspect_image(path: Path) -> ImageMeta:
    sha = sha256_file(path)
    size = path.stat().st_size
    width = height = None
    taken_at = None
    lat = lon = None

    try:
        from PIL import ExifTags, Image

        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except Exception:
            pass

        with Image.open(path) as img:
            width, height = img.size
            exif = img.getexif()
            if exif:
                tag_names = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
                raw_taken = tag_names.get("DateTimeOriginal") or tag_names.get("DateTime")
                if raw_taken:
                    taken_at = _parse_exif_time(str(raw_taken))
                gps_info = tag_names.get("GPSInfo")
                if gps_info:
                    gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
                    lat = _gps_to_degrees(gps.get("GPSLatitude"), str(gps.get("GPSLatitudeRef", "N")))
                    lon = _gps_to_degrees(gps.get("GPSLongitude"), str(gps.get("GPSLongitudeRef", "E")))
    except Exception:
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
    )


def make_thumbnail(path: Path, output_dir: Path, size: int = 384) -> Path | None:
    try:
        from PIL import Image

        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except Exception:
            pass

        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{sha256_file(path)[:24]}.jpg"
        if out.exists():
            return out
        with Image.open(path) as img:
            img.thumbnail((size, size))
            img.convert("RGB").save(out, "JPEG", quality=86, optimize=True)
        return out
    except Exception:
        return None


def _parse_exif_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _ratio_to_float(value: object) -> float:
    try:
        return float(value)
    except TypeError:
        return value.numerator / value.denominator  # type: ignore[attr-defined]


def _gps_to_degrees(values: object, ref: str) -> float | None:
    try:
        deg, minutes, seconds = values  # type: ignore[misc]
        result = _ratio_to_float(deg) + _ratio_to_float(minutes) / 60 + _ratio_to_float(seconds) / 3600
        if ref in {"S", "W"}:
            result *= -1
        return result
    except Exception:
        return None
