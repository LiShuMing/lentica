from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from backend.app.db.models import File, GeocodeCache, PhotoMetadataCandidate


DATE_PATTERNS = [
    re.compile(r"(?P<year>20\d{2}|19\d{2})[-_.年]?(?P<month>0[1-9]|1[0-2])[-_.月]?(?P<day>0[1-9]|[12]\d|3[01])"),
    re.compile(r"(?P<year>20\d{2}|19\d{2})[-_.年](?P<month>0?[1-9]|1[0-2])"),
]

LOCATION_WORD_RE = re.compile(r"[\u4e00-\u9fff]{2,12}")
GENERIC_PATH_WORDS = {
    "照片",
    "图片",
    "下载",
    "用户",
    "相册",
    "导出",
    "微信",
    "截图",
    "收藏",
    "视频",
    "文件",
    "icloud",
}


@dataclass(frozen=True)
class MetadataCandidate:
    kind: str
    value: str
    source: str
    confidence: str


def upsert_photo_metadata_candidates(db: Session, file: File, image_meta, context: dict[str, object]) -> list[MetadataCandidate]:
    candidates = extract_photo_metadata_candidates(file, image_meta, context)
    db.execute(delete(PhotoMetadataCandidate).where(PhotoMetadataCandidate.file_id == file.id))
    for candidate in candidates:
        db.add(
            PhotoMetadataCandidate(
                file_id=file.id,
                kind=candidate.kind,
                value=candidate.value,
                source=candidate.source,
                confidence=candidate.confidence,
            )
        )
    return candidates


def extract_photo_metadata_candidates(file: File, image_meta, context: dict[str, object]) -> list[MetadataCandidate]:
    candidates: list[MetadataCandidate] = []
    if image_meta.exif_taken_at:
        candidates.append(
            MetadataCandidate("date", image_meta.exif_taken_at.strftime("%Y-%m-%d"), "exif_time", "confirmed")
        )
        candidates.append(MetadataCandidate("year", image_meta.exif_taken_at.strftime("%Y"), "exif_time", "confirmed"))
    if image_meta.gps_lat is not None and image_meta.gps_lon is not None:
        gps = f"{image_meta.gps_lat:.6f},{image_meta.gps_lon:.6f}"
        candidates.append(MetadataCandidate("location", gps, "exif_gps", "confirmed"))

    path = Path(file.canonical_path)
    for value, source in _date_candidates_from_path(path):
        candidates.append(MetadataCandidate("date", value, source, "candidate"))
        candidates.append(MetadataCandidate("year", value[:4], source, "candidate"))
    for value in _location_candidates_from_path(path):
        candidates.append(MetadataCandidate("location", value, "path", "candidate"))

    for value in _as_text_list(context.get("location_clues")):
        candidates.append(MetadataCandidate("location", value, "vision_location_clue", "inferred"))
    location = str(context.get("location") or "").strip()
    if location:
        candidates.append(MetadataCandidate("location", location, "vision_location", "inferred"))
    for value in _as_text_list(context.get("time_clues")):
        candidates.append(MetadataCandidate("time", value, "vision_time_clue", "inferred"))
    time_value = str(context.get("time") or "").strip()
    if time_value:
        candidates.append(MetadataCandidate("time", time_value, "vision_time", "inferred"))

    return _dedupe_candidates(candidates)


def ensure_local_geocode_cache(db: Session, lat: str | None, lon: str | None) -> str | None:
    if not lat or not lon:
        return None
    try:
        lat_value = float(lat)
        lon_value = float(lon)
    except ValueError:
        return None
    lat_key = f"{lat_value:.3f}"
    lon_key = f"{lon_value:.3f}"
    display = f"GPS {lat_value:.5f}, {lon_value:.5f}"
    existing = (
        db.query(GeocodeCache)
        .filter(GeocodeCache.lat_key == lat_key, GeocodeCache.lon_key == lon_key)
        .one_or_none()
    )
    if existing:
        return existing.display_name or display
    db.add(GeocodeCache(lat_key=lat_key, lon_key=lon_key, display_name=display, provider="local"))
    return display


def candidates_text(candidates: list[MetadataCandidate]) -> str:
    return " ".join(f"{item.kind}:{item.value} {item.source}:{item.confidence}" for item in candidates)


def _date_candidates_from_path(path: Path) -> list[tuple[str, str]]:
    text = " ".join([path.name, *path.parts[-5:]])
    values: list[tuple[str, str]] = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            year = int(match.group("year"))
            month = int(match.group("month"))
            day = int(match.groupdict().get("day") or 1)
            try:
                parsed = datetime(year, month, day)
            except ValueError:
                continue
            values.append((parsed.strftime("%Y-%m-%d"), "filename_or_path"))
    return values


def _location_candidates_from_path(path: Path) -> list[str]:
    values: list[str] = []
    for part in path.parts[-6:-1]:
        lower = part.lower()
        if any(generic in lower for generic in GENERIC_PATH_WORDS):
            continue
        for match in LOCATION_WORD_RE.findall(part):
            if match.lower() not in GENERIC_PATH_WORDS:
                values.append(match)
    return values[:8]


def _dedupe_candidates(candidates: list[MetadataCandidate]) -> list[MetadataCandidate]:
    seen: set[tuple[str, str, str]] = set()
    result: list[MetadataCandidate] = []
    for candidate in candidates:
        value = candidate.value.strip()
        if not value:
            continue
        key = (candidate.kind, value.lower(), candidate.source)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result[:64]


def _as_text_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []
