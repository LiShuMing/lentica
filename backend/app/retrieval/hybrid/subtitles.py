"""Explicit bounded SRT/WebVTT cue ingestion, separate from model transcription."""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from . import artifacts, contracts, units

_TIME = re.compile(r"(?:(\d{2,}):)?(\d{2}):(\d{2})[.,](\d{3})")


@dataclass(frozen=True)
class SubtitleCue:
    """One original cue; supplied subtitles are not automatically verified speech."""

    identifier: str
    start_seconds: float
    end_seconds: float
    text: str


def _seconds(value: str) -> float:
    match = _TIME.fullmatch(value.strip())
    if match is None:
        raise ValueError("Subtitle timestamps require HH:MM:SS.mmm or MM:SS.mmm")
    hours, minutes, seconds, milliseconds = match.groups()
    if int(minutes) >= 60 or int(seconds) >= 60:
        raise ValueError("Subtitle minute/second component is out of range")
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def parse(text: str, *, format: str, max_cues: int = 5000, max_seconds: int = 3600) -> tuple[SubtitleCue, ...]:
    """Parse local relative-time cues; reject unsupported global timestamp maps.

    WebVTT cue positions/style/notes do not become spatial locators. Cue markup
    is converted to plain text; timing, cue identity and source order remain.
    Overlapping cues are valid, decreasing starts or incomplete cues are errors.
    """
    if (
        format not in {"srt", "vtt"}
        or not isinstance(text, str)
        or len(text.encode("utf-8")) > 1024**2
        or type(max_cues) is not int
        or not 1 <= max_cues <= 5000
        or type(max_seconds) is not int
        or not 1 <= max_seconds <= 86400
    ):
        raise ValueError("Subtitle format/file/cue/duration budget is invalid")
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if "X-TIMESTAMP-MAP" in text:
        raise ValueError("External timestamp maps require explicit media alignment")
    blocks = re.split(r"\n[ \t]*\n", text)
    if format == "vtt":
        header = blocks[0].splitlines() if blocks else []
        if not header or not re.fullmatch(r"WEBVTT(?:[ \t].*)?", header[0]) or "-->" in blocks[0]:
            raise ValueError("WebVTT header is missing")
        blocks = blocks[1:]
    result = []
    for block in blocks:
        lines = block.splitlines()
        if format == "vtt" and lines and (lines[0].startswith("NOTE") or lines[0] in {"STYLE", "REGION"}):
            continue
        if not lines:
            continue
        timing_index = 0 if "-->" in lines[0] else 1
        if timing_index >= len(lines) or "-->" not in lines[timing_index] or len(lines) <= timing_index + 1:
            raise ValueError("Subtitle cue needs timestamps and nonempty text")
        begin, end = lines[timing_index].split("-->", 1)
        start_seconds, end_seconds = _seconds(begin), _seconds(end.strip().split()[0])
        if not 0 <= start_seconds < end_seconds <= max_seconds or result and start_seconds < result[-1].start_seconds:
            raise ValueError("Subtitle time order or duration budget is invalid")
        cue_text = html.unescape(re.sub(r"<[^<>]{0,256}>", "", "\n".join(lines[timing_index + 1 :]))).strip()
        if not cue_text or len(cue_text) > 8000 or len(result) >= max_cues:
            raise ValueError("Subtitle cue text/count budget exceeded")
        result.append(
            SubtitleCue(lines[0] if timing_index else str(len(result) + 1), start_seconds, end_seconds, cue_text)
        )
    return tuple(result)


def extract_units(
    path: Path,
    *,
    format: str,
    asset_id: int,
    first_unit_id: int,
    max_cues: int = 5000,
    max_seconds: int = 3600,
    metadata_values: Mapping[str, str | int | float | bool | None] | None = None,
) -> tuple[units.RetrievalUnit, ...]:
    """Read only explicit subtitle files and inherit resource filters to cues."""
    contracts.asset_id(asset_id)
    contracts.asset_id(first_unit_id)
    if Path(path).stat().st_size > 1024**2:
        raise ValueError("Subtitle file exceeds1MiB")
    raw = Path(path).read_bytes()
    cues = parse(raw.decode("utf-8-sig"), format=format, max_cues=max_cues, max_seconds=max_seconds)
    version = artifacts.space_id(
        "subtitle-cues-v1",
        {"source": hashlib.sha256(raw).hexdigest()},
        {"format": format, "max_cues": max_cues, "max_seconds": max_seconds},
    )
    return tuple(
        units.RetrievalUnit(
            first_unit_id + index,
            asset_id,
            "video",
            units.Locator(start_seconds=cue.start_seconds, end_seconds=cue.end_seconds),
            (units.TextEvidence("transcript", cue.text, f"provided-subtitle-{format}", version),),
            metadata_values or {},
        )
        for index, cue in enumerate(cues)
    )
