from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationCase:
    query_id: str
    query: str
    relevant_asset_ids: set[int]
    category: str


def load_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        payload = json.loads(raw)
        try:
            cases.append(
                EvaluationCase(
                    query_id=str(payload["query_id"]),
                    query=str(payload["query"]),
                    relevant_asset_ids={int(value) for value in payload["relevant_asset_ids"]},
                    category=str(payload.get("category") or "uncategorized"),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid evaluation row {line_number} in {path}") from exc
    return cases
