"""Resumed pair inference is identity-bound and corruption fails closed."""

from __future__ import annotations

import hashlib
import json

import pytest

from backend.app.evaluation.hybrid_eval import pair_checkpoints


def test_pair_scores_reopen_and_reject_source_change_overwrite_and_corruption(tmp_path):
    path = tmp_path / "scores.jsonl"
    key = hashlib.sha256(b"query/candidate-list").hexdigest()
    first = pair_checkpoints.PairScoreCache(path, {"model": "a", "source": "b"})
    first.put(key, (1.0, -2.0))
    reloaded = pair_checkpoints.PairScoreCache(path, {"model": "a", "source": "b"})
    assert reloaded.get(key, 2) == (1.0, -2.0)
    with pytest.raises(ValueError, match="count"):
        reloaded.get(key, 3)
    with pytest.raises(ValueError, match="overwrite"):
        reloaded.put(key, (9.0, 1.0))
    with pytest.raises(ValueError, match="identity"):
        pair_checkpoints.PairScoreCache(path, {"model": "changed", "source": "b"})
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["scores"][0] = 99.0
    path.write_text(lines[0] + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="digest"):
        pair_checkpoints.PairScoreCache(path, {"model": "a", "source": "b"})
