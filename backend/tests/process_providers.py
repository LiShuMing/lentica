"""Small spawn-safe providers for actual process lifecycle acceptance checks."""

from __future__ import annotations

import time
from pathlib import Path

from backend.app.retrieval.hybrid import execution


class DelayedProvider:
    """Return explicit IDs after optional delay, recording only an owned test PID."""

    def __init__(self, *, ids=None, delay=0.0, pid_file=None, failure=False, invalid_filter=False):
        self.ids, self.delay, self.pid_file = ids or [], delay, pid_file
        self.failure, self.invalid_filter = failure, invalid_filter

    def retrieve(self, query, limit):
        import os

        if self.pid_file is not None:
            Path(self.pid_file).write_text(str(os.getpid()))
        time.sleep(self.delay)
        if self.invalid_filter:
            raise execution.FilterContractError("Unsupported exact-match filter")
        if self.failure:
            raise OSError("provider failure")
        return self.ids[:limit]


class StatefulProvider:
    """Expose per-instance call state to prove reuse/reconstruction with real PIDs."""

    def __init__(self, *, delay_query=None):
        self.calls = 0
        self.delay_query = delay_query

    def retrieve(self, query, limit):
        self.calls += 1
        if query.text == self.delay_query:
            time.sleep(10)
        return [self.calls][:limit]


class DelayedStore:
    """Record a final batch without relying on shared parent-process state."""

    def __init__(self, *, call_file=None, pid_file=None, delay=0.0, init_delay=0.0, missing=False, oversized=False):
        import os

        self.call_file, self.delay = call_file, delay
        self.missing, self.oversized = missing, oversized
        if pid_file is not None:
            Path(pid_file).write_text(str(os.getpid()))
        time.sleep(init_delay)

    def fetch(self, ids):
        import json

        if self.call_file is not None:
            Path(self.call_file).write_text(json.dumps(list(ids)))
        time.sleep(self.delay)
        if self.missing:
            return {}
        return dict.fromkeys(ids, "x" * (execution.MAX_FINAL_BYTES + 1) if self.oversized else "record")


class DelayedFusion:
    """Deliberately stall fusion to test the final execution boundary."""

    def __init__(self, *, pid_file, delay=10):
        import os

        Path(pid_file).write_text(str(os.getpid()))
        self.delay = delay

    def fuse(self, rankings, config):
        time.sleep(self.delay)
        return ()
