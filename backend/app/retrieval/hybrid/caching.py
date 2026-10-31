"""Opt-in bounded local candidate caching; final resource records are never cached."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from . import contracts, fusion, snapshots


class CandidateCache:
    """Transactional SQLite TTL/LRU cache with explicit row and payload budgets.

    Each operation owns a short-lived connection. The byte budget counts stored
    UTF-8 ID payloads, not SQLite page/free-list overhead or the filesystem size.
    An expired entry is a miss; accessing it does not extend its lifetime.
    """

    def __init__(
        self,
        path: Path,
        *,
        ttl_seconds: float = 300,
        max_entries: int = 1024,
        max_bytes: int = 4 * 1024**2,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not math.isfinite(ttl_seconds) or not 0 < ttl_seconds <= 86400:
            raise ValueError("Cache TTL must be positive and at most one day")
        if (
            type(max_entries) is not int
            or not 1 <= max_entries <= 100_000
            or type(max_bytes) is not int
            or not 1 <= max_bytes <= 64 * 1024**2
        ):
            raise ValueError("Cache row/byte budgets are outside the supported range")
        self.path = Path(path)
        self.ttl_seconds, self.max_entries, self.max_bytes, self.clock = ttl_seconds, max_entries, max_bytes, clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS info(version TEXT NOT NULL, tick INTEGER NOT NULL)")
            row = connection.execute("SELECT version FROM info").fetchall()
            if not row:
                connection.execute("INSERT INTO info VALUES ('lentica-candidate-cache-v1', 0)")
            elif row != [("lentica-candidate-cache-v1",)]:
                raise ValueError("Unsupported candidate cache version")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entries(namespace TEXT, key TEXT, payload TEXT, bytes INTEGER, expires REAL, accessed INTEGER, PRIMARY KEY(namespace,key))"
            )

    @contextlib.contextmanager
    def _connect(self):
        with contextlib.closing(sqlite3.connect(self.path, timeout=1)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            yield connection

    @staticmethod
    def _tick(connection) -> int:
        connection.execute("UPDATE info SET tick=tick+1")
        return connection.execute("SELECT tick FROM info").fetchone()[0]

    @staticmethod
    def _validate_key(namespace: str, key: str) -> None:
        if (
            not isinstance(namespace, str)
            or not 1 <= len(namespace) <= 512
            or not isinstance(key, str)
            or len(key) != 64
            or any(char not in "0123456789abcdef" for char in key)
        ):
            raise ValueError("Cache requires an explicit namespace and SHA-256 request key")

    def get(self, namespace: str, key: str) -> tuple[int, ...] | None:
        """Return unexpired ordered IDs, touching LRU but never extending TTL."""
        self._validate_key(namespace, key)
        with self._connect() as connection:
            connection.execute("DELETE FROM entries WHERE expires<=?", (self.clock(),))
            row = connection.execute(
                "SELECT payload FROM entries WHERE namespace=? AND key=?", (namespace, key)
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE entries SET accessed=? WHERE namespace=? AND key=?", (self._tick(connection), namespace, key)
            )
            values = json.loads(row[0])
            if not isinstance(values, list) or len(values) > contracts.MAX_WINDOW:
                raise ValueError("Invalid cached candidate payload")
            return tuple(contracts.asset_id(value) for value in values)

    def put(self, namespace: str, key: str, ids: Sequence[int]) -> bool:
        """Store successful ordered IDs; evict expired/least-recently-used entries.

        Returns False for an entry larger than the entire payload budget, without
        evicting useful entries to make room for an entry that cannot fit.
        """
        self._validate_key(namespace, key)
        if len(ids) > contracts.MAX_WINDOW:
            raise ValueError("Candidate cache window exceeds its response bound")
        payload = snapshots.canonical([contracts.asset_id(value) for value in ids])
        size = len(payload.encode("utf-8"))
        if size > self.max_bytes:
            return False
        with self._connect() as connection:
            now = self.clock()
            connection.execute("DELETE FROM entries WHERE expires<=?", (now,))
            connection.execute(
                "INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?)",
                (namespace, key, payload, size, now + self.ttl_seconds, self._tick(connection)),
            )
            count, total = connection.execute("SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM entries").fetchone()
            while count > self.max_entries or total > self.max_bytes:
                namespace_, key_, bytes_ = connection.execute(
                    "SELECT namespace,key,bytes FROM entries ORDER BY accessed LIMIT 1"
                ).fetchone()
                connection.execute("DELETE FROM entries WHERE namespace=? AND key=?", (namespace_, key_))
                count, total = count - 1, total - bytes_
        return True


class CachedRetriever:
    """Cache successful ID windows under an explicit immutable index namespace.

    Use bundles.cache_namespace or an equally complete source/model/config/index
    identity. The wrapped retriever is still responsible for pre-window filtering;
    both cached and fresh results are checked against the full metadata domain.
    Cache I/O failures propagate; no hidden unfiltered fallback is permitted.
    """

    def __init__(
        self,
        retriever: contracts.Retriever,
        cache: CandidateCache,
        namespace: str,
        metadata: Mapping[int, Mapping[str, object]],
    ) -> None:
        self.retriever, self.cache, self.namespace = retriever, cache, namespace
        self.metadata = {contracts.asset_id(key): dict(value) for key, value in metadata.items()}

    def retrieve(self, query: contracts.SearchQuery, limit: int) -> tuple[int, ...]:
        """Respect filters/window on hits and misses; never cache a failed call."""
        if type(limit) is not int or not 0 <= limit <= contracts.MAX_WINDOW:
            raise ValueError("Candidate cache limit is outside its window budget")
        if not limit:
            return ()
        key = hashlib.sha256(
            snapshots.canonical(
                {
                    "text": query.text,
                    "equals": dict(query.filters.equals),
                    "allowed_ids": None if query.filters.allowed_ids is None else sorted(query.filters.allowed_ids),
                    "limit": limit,
                }
            ).encode()
        ).hexdigest()
        ids = self.cache.get(self.namespace, key)
        cached = ids is not None
        if ids is None:
            ids = fusion.unique_ids(self.retriever.retrieve(query, limit), limit)
        if (
            len(ids) > limit
            or len(set(ids)) != len(ids)
            or any(key not in self.metadata or not query.filters.matches(key, self.metadata[key]) for key in ids)
        ):
            raise ValueError("Cached or fresh retriever violated the candidate/filter domain")
        if not cached:
            self.cache.put(self.namespace, key, ids)
        return ids
