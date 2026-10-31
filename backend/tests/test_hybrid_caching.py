"""Candidate-cache persistence, expiry, eviction and filter isolation."""

from __future__ import annotations

import hashlib

import pytest

from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import caching


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_cache_ttl_is_not_extended_by_reads_and_reopens(tmp_path):
    now = [100.0]
    path = tmp_path / "cache.sqlite"
    cache = caching.CandidateCache(path, ttl_seconds=5, clock=lambda: now[0])
    cache.put("version1", key("query"), [1, 2])
    now[0] += 4
    assert cache.get("version1", key("query")) == (1, 2)
    reopened = caching.CandidateCache(path, ttl_seconds=5, clock=lambda: now[0])
    assert reopened.get("version1", key("query")) == (1, 2)
    assert reopened.get("version2", key("query")) is None
    now[0] += 1
    assert reopened.get("version1", key("query")) is None


def test_lru_and_payload_budget_evict_across_namespaces_without_harming_oversized_entries(tmp_path):
    cache = caching.CandidateCache(tmp_path / "cache.sqlite", max_entries=2, max_bytes=7)
    cache.put("v1", key("first"), [1])
    cache.put("v1", key("second"), [2])
    assert cache.get("v1", key("first")) == (1,)
    cache.put("v2", key("third"), [3])
    assert cache.get("v1", key("second")) is None
    assert cache.get("v1", key("first")) == (1,)
    assert not cache.put("v1", key("oversized"), [10000000])
    assert cache.get("v2", key("third")) == (3,)
    # Fits individually, but forces both previous three-byte payloads out.
    cache.put("v1", key("larger"), [4, 5, 6])
    assert cache.get("v1", key("first")) is None
    assert cache.get("v2", key("third")) is None
    assert cache.get("v1", key("larger")) == (4, 5, 6)


def test_wrapper_keys_include_filters_limits_and_version_and_revalidate_cached_domains(tmp_path):
    calls = []

    class Filtered:
        def retrieve(self, query, limit):
            calls.append((query.text, limit))
            return tuple(identifier for identifier in [1, 2] if query.filters.matches(identifier, {}))[:limit]

    cache = caching.CandidateCache(tmp_path / "cache.sqlite")
    route = caching.CachedRetriever(Filtered(), cache, "bundle1", {1: {}, 2: {}})
    query = hybrid.SearchQuery("query")
    assert route.retrieve(query, 2) == route.retrieve(query, 2) == (1, 2)
    assert len(calls) == 1
    assert route.retrieve(hybrid.SearchQuery("query", hybrid.SearchFilter(allowed_ids=frozenset({2}))), 2) == (2,)
    assert route.retrieve(query, 1) == (1,)
    another_version = caching.CachedRetriever(Filtered(), cache, "bundle2", {1: {}, 2: {}})
    assert another_version.retrieve(query, 2) == (1, 2)
    assert len(calls) == 4
    stale_domain = caching.CachedRetriever(Filtered(), cache, "bundle1", {1: {}})
    with pytest.raises(ValueError, match="domain"):
        stale_domain.retrieve(query, 2)
    assert len(calls) == 4


def test_fresh_filter_violation_and_failure_are_not_cached(tmp_path):
    class Broken:
        def __init__(self):
            self.calls = 0

        def retrieve(self, query, limit):
            self.calls += 1
            if self.calls == 1:
                return [2]
            raise OSError("retriever failed")

    inner = Broken()
    route = caching.CachedRetriever(inner, caching.CandidateCache(tmp_path / "cache.sqlite"), "version", {1: {}, 2: {}})
    query = hybrid.SearchQuery("query", hybrid.SearchFilter(allowed_ids=frozenset({1})))
    with pytest.raises(ValueError, match="filter"):
        route.retrieve(query, 10)
    with pytest.raises(OSError):
        route.retrieve(query, 10)
    assert inner.calls == 2
