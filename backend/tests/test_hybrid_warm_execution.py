"""Real warm-process reuse, deadline replacement, failure cleanup and ownership."""

from __future__ import annotations

import os

import pytest

from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import execution, warm_execution


def make_searcher():
    spec = execution.ProviderSpec("backend.tests.process_providers", "StatefulProvider", {"delay_query": "stall"})
    return warm_execution.WarmProcessSearch({"route": spec}, {1: {}, 2: {}, 3: {}})


def assert_reaped(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_captured_factory_map_cannot_silently_retarget_a_warm_route():
    original = execution.ProviderSpec("backend.tests.process_providers", "StatefulProvider")
    mapping = {"route": original}
    searcher = warm_execution.WarmProcessSearch(mapping, {1: {}})
    replacement = execution.ProviderSpec(
        "backend.tests.process_providers", "StatefulProvider", {"delay_query": "other"}
    )
    mapping["route"] = replacement
    assert searcher.providers["route"] is original
    with pytest.raises(TypeError):
        searcher.providers["route"] = replacement
    with pytest.raises(AttributeError):
        searcher.providers = mapping


def test_state_and_pid_are_reused_then_reaped_on_context_exit():
    config = hybrid.SearchConfig((hybrid.Branch("route"),))
    with make_searcher() as searcher:
        one = searcher.search(hybrid.SearchQuery("first"), config)
        two = searcher.search(hybrid.SearchQuery("second"), config)
        assert [hit.asset_id for hit in one.result.hits] == [1]
        assert [hit.asset_id for hit in two.result.hits] == [2]
        assert one.workers[0].pid == two.workers[0].pid
        assert two.workers[0].exit_code is None
        os.kill(two.workers[0].pid, 0)
    assert_reaped(two.workers[0].pid)
    with pytest.raises(RuntimeError, match="with block"):
        searcher.search(hybrid.SearchQuery("closed"), config)


def test_timed_out_warm_instance_is_reaped_and_next_request_reconstructs():
    config = hybrid.SearchConfig((hybrid.Branch("route"),), failure_policy="partial")
    with make_searcher() as searcher:
        one = searcher.search(hybrid.SearchQuery("first"), config)
        failed = searcher.search(
            hybrid.SearchQuery("stall"), config, budget=execution.ExecutionBudget(branch_seconds=0.3, request_seconds=1)
        )
        assert failed.result.degraded and not failed.result.hits
        assert_reaped(one.workers[0].pid)
        replaced = searcher.search(hybrid.SearchQuery("first"), config)
        assert replaced.workers[0].pid != one.workers[0].pid
        assert [hit.asset_id for hit in replaced.result.hits] == [1]
    assert_reaped(replaced.workers[0].pid)


def test_strict_contract_violation_retires_worker_and_close_still_cleans_owned_state():
    config = hybrid.SearchConfig((hybrid.Branch("route"),), failure_policy="partial")
    with make_searcher() as searcher:
        one = searcher.search(hybrid.SearchQuery("first"), config)
        with pytest.raises(hybrid.RetrievalError, match="contract"):
            searcher.search(hybrid.SearchQuery("second", hybrid.SearchFilter(allowed_ids=frozenset({1}))), config)
        assert_reaped(one.workers[0].pid)
        assert not searcher._workers


def test_single_instance_rejects_concurrent_use_and_requires_explicit_capacity():
    with make_searcher() as searcher:
        config = hybrid.SearchConfig((hybrid.Branch("route"),))
        searcher._lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="Concurrent"):
                searcher.search(hybrid.SearchQuery("query"), config)
            with pytest.raises(RuntimeError, match="active"):
                searcher.close()
        finally:
            searcher._lock.release()
