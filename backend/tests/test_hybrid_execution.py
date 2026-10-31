"""Actual spawn, deadlines, process reaping and fail-closed filter checks."""

from __future__ import annotations

import json
import os
import time

import pytest

from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import execution


def provider(**kwargs):
    return execution.ProviderSpec("backend.tests.process_providers", "DelayedProvider", kwargs)


def assert_reaped(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_parallel_deadline_really_terminates_worker_and_preserves_fast_result(tmp_path):
    pid_file = tmp_path / "slow.pid"
    searcher = execution.ProcessSearch(
        {"fast": provider(ids=[1]), "slow": provider(ids=[2], delay=10, pid_file=str(pid_file))}, {1: {}, 2: {}}
    )
    config = hybrid.SearchConfig((hybrid.Branch("fast"), hybrid.Branch("slow")), failure_policy="partial")
    started = time.monotonic()
    actual = searcher.search(
        hybrid.SearchQuery("query"),
        config,
        budget=execution.ExecutionBudget(request_seconds=3, branch_seconds=0.7, max_workers=2),
    )
    assert time.monotonic() - started < 3
    assert [hit.asset_id for hit in actual.result.hits] == [1] and actual.result.degraded
    slow = next(item for item in actual.workers if item.branch == "slow")
    assert slow.status == "timeout" and slow.exit_code is not None
    assert pid_file.exists() and int(pid_file.read_text()) == slow.pid
    for event in actual.workers:
        assert_reaped(event.pid)


def test_request_deadline_covers_queued_branches(tmp_path):
    specs = {"first": provider(delay=10), "queued": provider(pid_file=str(tmp_path / "not-started"))}
    config = hybrid.SearchConfig(tuple(hybrid.Branch(name) for name in specs), failure_policy="partial")
    actual = execution.ProcessSearch(specs, {}).search(
        hybrid.SearchQuery("query"),
        config,
        budget=execution.ExecutionBudget(request_seconds=0.4, branch_seconds=2, max_workers=1),
    )
    assert actual.result.degraded and not actual.result.hits
    assert actual.workers[1].status == "not_started" and actual.workers[1].pid is None
    assert not (tmp_path / "not-started").exists()
    assert_reaped(actual.workers[0].pid)


@pytest.mark.parametrize("kwargs", [{"ids": [True]}, {"ids": [99]}, {"ids": [2]}, {"invalid_filter": True}])
def test_candidate_and_filter_violations_never_degrade(kwargs):
    config = hybrid.SearchConfig((hybrid.Branch("route"),), failure_policy="partial")
    query = hybrid.SearchQuery("query", hybrid.SearchFilter(allowed_ids=frozenset({1})))
    searcher = execution.ProcessSearch({"route": provider(**kwargs)}, {1: {}, 2: {}})
    with pytest.raises(hybrid.RetrievalError, match="contract|filter"):
        searcher.search(query, config)


def test_strict_failure_cancels_other_owned_workers(tmp_path):
    pid_file = tmp_path / "worker.pid"
    config = hybrid.SearchConfig((hybrid.Branch("slow"), hybrid.Branch("broken")))
    searcher = execution.ProcessSearch(
        {"slow": provider(delay=10, pid_file=str(pid_file)), "broken": provider(delay=0.2, failure=True)}, {}
    )
    with pytest.raises(hybrid.RetrievalError) as caught:
        searcher.search(hybrid.SearchQuery("query"), config)
    assert isinstance(caught.value.__cause__, execution.RemoteProviderError)
    assert pid_file.exists()
    assert_reaped(int(pid_file.read_text()))


def test_only_final_ids_are_fetched_once_and_disabled_factories_are_not_required(tmp_path):
    call_file = tmp_path / "calls.json"
    store = execution.ProviderSpec("backend.tests.process_providers", "DelayedStore", {"call_file": str(call_file)})
    config = hybrid.SearchConfig((hybrid.Branch("route", 3), hybrid.Branch("disabled", weight=0)), top_k=1)
    result = execution.ProcessSearch({"route": provider(ids=[2, 1, 1])}, {1: {}, 2: {}}, store=store).search(
        hybrid.SearchQuery("query"), config
    )
    assert json.loads(call_file.read_text()) == [2] and not result.result.degraded
    assert result.result.records == {2: "record"}
    assert result.result.branches[1].status == "disabled"
    assert_reaped(result.finalization.pid)
    assert result.finalization.exit_code == 0


@pytest.mark.parametrize("stage", ["fetch", "construct", "fusion"])
def test_final_plugins_share_request_deadline_and_are_reaped(tmp_path, stage):
    pid_file = tmp_path / "final.pid"
    kwargs = {"pid_file": str(pid_file)}
    kwargs["init_delay" if stage == "construct" else "delay"] = 10
    factory = "DelayedFusion" if stage == "fusion" else "DelayedStore"
    spec = execution.ProviderSpec("backend.tests.process_providers", factory, kwargs)
    searcher = execution.ProcessSearch(
        {"route": provider(ids=[1])}, {1: {}}, **{"fusion_strategy" if stage == "fusion" else "store": spec}
    )
    config = hybrid.SearchConfig((hybrid.Branch("route"),), failure_policy="partial")
    started = time.monotonic()
    with pytest.raises(execution.RequestDeadlineError):
        searcher.search(
            hybrid.SearchQuery("query"),
            config,
            budget=execution.ExecutionBudget(request_seconds=1.5, branch_seconds=1),
        )
    assert time.monotonic() - started < 3
    assert pid_file.exists()
    assert_reaped(int(pid_file.read_text()))


@pytest.mark.parametrize(
    "kwargs,exception", [({"missing": True}, hybrid.HydrationError), ({"oversized": True}, hybrid.RetrievalError)]
)
def test_invalid_final_results_cannot_degrade(kwargs, exception):
    store = execution.ProviderSpec("backend.tests.process_providers", "DelayedStore", kwargs)
    searcher = execution.ProcessSearch({"route": provider(ids=[1])}, {1: {}}, store=store)
    with pytest.raises(exception):
        searcher.search(
            hybrid.SearchQuery("query"), hybrid.SearchConfig((hybrid.Branch("route"),), failure_policy="partial")
        )


def test_parent_plugin_instances_are_rejected_before_starting_workers():
    with pytest.raises(TypeError, match="ProviderSpec"):
        execution.ProcessSearch({}, {}, store=object())


def test_provider_arguments_are_snapshotted_without_loading_provider_modules():
    original = {"ids": [1]}
    spec = provider(**original)
    original["ids"].append(2)
    assert spec.kwargs["ids"] == [1]
    with pytest.raises(ValueError):
        execution.ExecutionBudget(branch_seconds=float("nan"))
