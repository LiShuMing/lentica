"""Opt-in POSIX process isolation with bounded parallelism and real cancellation.

Providers are constructed inside disposable workers. A branch deadline includes
spawn and model/index initialization. This correctness-first local executor does
not claim the warm latency of a persistent model-serving pool.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import multiprocessing
import multiprocessing.connection
import os
import signal
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from . import adapters, contracts, engine, fusion, snapshots


class FilterContractError(ValueError):
    """An adapter cannot honor a requested filter; never eligible for degradation."""


class RemoteProviderError(RuntimeError):
    """A worker failed; only its exception class is exposed across the boundary."""


class RequestDeadlineError(engine.RetrievalError):
    """Required finalization did not finish within the request's wall budget."""


MAX_FINAL_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ProviderSpec:
    """A trusted application-selected module/factory plus JSON-only constructor args.

    This configuration is not accepted from untrusted search text. No module is
    imported at construction. JSON freezes kwargs and avoids arbitrary object
    pickling from a caller's retriever/model instances.
    """

    module: str
    factory: str
    kwargs: Mapping[str, object] = field(default_factory=dict)
    _kwargs_json: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not self.module
            or any(not part.isidentifier() for part in self.module.split("."))
            or not self.factory.isidentifier()
        ):
            raise ValueError("Provider module/factory must be explicit Python identifiers")
        encoded = snapshots.canonical(dict(self.kwargs))
        object.__setattr__(self, "_kwargs_json", encoded)
        object.__setattr__(self, "kwargs", MappingProxyType(json.loads(encoded)))


@dataclass(frozen=True)
class ExecutionBudget:
    """Request/branch deadlines and retriever capacity, plus one final worker."""

    request_seconds: float = 30.0
    branch_seconds: float = 10.0
    max_workers: int = 2

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or not 0 < value <= 300 for value in (self.request_seconds, self.branch_seconds)
        ):
            raise ValueError("Execution deadlines must be finite, positive and at most 300 seconds")
        if type(self.max_workers) is not int or not 1 <= self.max_workers <= 8:
            raise ValueError("Worker budget must be between 1 and 8")


@dataclass(frozen=True)
class WorkerEvent:
    """Owned worker identity and observed completion; cancelled workers are reaped."""

    branch: str
    pid: int | None
    status: str
    exit_code: int | None
    elapsed_ms: float


@dataclass(frozen=True)
class ExecutionResult:
    """Compatible final search result plus actual worker lifecycle evidence."""

    result: contracts.SearchResult
    workers: tuple[WorkerEvent, ...]
    finalization: WorkerEvent | None = None


def _construct(provider: tuple[str, str, str] | None):
    if provider is None:
        return None
    module, factory, kwargs = provider
    return getattr(importlib.import_module(module), factory)(**json.loads(kwargs))


def _provider_tuple(provider: ProviderSpec | None) -> tuple[str, str, str] | None:
    return None if provider is None else (provider.module, provider.factory, provider._kwargs_json)


def _query(request: str) -> contracts.SearchQuery:
    value = json.loads(request)
    return contracts.SearchQuery(value["text"], contracts.SearchFilter(value["equals"], value["allowed_ids"]))


def _final_worker(connection, rankings, request, config, store, strategy, output: str) -> None:
    """Run every arbitrary final plugin inside the same cancellable boundary.

    Only a small status crosses the pipe: large final records are bounded on
    disk, so a pipe becoming readable cannot lead to an unbounded recv of records.
    """
    os.setsid()
    try:
        searcher = engine.HybridSearch(
            {name: adapters.RankedListRetriever(ids) for name, ids in rankings.items()},
            store=_construct(store),
            fusion=_construct(strategy),
        )
        result = searcher.search(_query(request), config)
        payload = {
            "hits": [dataclasses.asdict(hit) for hit in result.hits],
            "records": {str(key): value for key, value in result.records.items()},
        }
        # Stream encoding bounds allocation/output even for an oversized record.
        size = 0
        with Path(output).open("xb") as stream:
            for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(payload):
                encoded = chunk.encode("utf-8")
                size += len(encoded)
                if size > MAX_FINAL_BYTES:
                    raise ValueError("Final result exceeds the cross-process byte budget")
                stream.write(encoded)
        connection.send(("ok", None))
    except engine.HydrationError:
        connection.send(("hydration_failed", "HydrationError"))
    except Exception as exc:
        connection.send(("failed", type(exc).__name__))
    finally:
        connection.close()


def _worker(connection, provider: tuple[str, str, str], request: str, limit: int) -> None:
    """Own a new process group so cancellation also reaches native tool children."""
    os.setsid()
    try:
        retriever = _construct(provider)
        query = _query(request)
        try:
            identifiers = retriever.retrieve(query, limit)
        except FilterContractError:
            connection.send(("invalid", "FilterContractError"))
            return
        try:
            bounded = fusion.unique_ids(identifiers, limit)
        except (TypeError, ValueError):
            connection.send(("invalid", "InvalidCandidateIdentity"))
            return
        connection.send(("ok", bounded))
    except Exception as exc:
        # Arbitrary provider factories/runtime libraries are extension points.
        # Preserve the class without leaking query content or provider secrets.
        connection.send(("failed", type(exc).__name__))
    finally:
        connection.close()


def _stop(process) -> None:
    """Terminate only this owned child/group, escalate if needed, then reap it."""
    if process.is_alive():
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()  # Worker has not yet created its own session.
        except ProcessLookupError:
            pass
        process.join(timeout=0.15)
    if process.is_alive():
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
    process.join(timeout=1)
    if process.is_alive():
        raise RuntimeError("Cannot confirm cancellation/reaping of an owned worker")


@dataclass
class _Running:
    name: str
    process: object
    connection: object
    started: float
    deadline: float


class ProcessSearch:
    """Run explicit text-route factories concurrently, validate IDs, then fuse.

    Metadata is the complete unit/asset domain for these providers. It enables
    fail-closed checks before fusion: unknown or excluded returned IDs always
    fail, even in partial mode. Only runtime failures and deadlines can degrade.
    No final store read occurs before all required contracts have passed.
    """

    def __init__(
        self,
        providers: Mapping[str, ProviderSpec],
        metadata: Mapping[int, Mapping[str, object]],
        *,
        store: ProviderSpec | None = None,
        fusion_strategy: ProviderSpec | None = None,
    ) -> None:
        if os.name != "posix" or not hasattr(os, "setsid"):
            raise RuntimeError("Process-group cancellation requires a POSIX host")
        if any(value is not None and not isinstance(value, ProviderSpec) for value in (store, fusion_strategy)):
            raise TypeError("ProcessSearch final plugins must be ProviderSpec factories")
        self._providers = MappingProxyType(dict(providers))
        self.metadata = {contracts.asset_id(key): dict(value) for key, value in metadata.items()}
        self.store, self.fusion = store, fusion_strategy

    @property
    def providers(self) -> Mapping[str, ProviderSpec]:
        """Captured factories; new configuration requires a new executor owner."""
        return self._providers

    def search(
        self, query: contracts.SearchQuery, config: contracts.SearchConfig, *, budget: ExecutionBudget | None = None
    ) -> ExecutionResult:
        """Run bounded workers; finally cancel/reap every remaining owned process."""
        budget = budget or ExecutionBudget()
        enabled = [branch for branch in config.branches if branch.weight and branch.window]
        if any(branch.name not in self.providers for branch in enabled):
            raise ValueError("No factory configured for an enabled branch")
        deadline = time.monotonic() + budget.request_seconds
        context = multiprocessing.get_context("spawn")
        request = snapshots.canonical(
            {
                "text": query.text,
                "equals": dict(query.filters.equals),
                "allowed_ids": None if query.filters.allowed_ids is None else sorted(query.filters.allowed_ids),
            }
        )
        pending = list(enabled)
        running: dict[str, _Running] = {}
        rankings: dict[str, tuple[int, ...]] = {}
        diagnostics = {
            branch.name: contracts.BranchDiagnostic(branch.name, "disabled")
            for branch in config.branches
            if branch not in enabled
        }
        events: dict[str, WorkerEvent] = {}
        try:
            while pending or running:
                self._launch(context, pending, running, request, budget, deadline)
                now = time.monotonic()
                if now >= deadline:
                    for branch in pending:
                        diagnostics[branch.name] = contracts.BranchDiagnostic(
                            branch.name, "failed", error="RequestDeadline"
                        )
                        events[branch.name] = WorkerEvent(branch.name, None, "not_started", None, 0)
                    pending.clear()
                ready = (
                    multiprocessing.connection.wait([item.connection for item in running.values()], timeout=0.02)
                    if running
                    else ()
                )
                for name, item in tuple(running.items()):
                    status, payload = self._poll(item, ready, deadline)
                    if status is None:
                        continue
                    if status == "ok" and any(
                        identifier not in self.metadata
                        or not query.filters.matches(identifier, self.metadata[identifier])
                        for identifier in payload
                    ):
                        status, payload = "invalid", "PreWindowFilterViolation"
                    event = self._release(item, status)
                    elapsed = event.elapsed_ms
                    del running[name]
                    events[name] = event
                    if status == "invalid":
                        raise engine.RetrievalError(f"Provider {name} violated the filter/candidate contract")
                    if status == "ok":
                        rankings[name] = payload
                        diagnostics[name] = contracts.BranchDiagnostic(name, "ok", len(payload), elapsed)
                    else:
                        diagnostics[name] = contracts.BranchDiagnostic(
                            name, "failed", elapsed_ms=elapsed, error=payload
                        )
                if config.failure_policy == "raise" and any(item.status == "failed" for item in diagnostics.values()):
                    raise engine.RetrievalError(
                        "Required process retriever failed or timed out"
                    ) from RemoteProviderError(
                        next(item.error for item in diagnostics.values() if item.status == "failed")
                    )
            # Disabled/failed branches are represented explicitly; replacing the
            # execution branches only avoids asking HybridSearch to rerun failures.
            active = tuple(branch for branch in config.branches if branch.name in rankings)
            if not active:
                result = contracts.SearchResult((), MappingProxyType({}), (), bool(enabled))
                final_event = None
            else:
                result, final_event = self._finalize(
                    context, rankings, request, dataclasses.replace(config, branches=active), deadline
                )
            ordered_diagnostics = tuple(diagnostics[branch.name] for branch in config.branches)
            result = dataclasses.replace(
                result,
                branches=ordered_diagnostics,
                degraded=any(item.status == "failed" for item in ordered_diagnostics),
            )
            return ExecutionResult(result, tuple(events[branch.name] for branch in enabled), final_event)
        finally:
            cleanup_failure = None
            for item in running.values():
                try:
                    _stop(item.process)
                except BaseException as exc:
                    cleanup_failure = exc
                finally:
                    item.connection.close()
            if cleanup_failure is not None:
                raise RuntimeError("Owned worker cleanup could not be confirmed") from cleanup_failure

    def _release(self, item: _Running, status: str) -> WorkerEvent:
        """Retire a disposable worker; managed warm execution overrides reuse."""
        if status != "timeout":
            item.process.join(timeout=0.05)
        _stop(item.process)
        item.connection.close()
        return WorkerEvent(
            item.name, item.process.pid, status, item.process.exitcode, (time.monotonic() - item.started) * 1000
        )

    def _finalize(self, context, rankings, request, config, deadline):
        """Share the request deadline across final plugin construction and work."""
        if time.monotonic() >= deadline:
            raise RequestDeadlineError("Request deadline exhausted before finalization")
        receive, send = context.Pipe(duplex=False)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="lentica-final-") as directory:
            output = Path(directory) / "result.json"
            process = context.Process(
                target=_final_worker,
                args=(
                    send,
                    rankings,
                    request,
                    config,
                    _provider_tuple(self.store),
                    _provider_tuple(self.fusion),
                    str(output),
                ),
            )
            try:
                process.start()
                send.close()
                item = _Running("finalization", process, receive, started, deadline)
                while True:
                    ready = multiprocessing.connection.wait(
                        [receive], timeout=min(0.02, max(0, deadline - time.monotonic()))
                    )
                    status, error = self._poll(item, ready, deadline)
                    if status is not None:
                        break
                if status == "timeout":
                    raise RequestDeadlineError("Request deadline exceeded during fusion/final fetch")
                if status == "hydration_failed":
                    raise engine.HydrationError("Final batch fetch failed") from RemoteProviderError(error)
                if status != "ok":
                    raise engine.RetrievalError("Finalization provider failed") from RemoteProviderError(error)
                if output.stat().st_size > MAX_FINAL_BYTES:
                    raise engine.RetrievalError("Final response exceeded its byte budget")
                value = json.loads(output.read_bytes())
                hits = tuple(
                    contracts.FusedHit(
                        hit["asset_id"], hit["score"], tuple(contracts.Contribution(**c) for c in hit["contributions"])
                    )
                    for hit in value["hits"]
                )
                result = contracts.SearchResult(
                    hits, MappingProxyType({int(key): record for key, record in value["records"].items()}), (), False
                )
                if time.monotonic() >= deadline:
                    raise RequestDeadlineError("Request deadline exceeded while decoding final response")
            finally:
                send.close()
                receive.close()
                if process.pid is not None:
                    process.join(timeout=0.05)
                    _stop(process)
            event = WorkerEvent(
                "finalization", process.pid, "ok", process.exitcode, (time.monotonic() - started) * 1000
            )
        return result, event

    def _launch(self, context, pending, running, request, budget, deadline) -> None:
        while pending and len(running) < budget.max_workers and time.monotonic() < deadline:
            branch = pending.pop(0)
            receive, send = context.Pipe(duplex=False)
            provider = self.providers[branch.name]
            process = context.Process(
                target=_worker,
                args=(send, (provider.module, provider.factory, provider._kwargs_json), request, branch.window),
            )
            started = time.monotonic()
            try:
                process.start()
            except BaseException:
                if process.pid is not None:
                    _stop(process)
                receive.close()
                send.close()
                raise
            send.close()
            running[branch.name] = _Running(
                branch.name, process, receive, started, min(deadline, started + budget.branch_seconds)
            )

    @staticmethod
    def _poll(item, ready, request_deadline):
        if time.monotonic() >= item.deadline:
            return "timeout", "RequestDeadline" if time.monotonic() >= request_deadline else "BranchDeadline"
        if item.connection in ready:
            try:
                return item.connection.recv()
            except EOFError:
                return "failed", "WorkerExitedWithoutResult"
        if not item.process.is_alive():
            return "failed", "WorkerExitedWithoutResult"
        return None, None
