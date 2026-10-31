"""Explicitly managed warm retriever processes with cancellation and replacement."""

from __future__ import annotations

import os
import threading
import time

from . import contracts, execution, fusion


def _warm_worker(connection, provider) -> None:
    """Load once, then handle serial JSON requests until this owned worker closes."""
    os.setsid()
    try:
        retriever = execution._construct(provider)
        while True:
            try:
                request, limit = connection.recv()
            except EOFError:
                return
            try:
                ids = retriever.retrieve(execution._query(request), limit)
                try:
                    bounded = fusion.unique_ids(ids, limit)
                except (TypeError, ValueError):
                    connection.send(("invalid", "InvalidCandidateIdentity"))
                    return
                connection.send(("ok", bounded))
            except execution.FilterContractError:
                connection.send(("invalid", "FilterContractError"))
                return
            except Exception as exc:
                # Arbitrary native providers are an explicit plugin boundary.
                connection.send(("failed", type(exc).__name__))
                return
    except Exception as exc:
        connection.send(("failed", type(exc).__name__))
    finally:
        connection.close()


class WarmProcessSearch(execution.ProcessSearch):
    """Reuse successful per-route models inside an explicit with block.

    Workers remain alive on success and are reaped on timeout/failure/close.
    The inherited request deadline also covers cold final fusion/fetch workers.
    One instance supports one request at a time; concurrent calls fail visibly,
    rather than silently interleaving messages on a shared process connection.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._workers: dict[str, tuple[object, object]] = {}
        self._lock = threading.Lock()
        self._entered = False

    def __enter__(self) -> WarmProcessSearch:
        if self._entered:
            raise RuntimeError("Warm executor is already entered")
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Reap every owned worker; fail instead of abandoning active requests."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close a warm executor during an active request")
        try:
            failures = []
            for process, connection in self._workers.values():
                try:
                    execution._stop(process)
                except BaseException as exc:
                    failures.append(exc)
                finally:
                    connection.close()
            if failures:
                raise RuntimeError("Warm worker cleanup could not be confirmed") from failures[0]
            self._workers.clear()
            self._entered = False
        finally:
            self._lock.release()

    def search(
        self,
        query: contracts.SearchQuery,
        config: contracts.SearchConfig,
        *,
        budget: execution.ExecutionBudget | None = None,
    ) -> execution.ExecutionResult:
        """Apply the unchanged process contracts, retaining only successful workers."""
        if not self._entered:
            raise RuntimeError("Warm executor must be used inside a with block")
        budget = budget or execution.ExecutionBudget()
        if len(self.providers) > budget.max_workers:
            raise ValueError("Warm provider count exceeds the total process capacity")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent requests require separate warm executor instances")
        try:
            return super().search(query, config, budget=budget)
        finally:
            self._lock.release()

    def _launch(self, context, pending, running, request, budget, deadline) -> None:
        while pending and len(running) < budget.max_workers and time.monotonic() < deadline:
            branch = pending.pop(0)
            worker = self._workers.get(branch.name)
            if worker is not None and not worker[0].is_alive():
                execution._stop(worker[0])
                worker[1].close()
                del self._workers[branch.name]
                worker = None
            started = time.monotonic()
            if worker is None:
                parent, child = context.Pipe(duplex=True)
                process = context.Process(
                    target=_warm_worker, args=(child, execution._provider_tuple(self.providers[branch.name]))
                )
                try:
                    process.start()
                except BaseException:
                    if process.pid is not None:
                        execution._stop(process)
                    parent.close()
                    child.close()
                    raise
                child.close()
                worker = process, parent
                self._workers[branch.name] = worker
            process, connection = worker
            # Register ownership before send: an interrupted/failed send must
            # still be cancelled by the inherited request's finally block.
            running[branch.name] = execution._Running(
                branch.name, process, connection, started, min(deadline, started + budget.branch_seconds)
            )
            connection.send((request, branch.window))

    def _release(self, item: execution._Running, status: str) -> execution.WorkerEvent:
        if status == "ok" and item.process.is_alive():
            return execution.WorkerEvent(
                item.name, item.process.pid, "ok", None, (time.monotonic() - item.started) * 1000
            )
        self._workers.pop(item.name, None)
        return super()._release(item, status)
