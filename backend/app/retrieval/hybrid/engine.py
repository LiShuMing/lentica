"""Synchronous coordinator with explicit budgets, failures and one final fetch."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from . import contracts
from . import fusion as rank_fusion


class RetrievalError(RuntimeError):
    """A retriever or fusion provider violated the request contract."""


class HydrationError(RuntimeError):
    """The final document fetch failed or omitted a ranked record."""


class HybridSearch:
    """Coordinate filtered retrieval, rank fusion and one final batch fetch.

    Retriever and store failures are wrapped with chained causes; partial mode
    tolerates only retriever failures. Custom fusion exceptions propagate as
    raised; successfully returned hits are checked before the final fetch.
    """

    def __init__(
        self,
        retrievers: Mapping[str, contracts.Retriever],
        *,
        store: contracts.DocumentStore | None = None,
        fusion: contracts.FusionStrategy | None = None,
    ) -> None:
        """Register named retrievers and optional store/fusion providers."""
        self.retrievers = dict(retrievers)
        self.store = store
        self.fusion = fusion or rank_fusion.RrfFusion()

    def search(self, query: contracts.SearchQuery, config: contracts.SearchConfig) -> contracts.SearchResult:
        """Retrieve and fuse IDs, then fetch only final records.

        Args:
            query: Text and shared filters for every enabled branch.
            config: Candidate budgets, weights, final limit and failure policy.

        Returns:
            Ranked hits, optional records and explicit branch diagnostics.

        Raises:
            ValueError: An enabled branch has no registered retriever.
            RetrievalError: A required retriever fails or fusion returns invalid hits.
            HydrationError: Final record fetching fails or is incomplete.

        Exceptions from the custom fusion implementation propagate unchanged.
        """
        # Configuration mistakes are never classified as tolerated runtime failures.
        for branch in config.branches:
            if branch.weight and branch.window and branch.name not in self.retrievers:
                raise ValueError(f"No retriever registered for {branch.name}")
        rankings, diagnostics = self._retrieve(query, config)
        hits = tuple(self.fusion.fuse(rankings, config))
        self._validate_hits(hits, rankings, config.top_k)
        records = self._fetch([hit.asset_id for hit in hits])
        return contracts.SearchResult(
            hits, MappingProxyType(records), tuple(diagnostics), any(item.status == "failed" for item in diagnostics)
        )

    def _retrieve(
        self, query: contracts.SearchQuery, config: contracts.SearchConfig
    ) -> tuple[dict[str, tuple[int, ...]], list[contracts.BranchDiagnostic]]:
        """Run bounded branches, translating failures at the retriever boundary."""
        rankings: dict[str, tuple[int, ...]] = {}
        diagnostics: list[contracts.BranchDiagnostic] = []
        for branch in config.branches:
            if not branch.weight or not branch.window:
                diagnostics.append(contracts.BranchDiagnostic(branch.name, "disabled"))
                continue
            started = time.perf_counter()
            try:
                ids = self.retrievers[branch.name].retrieve(query, branch.window)
                if len(ids) > contracts.MAX_WINDOW:
                    raise ValueError("Retriever exceeded maximum response budget")
                rankings[branch.name] = rank_fusion.unique_ids(ids, branch.window)
                diagnostics.append(
                    contracts.BranchDiagnostic(
                        branch.name, "ok", len(rankings[branch.name]), (time.perf_counter() - started) * 1000
                    )
                )
            except Exception as exc:
                # Retriever plugins may raise arbitrary exceptions. Keep their
                # causes in strict mode; partial diagnostics expose class names.
                if config.failure_policy == "raise":
                    raise RetrievalError(f"Retriever {branch.name} failed") from exc
                diagnostics.append(
                    contracts.BranchDiagnostic(
                        branch.name,
                        "failed",
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        error=type(exc).__name__,
                    )
                )
        return rankings, diagnostics

    @staticmethod
    def _validate_hits(hits: Sequence[contracts.FusedHit], rankings: Mapping[str, Sequence[int]], top_k: int) -> None:
        """Reject invalid custom fusion output before touching the document store."""
        ids = [hit.asset_id for hit in hits]
        if (
            len(ids) > top_k
            or any(not math.isfinite(hit.score) for hit in hits)
            or len(set(ids)) != len(ids)
            or not set(ids) <= {identifier for values in rankings.values() for identifier in values}
        ):
            raise RetrievalError("Fusion returned IDs outside the candidate contract")

    def _fetch(self, ids: Sequence[int]) -> dict[int, object]:
        """Translate arbitrary store failures at the final-fetch extension point."""
        records: dict[int, object] = {}
        if self.store is not None and ids:
            try:
                fetched = self.store.fetch(ids)
                records = {identifier: fetched[identifier] for identifier in ids}
            except Exception as exc:
                # Final records are mandatory even when retrieval was partial.
                raise HydrationError("Final batch fetch failed or omitted a ranked record") from exc
        return records
