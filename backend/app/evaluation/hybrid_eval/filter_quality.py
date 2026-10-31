"""Real two-route prefilter vs deliberately incorrect post-filter controls."""

from __future__ import annotations

import dataclasses

from backend.app.evaluation.hybrid_eval import dataset, index_quality, reporting
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import adapters, ann, fulltext


def _fuse(query, text_ids, dense_ids):
    config = hybrid.SearchConfig((hybrid.Branch("text", 50), hybrid.Branch("dense", 50)))
    result = hybrid.HybridSearch(
        {"text": adapters.RankedListRetriever(text_ids), "dense": adapters.RankedListRetriever(dense_ids)}
    ).search(query, config)
    return tuple(hit.asset_id for hit in result.hits)


def run(source: dataset.Dataset, text: fulltext.TantivyRetriever, dense: ann.HnswRetriever) -> reporting.JsonObject:
    """Compare identical text/ANN domains and bounds; report every loss, no tuning."""
    identity = dataset.dataset_identity(source)
    if (
        text.source_version != identity
        or dense.source_version != identity
        or set(text.metadata) != set(dense.ids)
        or set(dense.ids) != {doc.asset_id for doc in source.documents}
    ):
        raise ValueError("Filter controls require the same source/unit domain")
    domains = {}
    for modulus in (4, 100):
        allowed = frozenset(identifier for identifier in dense.ids if identifier % modulus == 0)
        cases = [
            dataclasses.replace(case, relevant_asset_ids=case.relevant_asset_ids & allowed) for case in source.cases
        ]
        rankings = {name: {} for name in ("prefilter", "after-route50", "after-fused10")}
        for case in cases:
            query = hybrid.SearchQuery(case.query)
            filtered = hybrid.SearchQuery(case.query, hybrid.SearchFilter(allowed_ids=allowed))
            pretext, predense = text.retrieve(filtered, 50), dense.retrieve(filtered, 50)
            if not set(pretext + predense) <= allowed:
                raise ValueError("A native branch violated pre-window filtering")
            rankings["prefilter"][case.query_id] = _fuse(query, pretext, predense)
            rawtext, rawdense = text.retrieve(query, 50), dense.retrieve(query, 50)
            rankings["after-route50"][case.query_id] = _fuse(
                query,
                tuple(identifier for identifier in rawtext if identifier in allowed),
                tuple(identifier for identifier in rawdense if identifier in allowed),
            )
            rankings["after-fused10"][case.query_id] = tuple(
                identifier for identifier in _fuse(query, rawtext, rawdense) if identifier in allowed
            )
        domains[str(modulus)] = {
            "eligible_documents": len(allowed),
            "quality": {name: index_quality.summarize(cases, ranks) for name, ranks in rankings.items()},
            "mean_returned": {
                name: sum(len(ids) for ids in ranks.values()) / len(cases) for name, ranks in rankings.items()
            },
            "rankings": rankings,
        }
    return {
        "domains": domains,
        "ranking_sha256": reporting.digest({name: values["rankings"] for name, values in domains.items()}),
        "scope": "Fixed deliberately incorrect controls; allowed IDs derive only from ID modulo, never qrels. Both correct branches filter before candidate windows.",
    }
