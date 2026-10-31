"""Development/validation selection and checksum-bound historical test evaluation."""

from __future__ import annotations

import dataclasses
import re
import sys
from collections.abc import Mapping, Sequence

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation import metrics
from backend.app.evaluation.hybrid_eval import dataset as datasets
from backend.app.evaluation.hybrid_eval import execution, reporting
from backend.app.retrieval import hybrid

SCHEMA = "lentica-hybrid-selection-v1"


def normalized_query(text: str) -> str:
    """Normalize case and word separators for conservative leakage detection."""
    return " ".join(re.findall(r"\w+", text.casefold()))


def partition_training(
    train: datasets.Dataset, test: datasets.Dataset, *, seed: int = 42
) -> tuple[list[evaluation_dataset.EvaluationCase], list[evaluation_dataset.EvaluationCase], reporting.JsonObject]:
    """Exclude test overlap and keep normalized-text groups in one partition.

    Args:
        train: Official training queries and the selected corpus.
        test: Historical test queries, used only to exclude text/ID overlap.
        seed: Stable group ordering seed, independent of relevance labels.

    Returns:
        Development queries, validation queries and partition evidence.

    Raises:
        ValueError: Corpora differ or too few nonempty training groups remain.
    """
    if datasets.dataset_identity(train) != datasets.dataset_identity(test):
        raise ValueError("Train/test corpus mismatch")
    test_texts = {normalized_query(case.query) for case in test.cases}
    test_ids = {case.query_id for case in test.cases}
    groups, excluded = {}, []
    for case in sorted(train.cases, key=lambda case: case.query_id):
        text = normalized_query(case.query)
        if not text:
            raise ValueError("Empty normalized training query")
        if text in test_texts or case.query_id in test_ids:
            excluded.append(case.query_id)
        else:
            groups.setdefault(text, []).append(case)
    # Partition text groups, not individual query IDs: differently labeled copies
    # of the same normalized text must never straddle development and validation.
    ordered = sorted(groups, key=lambda text: (reporting.digest([seed, text]), text))
    if len(ordered) < 5:
        raise ValueError("Need at least five distinct nonoverlapping training queries")
    validation_keys = set(ordered[: max(1, len(ordered) // 5)])
    development = sorted(
        [case for key, rows in groups.items() if key not in validation_keys for case in rows],
        key=lambda case: case.query_id,
    )
    validation = sorted(
        [case for key, rows in groups.items() if key in validation_keys for case in rows],
        key=lambda case: case.query_id,
    )
    return (
        development,
        validation,
        {
            "seed": seed,
            "method": "normalized-text groups; SHA256(seed,text) order; first floor(groups/5) validation",
            "excluded_train_query_ids": excluded,
            "development_queries": len(development),
            "validation_queries": len(validation),
            "development_groups": len(groups) - len(validation_keys),
            "validation_groups": len(validation_keys),
            "development_sha256": datasets.case_identity(development),
            "validation_sha256": datasets.case_identity(validation),
            "test_query_text_sha256": reporting.digest(sorted(test_texts)),
        },
    )


def best_name(reports: Mapping[str, reporting.JsonObject]) -> str:
    """Select by nDCG, Recall, MRR and finally stable name, without rounding."""
    return min(reports, key=lambda name: (*(-reports[name]["metrics"][key] for key in execution.METRICS), name))


def choose_validation(reports: Mapping[str, reporting.JsonObject]) -> str:
    """Require nDCG gain >= .005 and Recall loss <= .01 or retain baseline."""
    baseline = reports["baseline"]["metrics"]
    eligible = {
        name: report
        for name, report in reports.items()
        if name != "baseline"
        and report["metrics"]["ndcg_at_10"] >= baseline["ndcg_at_10"] + 0.005
        and report["metrics"]["recall_at_10"] >= baseline["recall_at_10"] - 0.01
    }
    return best_name(eligible) if eligible else "baseline"


def _develop_candidates(
    routes: Mapping[str, hybrid.Retriever],
    development: Sequence[evaluation_dataset.EvaluationCase],
    pools: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    progress: bool,
) -> tuple[reporting.JsonObject, dict[str, hybrid.SearchConfig]]:
    """Pick one candidate per family using development labels only."""
    development_reports = {}
    lexical_configs = {name: execution.config_for(name) for name in routes if name.startswith("bm25-")}
    for name, config in lexical_configs.items():
        development_reports[name], _ = execution.evaluate_config(config, development, pools)
    # Freeze the development-best BM25 before comparing secondary families;
    # validation only chooses among the resulting five finalists.
    lexical = best_name(development_reports)
    finalists = {"baseline": execution.config_for(execution.BASELINE), "lexical": execution.config_for(lexical)}
    for secondary in ("word", "char", "lsa"):
        family = {}
        configs = {}
        for weight in (0.1, 0.25, 0.5, 1.0):
            for constant in (1, 10, 60):
                for window in (50, 200):
                    name = f"{secondary}-weight{weight}-c{constant}-w{window}"
                    configs[name] = execution.config_for(
                        lexical, secondary=secondary, weight=weight, constant=constant, window=window
                    )
                    family[name], _ = execution.evaluate_config(configs[name], development, pools)
        development_reports.update(family)
        family_best = best_name(family)
        finalists[secondary] = configs[family_best]
        if progress:
            print(f"Completed development family {secondary}: {family_best}", file=sys.stderr, flush=True)
    return development_reports, finalists


def select(
    train: datasets.Dataset, test: datasets.Dataset, *, seed: int = 42, progress: bool = False
) -> reporting.JsonObject:
    """Develop candidates, choose on validation and seal the comparison artifact.

    Args:
        train: Training labels and corpus for development/validation.
        test: Same corpus and query text for overlap exclusion, not test scoring.
        seed: Deterministic training-group partition seed.
        progress: Emit explicit local progress to stderr.

    Returns:
        JSON selection payload including its canonical checksum.
        No files are written and no test labels choose parameters.
    """
    development, validation, partition = partition_training(train, test, seed=seed)
    routes, builds = execution.build_routes(train.documents, execution.route_definitions(), progress=progress)
    pools, latencies = execution.cache_pools(routes, development + validation, progress=progress)
    development_reports, finalists = _develop_candidates(routes, development, pools, progress=progress)
    validation_reports = {
        name: execution.evaluate_config(config, validation, pools)[0] for name, config in finalists.items()
    }
    selected_name = choose_validation(validation_reports)
    comparisons = {
        "baseline": execution.config_for(execution.BASELINE),
        "legacy-lsa": execution.config_for("lsa"),
        "legacy-equal": execution.config_for(execution.BASELINE, secondary="lsa", window=50),
        "tuned-bm25": finalists["lexical"],
        "word-only": execution.config_for("word"),
        "char-only": execution.config_for("char"),
        "selected": finalists[selected_name],
    }
    report = {
        "schema": SCHEMA,
        "corpus_sha256": datasets.dataset_identity(train),
        "dataset_provenance": train.provenance,
        "partition": partition,
        "routes": execution.route_definitions(),
        "environment": reporting.environment(),
        "development": development_reports,
        "validation": validation_reports,
        "selected_family": selected_name,
        "selection_rule": "max validation nDCG, Recall, MRR; nDCG gain >=.005 and Recall loss <=.01; else baseline",
        "comparisons": {name: dataclasses.asdict(config) for name, config in comparisons.items()},
        "build": builds,
        "route_latency_ms": latencies,
        "latency_scope": "One route at a time, query encoding/filtering/top200; candidates reuse cached ranks.",
        "test_exposure": "Historical test has been inspected in spec 0003; not a pristine unseen holdout.",
    }
    # Seal the full payload, including observed build/route timings. A fresh run
    # may have identical rankings yet a different checksum. The CLI writes this
    # artifact before the separate evaluate command scores test judgments.
    report["lock_sha256"] = reporting.digest(report)
    return report


def validate_lock(
    selection: reporting.JsonObject, train: datasets.Dataset, test: datasets.Dataset
) -> dict[str, hybrid.SearchConfig]:
    """Verify checksum, versions, corpus and partition before reading configs.

    The hash is an integrity check, not authentication. Version checks cover the
    recorded interpreter/numeric packages and route definitions, not a source-code
    fingerprint. Old artifacts require a compatible implementation as well.

    Raises:
        ValueError: The stored experiment identity no longer matches the inputs.
    """
    payload = {key: value for key, value in selection.items() if key != "lock_sha256"}
    if selection.get("schema") != SCHEMA or selection.get("lock_sha256") != reporting.digest(payload):
        raise ValueError("Selection artifact checksum/schema mismatch")
    if selection["routes"] != execution.route_definitions() or selection["environment"] != reporting.environment():
        raise ValueError("Selection implementation/environment mismatch; rerun selection explicitly")
    if selection["corpus_sha256"] != datasets.dataset_identity(test):
        raise ValueError("Selection corpus mismatch")
    _, _, partition = partition_training(train, test, seed=selection["partition"]["seed"])
    if partition != selection["partition"] or train.provenance != selection["dataset_provenance"]:
        raise ValueError("Selection dataset/partition mismatch")
    return {name: execution.read_config(value) for name, value in selection["comparisons"].items()}


def evaluate(
    selection: reporting.JsonObject, train: datasets.Dataset, test: datasets.Dataset, *, progress: bool = False
) -> reporting.JsonObject:
    """Run frozen test comparisons and real baseline/selected pipelines.

    Args:
        selection: Previously sealed selection artifact.
        train: Original training data, used to verify the lock.
        test: Historical test judgments on the selected corpus.
        progress: Emit build/retrieval progress to stderr.

    Returns:
        JSON report with per-query ranks, metrics, paired statistics and timings.

    Raises:
        ValueError: Selection validation or cached/live rank parity fails.
    """
    configs = validate_lock(selection, train, test)
    names = {branch.name for config in configs.values() for branch in config.branches}
    routes, builds = execution.build_routes(test.documents, names, progress=progress)
    pools, latencies = execution.cache_pools(routes, test.cases, progress=progress)
    reports, rankings = {}, {}
    for name, config in configs.items():
        reports[name], rankings[name] = execution.evaluate_config(config, test.cases, pools)
    live = {
        name: execution.live_pipeline(configs[name], test, routes, rankings[name]) for name in ("baseline", "selected")
    }
    return {
        "selection_lock_sha256": selection["lock_sha256"],
        "selected_family": selection["selected_family"],
        "dataset": {**test.provenance, "documents": len(test.documents), "queries": len(test.cases)},
        "environment": reporting.environment(),
        "comparisons": reports,
        "selected_vs_baseline": reporting.paired_comparison(test.cases, rankings["baseline"], rankings["selected"]),
        "selected_vs_tuned_bm25": reporting.paired_comparison(test.cases, rankings["tuned-bm25"], rankings["selected"]),
        "ranking_sha256": reporting.digest(rankings),
        "build": builds,
        "route_latency_ms": latencies,
        "latency_scope": selection["latency_scope"],
        "live_pipeline": live,
        "per_query": [
            {
                "query_id": case.query_id,
                "query": case.query,
                "relevant_source_ids": [test.source_ids[index] for index in sorted(case.relevant_asset_ids)],
                "top10_source_ids": {
                    name: [test.source_ids[index] for index in ranking[case.query_id]]
                    for name, ranking in rankings.items()
                },
                "metrics": {
                    name: metrics.metrics_for_case(case, ranking[case.query_id]) for name, ranking in rankings.items()
                },
            }
            for case in test.cases
        ],
        "test_exposure": selection["test_exposure"],
    }
