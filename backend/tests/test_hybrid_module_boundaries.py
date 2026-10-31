"""Compatibility and import boundaries of the Google-derived module refactor."""

from __future__ import annotations

import subprocess
import sys

import pytest

from backend.app.evaluation import hybrid_benchmark, hybrid_optimize
from backend.app.evaluation.hybrid_eval import benchmark, dataset, encoders, workflow
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import adapters, documents, lexical, vector


def test_legacy_imports_resolve_to_the_same_public_implementations():
    assert adapters.CorpusDocument is documents.CorpusDocument
    assert adapters.MemoryDocumentStore is documents.MemoryDocumentStore
    assert adapters.Bm25Retriever is lexical.Bm25Retriever
    assert adapters.ExactVectorRetriever is vector.ExactVectorRetriever
    assert adapters.tokenize is lexical.tokenize
    assert hybrid_benchmark.Dataset is hybrid_optimize.Dataset is dataset.Dataset
    assert hybrid_benchmark.load_dataset is hybrid_optimize.load_dataset is dataset.load_dataset
    assert hybrid_benchmark.LatentSemanticEncoder is encoders.LatentSemanticEncoder
    assert hybrid_benchmark.run is benchmark.run
    assert hybrid_optimize.select is workflow.select
    assert hybrid_optimize.evaluate is workflow.evaluate


def test_evaluation_library_imports_without_cli_or_optional_runtime_dependencies():
    code = """
import sys
from backend.app.evaluation.hybrid_eval import benchmark, execution, workflow
assert 'backend.app.evaluation.hybrid_benchmark' not in sys.modules
assert 'backend.app.evaluation.hybrid_optimize' not in sys.modules
assert not any(name.startswith(('numpy', 'scipy', 'sklearn', 'torch',
    'sqlalchemy', 'backend.app.core', 'backend.app.db')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-S", "-c", code], check=True, timeout=10)


@pytest.mark.parametrize("stage", ["retriever", "store"])
def test_provider_exception_causes_survive_orchestration_refactor(stage):
    failure = OSError("provider-specific failure")

    class BrokenRetriever:
        def retrieve(self, query, limit):
            raise failure

    class BrokenStore:
        def fetch(self, ids):
            raise failure

    retriever = BrokenRetriever() if stage == "retriever" else adapters.RankedListRetriever([1])
    store = BrokenStore() if stage == "store" else None
    engine = hybrid.HybridSearch({"text": retriever}, store=store)
    expected = hybrid.RetrievalError if stage == "retriever" else hybrid.HydrationError
    with pytest.raises(expected) as caught:
        engine.search(hybrid.SearchQuery("query"), hybrid.SearchConfig((hybrid.Branch("text"),)))
    assert caught.value.__cause__ is failure
