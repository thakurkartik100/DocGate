import numpy as np
import pytest

from docgate.cache import TTLCache
from docgate.retrieval.bm25 import BM25, tokenize
from docgate.retrieval.fusion import rrf
from eval.run_eval import abstain_f1, metrics


def test_tokenize_splits_identifiers():
    assert tokenize("os.path.getSize snake_case HTTPServer2") == ["os", "path", "get", "size", "snake", "case", "http", "server", "2"]


def test_bm25_ranks_matching_doc_first_and_ignores_unknown_terms():
    docs = ["json dumps serialize object", "heap queue push", "regex substitute pattern"]
    s = BM25(docs).scores("serialize json")
    assert s.argmax() == 0 and s[1] == s[2] == 0
    assert BM25(docs).scores("zzzz").sum() == 0


def test_rrf_prefers_doc_ranked_well_by_both():
    a = np.array([0.9, 0.8, 0.1])  # ranks doc0 > doc1 > doc2
    b = np.array([0.1, 0.9, 0.8])  # ranks doc1 > doc2 > doc0
    assert rrf([a, b]).argmax() == 1


def test_metrics_known_values():
    m = metrics([["a", "b", "c"], ["x", "y", "z"]], [["b"], ["q"]])
    assert m["recall@1"] == 0 and m["recall@5"] == 0.5 and m["mrr@10"] == 0.25
    assert metrics([["a"]], [["a"]])["ndcg@10"] == 1.0


def test_abstain_f1_perfect_separation():
    assert abstain_f1([5, 6, -9, -8], [True, True, False, False], 0) == 1.0


def test_cache_ttl_and_lru(monkeypatch):
    c = TTLCache(size=2, ttl=10)
    c.put("a", 1), c.put("b", 2), c.put("c", 3)
    assert c.get("a") is None and c.get("c") == 3
    c.data["c"] = (c.data["c"][0] - 11, 3)
    assert c.get("c") is None


def test_unknown_mode_rejected():
    from docgate.retrieval.pipeline import Retriever

    r = Retriever.__new__(Retriever)
    with pytest.raises(ValueError):
        r.search("x", mode="nope")
