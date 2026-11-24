"""Latency/quality trade-off: how many RRF candidates should the cross-encoder rerank?"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import metrics  # noqa: E402
from docgate.corpus import load  # noqa: E402
from docgate.retrieval.pipeline import Retriever  # noqa: E402

golden = [g for g in map(json.loads, open(ROOT / "data" / "golden.jsonl", encoding="utf-8")) if g["answerable"]]
retr = Retriever(load(ROOT / "data" / "corpus.jsonl"), cache_dir=ROOT / "data" / ".cache")
retr.search("warmup")
rows = {}
for n in (5, 10, 20, 30, 50):
    ranked, ms = [], []
    for g in golden:
        t0 = time.perf_counter()
        ranked.append([h.id for h in retr.search(g["q"], k=10, candidates=n)])
        ms.append((time.perf_counter() - t0) * 1000)
    m = metrics(ranked, [g["relevant"] for g in golden])
    rows[n] = {"recall@5": m["recall@5"], "mrr@10": m["mrr@10"], "p50_ms": round(statistics.median(ms))}
    print(n, rows[n])
(ROOT / "eval" / "sweep.json").write_text(json.dumps(rows, indent=2))
