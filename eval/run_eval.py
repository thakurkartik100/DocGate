"""Offline evaluation: ablation of retrieval modes, abstention calibration on a dev/test split, CI gate.

Run:  python eval/run_eval.py [--gate] [--write-threshold]
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docgate.corpus import load  # noqa: E402
from docgate.retrieval.pipeline import MODES, Retriever  # noqa: E402

KS = (1, 5, 10)
# CI fails if the full pipeline regresses below these (set a few points under the measured values; see docs/evaluation.md).
GATES = {"recall@5": 0.60, "mrr@10": 0.45, "abstain_f1": 0.75}


def metrics(ranked: list[list[str]], relevant: list[list[str]]) -> dict:
    out = {f"recall@{k}": 0.0 for k in KS} | {"mrr@10": 0.0, "ndcg@10": 0.0}
    for ids, rel in zip(ranked, relevant):
        rel = set(rel)
        for k in KS:
            out[f"recall@{k}"] += bool(rel & set(ids[:k]))
        for r, i in enumerate(ids[:10], 1):
            if i in rel:
                out["mrr@10"] += 1 / r
                break
        dcg = sum(1 / math.log2(r + 1) for r, i in enumerate(ids[:10], 1) if i in rel)
        idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(rel), 10) + 1))
        out["ndcg@10"] += dcg / idcg
    return {k: round(v / len(ranked), 3) for k, v in out.items()}


def best_threshold(scores: list[float], answerable: list[bool]) -> float:
    """Threshold on top-1 score maximising F1 of 'answer' vs 'abstain' (abstain = negative class)."""
    best, best_f1 = min(scores), -1.0
    for t in sorted(set(scores)):
        f1 = abstain_f1(scores, answerable, t)
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best


def abstain_f1(scores, answerable, t) -> float:
    """F1 of the abstain decision (abstain when top score < t); positives are unanswerable queries."""
    tp = sum(s < t and not a for s, a in zip(scores, answerable))
    fp = sum(s < t and a for s, a in zip(scores, answerable))
    fn = sum(s >= t and not a for s, a in zip(scores, answerable))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--corpus", default="corpus.jsonl", help="file under data/ (corpus_plain.jsonl = no module context)")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--write-threshold", action="store_true")
    args = ap.parse_args()

    golden = [json.loads(line) for line in open(ROOT / "data" / "golden.jsonl", encoding="utf-8")]
    ans = [g for g in golden if g["answerable"]]
    retr = Retriever(load(ROOT / "data" / args.corpus), cache_dir=ROOT / "data" / ".cache")
    retr.search("warmup", mode="hybrid_rerank")

    results, latency = {}, {}
    for mode in MODES:
        ranked, times = [], []
        for g in ans:
            t0 = time.perf_counter()
            hits = retr.search(g["q"], k=10, mode=mode)
            times.append((time.perf_counter() - t0) * 1000)
            ranked.append([h.id for h in hits])
        results[mode] = metrics(ranked, [g["relevant"] for g in ans])
        latency[mode] = {"p50_ms": round(statistics.median(times), 1), "p95_ms": round(sorted(times)[int(0.95 * len(times)) - 1], 1)}
        if mode == "hybrid_rerank":
            misses = [{"q": g["q"], "expected": g["relevant"], "got": r[:3]} for g, r in zip(ans, ranked) if not set(g["relevant"]) & set(r[:5])]

    # Abstention: calibrate on even rows (dev), report on odd rows (test).
    top1 = [retr.search(g["q"], k=1, mode="hybrid_rerank")[0].score for g in golden]
    flags = [g["answerable"] for g in golden]
    dev, test = slice(0, None, 2), slice(1, None, 2)
    thr = best_threshold(top1[dev], flags[dev])
    abstain = {"threshold": round(thr, 3), "dev_f1": round(abstain_f1(top1[dev], flags[dev], thr), 3),
               "abstain_f1": round(abstain_f1(top1[test], flags[test], thr), 3), "n_test": len(top1[test])}

    report = {"n_answerable": len(ans), "n_unanswerable": len(golden) - len(ans), "retrieval": results,
              "latency": latency, "abstention": abstain, "misses_at_5": misses}
    (ROOT / "eval" / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.write_threshold:
        (ROOT / "data" / "abstain_threshold.json").write_text(json.dumps({"threshold": abstain["threshold"]}))

    print(f"{'mode':<15}" + "".join(f"{m:>10}" for m in results["bm25"]) + "   p50ms  p95ms")
    for mode, r in results.items():
        print(f"{mode:<15}" + "".join(f"{v:>10}" for v in r.values()) + f"{latency[mode]['p50_ms']:>8}{latency[mode]['p95_ms']:>7}")
    print("abstention:", abstain, "| misses@5:", len(misses))

    if args.gate:
        got = results["hybrid_rerank"] | {"abstain_f1": abstain["abstain_f1"]}
        failed = {k: (got[k], v) for k, v in GATES.items() if got[k] < v}
        if failed:
            sys.exit(f"GATE FAILED (got, required): {failed}")
        print("gate passed")


if __name__ == "__main__":
    main()
