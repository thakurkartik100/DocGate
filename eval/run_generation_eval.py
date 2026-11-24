"""Generation evaluation: does the LLM answer cite the right chunk, and is every claim supported?

Costs real money (two Claude calls per question). Try a few first:
    python eval/run_generation_eval.py --limit 5
Metrics, over answerable questions where retrieval did not abstain:
    cites_gold    answer cites at least one gold chunk id
    cited         answer cites at least one chunk
    faithful      an LLM judge finds every claim supported by the retrieved passages
    declined      the model said the passages do not answer the question (no citations)
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docgate import generate  # noqa: E402
from docgate.corpus import load  # noqa: E402
from docgate.retrieval.pipeline import Retriever  # noqa: E402

PRICE_PER_MTOK = {"input": 4.00, "output": 20.00}  # claude-opus-5-5, USD
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "unsupported_claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["supported", "unsupported_claims"],
    "additionalProperties": False,
}


def judge(client, question, answer_text, passages) -> tuple[dict, dict]:
    docs = "\n\n".join(f"<doc id='{cid}'>\n{text}\n</doc>" for cid, text in passages)
    resp = client.beta.messages.create(
        model=generate.MODEL, max_tokens=16000, betas=[generate.FALLBACK_BETA], fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        system="You check whether an answer is fully supported by the given documents. A claim is supported only "
               "if the documents state it or it follows directly from them. General Python knowledge not in the "
               "documents counts as unsupported. List each unsupported claim verbatim.",
        messages=[{"role": "user", "content": f"{docs}\n\nQuestion: {question}\n\nAnswer to check:\n{answer_text}"}],
    )
    usage = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
    if resp.stop_reason == "refusal":
        return {"supported": None, "unsupported_claims": []}, usage
    return json.loads(next(b.text for b in resp.content if b.type == "text")), usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N answerable questions (0 = all)")
    args = ap.parse_args()
    golden = [g for g in map(json.loads, open(ROOT / "data" / "golden.jsonl", encoding="utf-8")) if g["answerable"]]
    golden = golden[: args.limit] if args.limit else golden
    threshold = json.loads((ROOT / "data" / "abstain_threshold.json").read_text())["threshold"]
    retr = Retriever(load(ROOT / "data" / "corpus.jsonl"), cache_dir=ROOT / "data" / ".cache")
    client = anthropic.Anthropic()

    rows, tokens, gen_ms = [], {"input": 0, "output": 0}, []
    for g in golden:
        hits = retr.search(g["q"], k=5)
        if not hits or hits[0].score < threshold:
            rows.append({"q": g["q"], "abstained": True})
            continue
        passages = [(h.id, h.text) for h in hits]
        t0 = time.perf_counter()
        try:
            out = generate.answer(g["q"], passages, client=client)
        except TypeError as e:
            if "authentication" in str(e):
                sys.exit("No Anthropic credentials: set ANTHROPIC_API_KEY first.")
            raise
        gen_ms.append((time.perf_counter() - t0) * 1000)
        verdict, jusage = judge(client, g["q"], out["text"], passages) if out["text"] else ({"supported": None, "unsupported_claims": []}, {"input": 0, "output": 0})
        for k in tokens:
            tokens[k] += out["usage"][k] + jusage[k]
        rows.append({"q": g["q"], "gold": g["relevant"], "answer": out["text"], "citations": out["citations"],
                     "refused": out["refused"], "supported": verdict["supported"], "unsupported_claims": verdict["unsupported_claims"]})
        print(f"{len(rows):>3} cites_gold={bool(set(g['relevant']) & set(out['citations']))!s:<5} faithful={verdict['supported']}  {g['q'][:60]}")

    done = [r for r in rows if not r.get("abstained")]
    n = len(done) or 1
    summary = {
        "n_questions": len(rows), "n_answered": len(done), "n_abstained": len(rows) - len(done),
        "cites_gold": round(sum(bool(set(r["gold"]) & set(r["citations"])) for r in done) / n, 3),
        "cited": round(sum(bool(r["citations"]) for r in done) / n, 3),
        "faithful": round(sum(r["supported"] is True for r in done) / n, 3),
        "declined": round(sum(bool(r["answer"]) and not r["citations"] for r in done) / n, 3),
        "refused": sum(r["refused"] for r in done),
        "generation_p50_ms": round(statistics.median(gen_ms)) if gen_ms else None,
        "tokens": tokens,
        "cost_usd": round(sum(tokens[k] * PRICE_PER_MTOK[k] for k in tokens) / 1e6, 3),
    }
    (ROOT / "eval" / "generation_results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
