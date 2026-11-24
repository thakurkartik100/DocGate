# Interview notes

## Before any interview: do these three things yourself
1. Run `pytest`, `python eval/run_eval.py --gate`, start the server and call it. If you cannot demo it, the project does not help you.
2. Run `docker compose up --build` once. The Docker files were written but never executed. Only list Docker on your resume after this works.
3. Read `src/docgate/` end to end (about 400 lines). An interviewer will ask you to explain any line. Also read the 88 questions in `data/golden.jsonl` and fix any label you disagree with, then re-run the eval and update the numbers in the README.

## 30-second pitch
"I built a search service over the Python standard library docs. It combines keyword and embedding search, reranks the top candidates with a cross-encoder, and refuses to answer out-of-scope questions. The part I'm proudest of is the evaluation: I wrote a labelled test set, measured each stage separately, and wired the metrics into CI so a quality regression fails the build."

## Resume bullets (adjust once you have verified them yourself)
- Built a hybrid retrieval service (BM25 + MiniLM embeddings + Reciprocal Rank Fusion + cross-encoder reranking) over 3,096 stdlib API docs; raised Recall@5 from 0.51 (BM25) to 0.69 and MRR@10 from 0.34 to 0.48.
- Designed an offline evaluation harness (Recall@k, MRR, nDCG, abstention F1 on a held-out split) and a CI quality gate that fails on retrieval regressions.
- Cut reranker latency 5.6x (850 ms to 150 ms p50) by sweeping candidate count, which also improved Recall@5 by 4 points.
- Served via FastAPI with validation, request IDs, JSON logging, Prometheus metrics and an LRU/TTL cache; load-tested to about 590 req/s cached and 6 req/s uncached on one CPU worker, and identified the CPU-bound saturation point.

## Questions to expect, and honest answers
**Why hybrid and not just embeddings?** Measured: dense alone had Recall@10 of 0.78, hybrid 0.82. It helps recall of the candidate pool; the reranker then fixes ordering. At Recall@5 the gain over dense was not significant on my set.

**Why is Recall@5 only 0.69?** Three reasons, all in `docs/evaluation.md`: terse docstrings, near-duplicate chunks that the labels count as wrong, and a reranker trained on web text. The labels are strict (one canonical chunk per question), so real recall is higher.

**How do you know it's good?** I don't claim it is state of the art. I have a baseline and a regression gate. The set is 88 questions written by me, which is a limitation.

**What did the evaluation change?** Three decisions: candidate count (30 to 10), keeping the module-context prefix (helps BM25, neutral overall), and the abstention threshold.

**What happens at 100x the corpus?** The numpy dot product becomes a bottleneck, so I would move to an ANN index; the inverted index for BM25 would move to a search engine like OpenSearch; embeddings would be computed in a batch job.

**What breaks under load?** One worker saturates at about 6 uncached requests/s because inference is CPU-bound and serialised. Fixes I would test: more workers (memory cost), a smaller reranker, batching, a GPU.

**What would you add next?** LLM answer generation with citations and a faithfulness metric; fine-tuning the reranker on stdlib question/chunk pairs; a larger, independently labelled test set; OpenTelemetry tracing.

**Something that went wrong?** My first load test reported a 5 ms median at 8 clients because the generator repeated queries and hit the cache. I noticed the number was implausible, found the cause, and added a per-run nonce.

## Job-posting keywords this project covers
From postings found while researching (examples: [Senior AI Engineer, Agentic and RAG Systems](https://www.dice.com/job-detail/ed8dc926-11d1-4b9f-9cf6-553c64174019), [AI Engineer JD guide](https://resources.rework.com/vi/libraries/job-description-templates/ai-engineer)):

| Keyword in postings | Where it appears here |
|---|---|
| RAG, retrieval, hybrid search, reranking | `retrieval/` |
| Evaluation, golden dataset, regression testing, quality thresholds | `eval/`, `data/golden.jsonl` |
| Python, FastAPI, REST APIs | `api.py` |
| CI/CD with eval gates | `.github/workflows/ci.yml` (runs on every push, passing) |
| Docker | `Dockerfile` (written, not yet built) |
| Observability, monitoring | Prometheus metrics, JSON logs, request IDs |
| Performance profiling, latency | sweep and load test |
| Not covered | LLM generation, LangSmith/Langfuse/OpenTelemetry, Kubernetes, cloud deployment |
