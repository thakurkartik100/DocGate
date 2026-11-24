# DocGate

[![ci](https://github.com/thakurkartik100/DocGate/actions/workflows/ci.yml/badge.svg)](https://github.com/thakurkartik100/DocGate/actions/workflows/ci.yml)

A search service for the Python standard-library documentation. It answers a plain-English question with ranked, cited passages, or says "I don't know" when the question is out of scope. It is built the way a production retrieval service is: hybrid search, a measured evaluation harness, a CI gate that fails on quality regressions, metrics, caching and a load test.

The point of the project is not the search itself. It is the engineering around it: every design choice below is backed by a number from `eval/`.

## Architecture

```
            query
              |
   +----------+-----------+
   v                      v
 BM25 (from scratch)   Dense (MiniLM embeddings, cosine)
   |                      |
   +-----> RRF fusion <---+          top-10 candidates
                |
        Cross-encoder rerank
                |
     top-1 score >= threshold ?  --no--> abstain (empty result)
                | yes
          cited passages  ->  TTL/LRU cache -> JSON log + Prometheus metrics
                |
   optional: Claude writes a short answer citing chunk ids (generate=true)
```

- **Corpus** (`src/docgate/corpus.py`): 3,096 chunks, one per public function, class or method in 66 stdlib modules, snapshotted to `data/corpus.jsonl`. Each chunk is prefixed with its module's one-line summary (a cheap form of contextual retrieval).
- **Retrieval** (`src/docgate/retrieval/`): BM25 implemented from scratch with an identifier-aware tokenizer; dense retrieval with `all-MiniLM-L6-v2`; Reciprocal Rank Fusion; `ms-marco-MiniLM-L-6-v2` cross-encoder reranking.
- **Abstention**: the top reranker score is compared to a threshold calibrated on a dev split and reported on a held-out test split.
- **Generation** (`src/docgate/generate.py`): with `"generate": true`, Claude (`claude-opus-5-5`) answers from the retrieved chunks only, using the API's native citations so each answer lists the chunk ids it relied on. Abstained queries never reach the LLM.
- **Service** (`src/docgate/api.py`): FastAPI, input validation, request IDs, structured JSON logs, Prometheus `/metrics`, `/health`.

## Results

73 answerable and 15 unanswerable questions, hand-written (`data/golden.jsonl`). Full analysis in [docs/evaluation.md](docs/evaluation.md).

| Retrieval mode | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|
| BM25 | 0.219 | 0.507 | 0.644 | 0.342 | 0.410 | 2 ms |
| Dense | 0.260 | 0.644 | 0.781 | 0.428 | 0.508 | 8 ms |
| Hybrid (RRF) | 0.260 | 0.630 | 0.822 | 0.425 | 0.513 | 11 ms |
| **Hybrid + rerank** | **0.342** | **0.685** | **0.822** | **0.484** | **0.557** | 136 ms |

- Reranking only 10 candidates beats reranking 30 or 50 on quality and is 5.6x faster than 30.
- Abstention F1 on held-out questions: 0.80 (44 questions).
- Load test, single worker, CPU only: about 6 requests/s for uncached queries, about 590 requests/s when cached. Throughput does not scale with concurrency (see evaluation doc).

These numbers come from a small, author-written test set. Read them as a regression baseline, not a benchmark.

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q                                   # 16 tests, no model download or API key needed
python eval/run_eval.py --gate              # full evaluation + quality gate (first run downloads ~150 MB of models)
python -m uvicorn docgate.api:app --port 8000
curl -X POST localhost:8000/query -H 'content-type: application/json' \
     -d '{"q": "split a string using a regular expression", "k": 3}'
python loadtest/loadtest.py --concurrency 4 --seconds 20
```

Or with Docker: `docker compose up --build`. CI builds the image on every push, starts the container, and checks one answered and one abstained query against it. (On some networks pip inside the build container times out reaching PyPI; the Dockerfile raises pip's timeout and retries, and CI is the reference build.)

## Layout

```
src/docgate/        corpus.py, cache.py, api.py, retrieval/{bm25,dense,fusion,pipeline}.py
eval/               run_eval.py (metrics, ablation, abstention, gate), sweep.py (candidate count)
data/               corpus.jsonl, golden.jsonl, abstain_threshold.json
loadtest/           loadtest.py and recorded results
tests/              unit tests and API tests
docs/               design.md, evaluation.md, interview-notes.md
.github/workflows/  ci.yml: tests, then the eval gate, then results uploaded as an artifact
```

## Known limitations

- Answer generation is implemented and unit-tested, but has not yet been evaluated on real API calls. `eval/run_generation_eval.py` measures citation accuracy and faithfulness; it needs `ANTHROPIC_API_KEY` and costs a little money to run.
- Test set is small and was written by one author, so labels favour one canonical answer per question (e.g. `re.split` is marked correct, `re.Pattern.split` is not). Recall is therefore understated.
- Single-process cache and a single worker. Not load-balanced, no authentication, no rate limiting.
