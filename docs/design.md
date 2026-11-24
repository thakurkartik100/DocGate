# Design decisions

Short architecture decision records. Each says what was chosen, why, and what it costs.

## 1. Corpus: Python standard-library docstrings
**Why:** real technical text that anyone can reproduce offline from a Python install, with no scraping, licensing or API key. Chunk granularity is one function, class or method, so an answer is a specific API.
**Cost:** many docstrings are terse, and C-implemented functions need special handling (their `__module__` is a private module like `_heapq`, so `corpus.py` also accepts names from `__all__`). The corpus is snapshotted to `data/corpus.jsonl` so results do not change with the Python version.

## 2. BM25 written from scratch
**Why:** about 40 lines, no dependency, and it lets the tokenizer be domain-specific: it splits `snake_case`, dotted names and camelCase so `os.path.getsize` matches "get size".
**Cost:** no stemming and a plain inverted index; fine at 3k documents, not at millions.

## 3. Dense retrieval with MiniLM and a numpy dot product
**Why:** at 3k vectors an exact scan takes under 10 ms, so a vector database or ANN index (FAISS, HNSW) would add operations cost for no benefit. Embeddings are cached on disk keyed by a hash of the corpus.
**Cost:** latency grows linearly with corpus size. Past roughly 100k chunks an ANN index becomes worth it.

## 4. Reciprocal Rank Fusion
**Why:** BM25 scores and cosine similarities are on different scales. RRF combines ranks, so it needs no score normalisation and has one hyperparameter (k=60, the value from the original paper, not tuned here).
**Cost:** it ignores score magnitude, so a very confident match is treated the same as a weak first place.

## 5. Cross-encoder reranking of a small candidate set
**Why:** a cross-encoder reads query and passage together and is far more precise than comparing independent embeddings, but is too slow to run on the whole corpus. Reranking only the top candidates keeps cost bounded.
**Decision from data:** 10 candidates. See Experiment 3 in `evaluation.md`.

## 6. Abstain instead of always answering
**Why:** returning plausible but wrong passages for an out-of-scope question is worse than returning nothing. The reranker's top score is a usable confidence signal; the threshold is calibrated on a dev split and scored on a test split.
**Cost:** the threshold is specific to this reranker and corpus. If either changes, recalibrate (`python eval/run_eval.py --write-threshold`).

## 7. Service design
- Synchronous `def` endpoint: retrieval is CPU-bound, so FastAPI runs it on its threadpool instead of blocking the event loop.
- Models load once in the FastAPI lifespan hook, not per request.
- Input validated with Pydantic (query length, `k` range).
- Cache stores abstentions too, so repeated out-of-scope queries are cheap. Shortcut: in-process only; use Redis if running several workers.
- Every request gets a request ID (accepts `x-request-id` from upstream), one JSON log line, and counters/latency histograms at `/metrics`.

## 8. Quality gate in CI
**Why:** retrieval quality can regress silently (a tokenizer change, a new model, a corpus rebuild) while all unit tests stay green. Treating the eval like a test makes that visible in the pull request.
**Cost:** the eval takes about a minute and downloads models, so CI caches the Hugging Face directory.

## 9. Generation with native citations
**Why:** sending each chunk as a separate document block with citations enabled makes the API attach chunk ids to the sentences that use them, so grounding can be checked (did it cite the gold chunk?) rather than trusted. Generation only runs after retrieval passes the abstention threshold, so out-of-scope questions cost nothing.
**Choices:** `claude-opus-5-5` at low effort (short factual answers; keeps latency and cost down), server-side refusal fallbacks, 503 when no credentials, 502 on upstream errors.
**Cost:** adds seconds of latency and a per-query API cost, so it is opt-in per request (`generate: true`).

## Not done, on purpose
Authentication, rate limiting, multi-worker deployment, Kubernetes, tracing with OpenTelemetry. Each is a reasonable extension but none was needed to demonstrate the retrieval and evaluation work, and none has been tested.
