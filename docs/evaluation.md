# Evaluation

All numbers are produced by `python eval/run_eval.py` and `eval/sweep.py` (raw output in `eval/results.json`, `eval/results_plain.json`, `eval/sweep.json`). Machine: Windows, 24 logical CPUs, CPU-only PyTorch.

## Test set

- 73 answerable questions across 66 stdlib modules, each labelled with the chunk id(s) that answer it (`data/golden.jsonl`).
- 15 unanswerable questions: a mix of plainly off-topic ("best pizza recipe", "weather in Pune today") and technical but outside the stdlib ("merge two pandas DataFrames", "SQL window functions", "interactive git rebase").
- Questions are paraphrases of what the function does, not its name ("fold a sequence into a single value" for `functools.reduce`), so lexical search is genuinely tested.
- Written by the same author as the system, from knowledge of the stdlib and not from the retriever's output. That avoids the circularity of generating questions from chunks, but it is not an independent test set.

## Experiment 1: retrieval ablation

| Mode | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | p50 | p95 |
|---|---|---|---|---|---|---|---|
| BM25 | 0.219 | 0.507 | 0.644 | 0.342 | 0.410 | 2.3 ms | 4.1 ms |
| Dense | 0.260 | 0.644 | 0.781 | 0.428 | 0.508 | 7.9 ms | 8.5 ms |
| Hybrid RRF | 0.260 | 0.630 | 0.822 | 0.425 | 0.513 | 11.0 ms | 13.5 ms |
| Hybrid + rerank | 0.342 | 0.685 | 0.822 | 0.484 | 0.557 | 136 ms | 246 ms |

Findings:
- Dense beats BM25 by about 14 points at Recall@5. Queries describe behaviour in words the docstring does not share.
- RRF improves Recall@10 (0.781 to 0.822) but not Recall@5. Fusion widens the candidate pool; it does not order it better.
- The reranker is what lifts Recall@1 (0.26 to 0.34) and MRR, at roughly 12x the latency of hybrid alone.

## Experiment 2: module-context prefix on each chunk

| Corpus | BM25 R@5 | Dense R@5 | Hybrid + rerank R@5 | Hybrid + rerank MRR |
|---|---|---|---|---|
| Plain chunk text | 0.411 | 0.658 | 0.712 | 0.502 |
| With `[module: summary]` prefix | 0.507 | 0.644 | 0.685 | 0.484 |

The prefix helps BM25 by 10 points, which is real. For the full pipeline it is 2 questions worse out of 73, which is within noise on a set this size. I kept the prefix as the default because it helps the lexical leg, but the data does not show it helping the end result. A larger test set would be needed to decide.

## Experiment 3: how many candidates to rerank

| Candidates | R@5 | MRR@10 | p50 latency |
|---|---|---|---|
| 5 | 0.630 | 0.415 | 67 ms |
| **10** | **0.685** | 0.484 | 152 ms |
| 20 | 0.671 | **0.498** | 506 ms |
| 30 | 0.644 | 0.487 | 853 ms |
| 50 | 0.644 | 0.471 | 2,385 ms |

More candidates is not better: past 10 to 20 the general-purpose reranker starts promoting plausible-looking wrong chunks. Latency grows linearly with candidates. Default set to 10. (My first version reranked 30, at 800 ms p50. This sweep is what found the 5.6x speed-up.)

## Experiment 4: abstention

The top reranker score is compared to a threshold. The threshold maximises F1 on even-numbered questions (dev) and is scored on odd-numbered ones (test).

| Split | n | F1 for detecting unanswerable |
|---|---|---|
| Dev (calibration) | 44 | 0.833 |
| Test (held out) | 44 | 0.800 |

With only 15 unanswerable questions in total, about 7 per split, one question moves F1 by several points. Treat 0.80 as "works, not precisely measured".

## Experiment 5: load test

Single uvicorn worker, closed-loop clients, 20 s per run (`loadtest/`).

| Scenario | Clients | Requests/s | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| Cold (unique queries) | 1 | 5.7 | 164 ms | 334 ms | 355 ms | 0 |
| Cold | 4 | 7.0 | 541 ms | 935 ms | 1,038 ms | 0 |
| Cold | 8 | 6.5 | 1,241 ms | 1,875 ms | 2,147 ms | 0 |
| Warm (cached) | 8 | 591.8 | 5 ms | 17 ms | 52 ms | 0 |

Throughput saturates at about 6 to 7 requests/s from the first extra client: the model inference is CPU-bound and one process serialises it, so extra clients only add queueing delay. Caching is what makes repeated queries cheap. Untested ideas for scaling: several uvicorn workers (each loads its own models, so memory grows), batching reranker calls, or a smaller reranker.

A bug found by this test: my first load generator reused query strings between runs, so the "cold" run hit the cache and reported a misleading p50 of 5 ms. Fixed with a per-run nonce.

## Failure analysis (hybrid + rerank, 23 of 73 queries miss at rank 5)

Sampled from `eval/results.json`:
- **Terse or generic docstrings**: `math.gcd` ("Greatest Common Divisor.") loses to `math.floor`.
- **Sibling functions**: "compute a SHA-256 digest" returns `hashlib.sha3_256` and `hashlib.file_digest`; "encode binary data as base64" returns `base64.encode`, `encodebytes` and `urlsafe_b64encode` instead of `b64encode`. Some of these are arguably valid answers that the labels do not list, so measured recall is understated.
- **Semantic confusion**: "serialize an object into bytes" retrieves `sqlite3.Connection.serialize` instead of `pickle.dumps`.
- **Cross-encoder domain mismatch**: the reranker was trained on web passages, not API docs. Fine-tuning it on stdlib pairs is the main untried improvement.

## CI quality gate

`python eval/run_eval.py --gate` exits non-zero if the full pipeline falls below Recall@5 0.60, MRR@10 0.45 or abstention F1 0.75 (`GATES` in `eval/run_eval.py`). The thresholds sit a few points below the measured values so normal noise does not fail the build, but a real regression (for example breaking the tokenizer) does. The gate passes locally and in GitHub Actions on every push.
