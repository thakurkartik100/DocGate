"""Closed-loop load test: C concurrent clients each send queries back-to-back for D seconds.

python loadtest/loadtest.py --url http://localhost:8000 --concurrency 8 --seconds 20 [--warm]
--warm replays a small query set (cache hits); default sends unique queries (cache misses = worst case).
"""
import argparse
import asyncio
import itertools
import json
import statistics
import time
import uuid
from pathlib import Path

import httpx

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden.jsonl"


async def worker(client, queries, stop, lat, errors):
    while time.monotonic() < stop:
        t0 = time.perf_counter()
        try:
            r = await client.post("/query", json={"q": next(queries)})
            r.raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
        except httpx.HTTPError:
            errors.append(1)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--warm", action="store_true")
    a = ap.parse_args()
    base = [json.loads(line)["q"] for line in open(GOLDEN, encoding="utf-8")]
    nonce = uuid.uuid4().hex[:6]  # per-run, so reruns do not hit entries cached by earlier runs
    # unique suffix defeats the cache so every request runs the full pipeline
    queries = itertools.cycle(base) if a.warm else (f"{base[n % len(base)]} #{nonce}{n}" for n in itertools.count())
    lat, errors = [], []
    async with httpx.AsyncClient(base_url=a.url, timeout=60) as client:
        await client.post("/query", json={"q": "warmup query"})
        stop = time.monotonic() + a.seconds
        await asyncio.gather(*(worker(client, queries, stop, lat, errors) for _ in range(a.concurrency)))
    q = statistics.quantiles(lat, n=100)
    print(json.dumps({"concurrency": a.concurrency, "warm_cache": a.warm, "requests": len(lat), "errors": len(errors),
                      "rps": round(len(lat) / a.seconds, 1), "p50_ms": round(q[49]), "p95_ms": round(q[94]), "p99_ms": round(q[98])}))


asyncio.run(main())
