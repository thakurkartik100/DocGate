"""FastAPI service: POST /query returns cited passages or abstains; /health, /metrics (Prometheus)."""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .cache import TTLCache
from .corpus import load
from .retrieval.pipeline import Retriever

DATA = Path(os.environ.get("DOCGATE_DATA", Path(__file__).resolve().parents[2] / "data"))
log = logging.getLogger("docgate")
logging.basicConfig(level=logging.INFO, format="%(message)s")  # one JSON object per line

REQS = Counter("docgate_requests_total", "Queries", ["outcome"])  # answered | abstained | cache_hit
LAT = Histogram("docgate_query_seconds", "End-to-end query latency", buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5))


class Query(BaseModel):
    q: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


class Passage(BaseModel):
    id: str
    score: float
    snippet: str


class Answer(BaseModel):
    request_id: str
    answered: bool
    passages: list[Passage]
    cached: bool = False
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = Retriever(load(DATA / "corpus.jsonl"), cache_dir=DATA / ".cache")
    app.state.threshold = json.loads((DATA / "abstain_threshold.json").read_text())["threshold"]
    app.state.cache = TTLCache()
    yield


app = FastAPI(title="DocGate", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(app.state.retriever.chunks)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# def (not async def): retrieval is CPU-bound, so FastAPI runs it in its threadpool instead of blocking the event loop.
@app.post("/query", response_model=Answer)
def query(body: Query, request: Request):
    t0, rid = time.perf_counter(), request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    key = (body.q.strip().lower(), body.k)
    cached = app.state.cache.get(key)
    if cached is not None:
        passages, answered, outcome = cached, bool(cached), "cache_hit"
    else:
        hits = app.state.retriever.search(body.q, k=body.k)
        answered = bool(hits) and hits[0].score >= app.state.threshold
        passages = [Passage(id=h.id, score=round(h.score, 3), snippet=h.text[:300]) for h in hits] if answered else []
        app.state.cache.put(key, passages)
        outcome = "answered" if answered else "abstained"
    dt = time.perf_counter() - t0
    REQS.labels(outcome).inc()
    LAT.observe(dt)
    log.info(json.dumps({"event": "query", "request_id": rid, "outcome": outcome, "top": passages[0].id if passages else None, "ms": round(dt * 1000, 1)}))
    return Answer(request_id=rid, answered=answered, passages=passages, cached=outcome == "cache_hit", latency_ms=round(dt * 1000, 1))
