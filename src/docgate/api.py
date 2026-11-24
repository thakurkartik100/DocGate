"""FastAPI service: POST /query returns cited passages or abstains; /health, /metrics (Prometheus)."""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .cache import TTLCache
from . import generate
from .corpus import load
from .retrieval.pipeline import Retriever

DATA = Path(os.environ.get("DOCGATE_DATA", Path(__file__).resolve().parents[2] / "data"))
log = logging.getLogger("docgate")
logging.basicConfig(level=logging.INFO, format="%(message)s")  # one JSON object per line

REQS = Counter("docgate_requests_total", "Queries", ["outcome"])  # answered | abstained | cache_hit
LAT = Histogram("docgate_query_seconds", "End-to-end query latency", buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5))
GEN_LAT = Histogram("docgate_generation_seconds", "LLM answer-generation latency", buckets=(.5, 1, 2, 4, 8, 16, 32))


class Query(BaseModel):
    q: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)
    generate: bool = False  # also write an LLM answer grounded in the passages (needs Anthropic credentials)


class Passage(BaseModel):
    id: str
    score: float
    snippet: str


class Answer(BaseModel):
    request_id: str
    answered: bool
    passages: list[Passage]
    answer: str | None = None
    citations: list[str] = []  # chunk ids the generated answer cites
    cached: bool = False
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = Retriever(load(DATA / "corpus.jsonl"), cache_dir=DATA / ".cache")
    app.state.threshold = json.loads((DATA / "abstain_threshold.json").read_text())["threshold"]
    app.state.cache = TTLCache()
    yield


app = FastAPI(title="DocGate", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def demo():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


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
    key = (body.q.strip().lower(), body.k, body.generate)
    cached = app.state.cache.get(key)
    if cached is not None:
        (passages, gen), outcome = cached, "cache_hit"
        answered = bool(passages)
    else:
        hits = app.state.retriever.search(body.q, k=body.k)
        answered = bool(hits) and hits[0].score >= app.state.threshold
        passages = [Passage(id=h.id, score=round(h.score, 3), snippet=h.text[:300]) for h in hits] if answered else []
        gen = _generate(body.q, hits) if answered and body.generate else None
        app.state.cache.put(key, (passages, gen))
        outcome = "answered" if answered else "abstained"
    dt = time.perf_counter() - t0
    REQS.labels(outcome).inc()
    LAT.observe(dt)
    log.info(json.dumps({"event": "query", "request_id": rid, "outcome": outcome, "top": passages[0].id if passages else None, "ms": round(dt * 1000, 1)}))
    return Answer(request_id=rid, answered=answered, passages=passages, answer=gen and gen["text"],
                  citations=gen["citations"] if gen else [], cached=outcome == "cache_hit", latency_ms=round(dt * 1000, 1))


def _generate(question, hits) -> dict:
    t0 = time.perf_counter()
    try:
        gen = generate.answer(question, [(h.id, h.text) for h in hits])
    except TypeError as e:  # the SDK raises TypeError when no Anthropic credentials are configured
        if "authentication" not in str(e):
            raise
        raise HTTPException(503, "Answer generation is not configured: set ANTHROPIC_API_KEY.") from e
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        raise HTTPException(502, f"Answer generation failed upstream: {type(e).__name__}") from e
    GEN_LAT.observe(time.perf_counter() - t0)
    return gen
