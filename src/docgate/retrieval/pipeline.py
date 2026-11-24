"""Retrieval pipeline: BM25 + dense -> RRF -> cross-encoder rerank -> abstain if not confident."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .dense import Dense
from .fusion import rrf

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    id: str
    score: float
    text: str


class Retriever:
    def __init__(self, chunks: list[dict], cache_dir: Path | None = None, load_dense: bool = True, load_reranker: bool = True):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.bm25 = BM25(texts)
        self.dense = Dense(texts, cache_dir) if load_dense else None
        self.reranker = None
        if load_reranker:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(RERANKER)

    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank", candidates: int = 10) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "bm25":
            scores = self.bm25.scores(query)
        elif mode == "dense":
            scores = self.dense.scores(query)
        else:
            scores = rrf([self.bm25.scores(query), self.dense.scores(query)])
        top = np.argsort(-scores)[: candidates if mode == "hybrid_rerank" else k]
        if mode == "hybrid_rerank":
            ce = self.reranker.predict([(query, self.chunks[i]["text"]) for i in top])
            order = np.argsort(-ce)[:k]
            return [Hit(self.chunks[top[j]]["id"], float(ce[j]), self.chunks[top[j]]["text"]) for j in order]
        return [Hit(self.chunks[i]["id"], float(scores[i]), self.chunks[i]["text"]) for i in top]
