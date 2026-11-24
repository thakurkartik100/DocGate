"""BM25 from scratch (Robertson/Sparck Jones, Okapi variant). No dependency, ~40 lines."""
import math
import re
from collections import Counter

import numpy as np

_SPLIT = re.compile(r"[A-Za-z]+|\d+")
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def tokenize(text: str) -> list[str]:
    """Lowercase; split snake_case, dotted names and camelCase so `os.path.getsize` matches 'get size'."""
    out = []
    for word in _SPLIT.findall(text):
        out.extend(t.lower() for t in _CAMEL.findall(word))
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        toks = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.dl = np.array([len(t) for t in toks], dtype=float)
        self.avgdl = self.dl.mean()
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, t in enumerate(toks):
            for term, tf in Counter(t).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n)
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, ()):
                s[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return s
