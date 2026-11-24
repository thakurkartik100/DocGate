import numpy as np


def rrf(score_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion (Cormack et al. 2009): sum of 1/(k+rank) over rankers.

    Rank-based, so BM25 and cosine scores need no normalisation to be combined.
    """
    fused = np.zeros(len(score_lists[0]))
    for s in score_lists:
        order = np.argsort(-s)
        fused[order] += 1.0 / (k + np.arange(1, len(s) + 1))
    return fused
