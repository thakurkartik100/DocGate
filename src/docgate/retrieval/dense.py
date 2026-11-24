"""Dense retrieval: sentence-transformer embeddings + cosine similarity in numpy (1.5k docs; no ANN index needed)."""
import hashlib
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Dense:
    def __init__(self, docs: list[str], cache_dir: Path | None = None, model_name: str = MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        key = hashlib.sha1((model_name + "\n".join(docs)).encode()).hexdigest()[:12]
        cache = cache_dir / f"emb_{key}.npy" if cache_dir else None
        if cache and cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(docs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            if cache:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.emb)

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return self.emb @ q
