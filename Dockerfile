FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/models
# CPU-only torch keeps the image ~1 GB smaller than the default CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
# Bake models and the embedding cache into the image so the container starts without network access.
COPY data ./data
RUN python -c "from docgate.corpus import load; from docgate.retrieval.pipeline import Retriever; from pathlib import Path; Retriever(load(Path('data/corpus.jsonl')), cache_dir=Path('data/.cache'))"
RUN useradd -m app && chown -R app /app
USER app
ENV HF_HUB_OFFLINE=1 DOCGATE_DATA=/app/data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "docgate.api:app", "--host", "0.0.0.0", "--port", "8000"]
