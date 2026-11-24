import json

import pytest
from fastapi.testclient import TestClient

from docgate import api
from docgate.retrieval.pipeline import Hit


class FakeRetriever:
    calls = 0

    def __init__(self, *a, **kw):
        self.chunks = [{"id": "json.dumps", "text": "json.dumps(obj)"}]

    def search(self, q, k=5, **kw):
        FakeRetriever.calls += 1
        score = 5.0 if "json" in q else -9.0
        return [Hit("json.dumps", score, "json.dumps(obj)\nSerialize obj")]


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "corpus.jsonl").write_text(json.dumps({"id": "x", "text": "x"}) + "\n")
    (tmp_path / "abstain_threshold.json").write_text('{"threshold": 0.0}')
    monkeypatch.setattr(api, "DATA", tmp_path)
    monkeypatch.setattr(api, "Retriever", FakeRetriever)
    FakeRetriever.calls = 0
    with TestClient(api.app) as c:
        yield c


def test_answers_confident_query_then_serves_from_cache(client):
    r1 = client.post("/query", json={"q": "serialize json"}).json()
    r2 = client.post("/query", json={"q": "serialize json"}).json()
    assert r1["answered"] and r1["passages"][0]["id"] == "json.dumps" and not r1["cached"]
    assert r2["cached"] and FakeRetriever.calls == 1


def test_abstains_below_threshold_and_caches_the_abstention(client):
    assert client.post("/query", json={"q": "pizza recipe"}).json()["passages"] == []
    assert client.post("/query", json={"q": "pizza recipe"}).json()["cached"]


def test_input_validation(client):
    assert client.post("/query", json={"q": "x"}).status_code == 422
    assert client.post("/query", json={"q": "valid query", "k": 99}).status_code == 422


def test_request_id_propagates_and_metrics_exposed(client):
    r = client.post("/query", json={"q": "json thing"}, headers={"x-request-id": "abc123"}).json()
    assert r["request_id"] == "abc123"
    assert "docgate_requests_total" in client.get("/metrics").text
    assert client.get("/health").json()["status"] == "ok"


def test_demo_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "DocGate" in r.text
