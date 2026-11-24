from types import SimpleNamespace as NS

from docgate import api, generate
from tests.test_api import client  # noqa: F401  (fixture)


class FakeClient:
    def __init__(self, resp):
        self.resp, self.sent = resp, None
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.sent = kw
        return self.resp


def _text(text, cited=()):
    return NS(type="text", text=text, citations=[NS(document_title=c) for c in cited])


def test_answer_collects_text_and_deduplicated_citations():
    resp = NS(stop_reason="end_turn", model=generate.MODEL, usage=NS(input_tokens=900, output_tokens=40),
              content=[NS(type="thinking"), _text("Use json.dumps. ", ["json.dumps"]), _text("It returns a str.", ["json.dumps", "json.JSONEncoder"])])
    fake = FakeClient(resp)
    out = generate.answer("to json?", [("json.dumps", "json.dumps(obj)")], client=fake)
    assert out == {"text": "Use json.dumps. It returns a str.", "citations": ["json.dumps", "json.JSONEncoder"], "refused": False, "model": generate.MODEL, "usage": {"input": 900, "output": 40}}
    doc = fake.sent["messages"][0]["content"][0]
    assert doc["title"] == "json.dumps" and doc["citations"] == {"enabled": True}
    assert fake.sent["fallbacks"] == "default" and generate.FALLBACK_BETA in fake.sent["betas"]


def test_answer_reports_refusal_without_reading_content():
    out = generate.answer("q", [("a", "b")], client=FakeClient(NS(stop_reason="refusal", model="m", content=None, usage=NS(input_tokens=1, output_tokens=0))))
    assert out["refused"] and out["text"] is None


def test_api_returns_generated_answer(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(generate, "answer", lambda q, p: {"text": "Use json.dumps.", "citations": ["json.dumps"], "refused": False, "model": "m"})
    r = client.post("/query", json={"q": "serialize json", "generate": True}).json()
    assert r["answer"] == "Use json.dumps." and r["citations"] == ["json.dumps"]
    assert client.post("/query", json={"q": "serialize json"}).json()["answer"] is None  # separate cache entry


def test_api_skips_generation_when_abstaining(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(generate, "answer", lambda *a: (_ for _ in ()).throw(AssertionError("must not be called")))
    assert client.post("/query", json={"q": "pizza recipe", "generate": True}).json()["answer"] is None


def test_api_503_when_no_credentials(client, monkeypatch):  # noqa: F811
    def no_creds(*a):
        raise TypeError("Could not resolve authentication method.")
    monkeypatch.setattr(generate, "answer", no_creds)
    r = client.post("/query", json={"q": "serialize json", "generate": True})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
