"""Answer generation: Claude writes a short answer grounded in the retrieved passages, with native citations.

Each retrieved chunk is sent as a separate document block with citations enabled, so every cited
sentence points back to a chunk id. That makes grounding checkable instead of trusted.
"""
import anthropic

MODEL = "claude-opus-5-5"
# Server-side refusal fallback: if the model declines, the API re-runs the request on a fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
SYSTEM = (
    "You answer questions about the Python standard library using only the documents provided. "
    "Ground every statement in the documents and cite them. If the documents do not answer the "
    "question, say so in one sentence instead of guessing. Keep answers under 120 words; include a "
    "short code example only when it makes the answer clearer."
)


def answer(question: str, passages: list[tuple[str, str]], client: anthropic.Anthropic | None = None) -> dict:
    """passages: (chunk_id, text) pairs, best first. Returns text, cited chunk ids, and refusal flag."""
    client = client or anthropic.Anthropic()
    docs = [
        {"type": "document", "title": cid, "citations": {"enabled": True},
         "source": {"type": "text", "media_type": "text/plain", "data": text}}
        for cid, text in passages
    ]
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        output_config={"effort": "low"},  # short grounded Q&A; low effort keeps latency and cost down
        messages=[{"role": "user", "content": [*docs, {"type": "text", "text": question}]}],
    )
    if resp.stop_reason == "refusal":
        return {"text": None, "citations": [], "refused": True, "model": resp.model}
    text, cited = [], []
    for block in resp.content:
        if block.type == "text":
            text.append(block.text)
            cited += [c.document_title for c in (block.citations or []) if getattr(c, "document_title", None)]
    return {"text": "".join(text).strip(), "citations": list(dict.fromkeys(cited)), "refused": False, "model": resp.model}
