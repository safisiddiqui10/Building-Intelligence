"""Offline tests: chat streams the same generation operation /v1/query returns (Story 3.2)."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from building_with_rag.config import Settings, get_settings
from building_with_rag.generation import answer as answer_module
from building_with_rag.main import app
from building_with_rag.models import QueryResult, RetrievedChunk
from building_with_rag.retrieval import RetrievalUnavailable

client = TestClient(app)

MURDER = ("Whoever commits murder shall be punished with death or imprisonment for life, "
          "and shall also be liable to fine.")
GOOD = "Murder is punished with death or imprisonment for life [E1]."
UNKNOWN = "Murder is punished with death or imprisonment for life [E9]."
CHUNK = RetrievedChunk(chunk_id="bns:103#v1#0", section_id="bns:103", act="BNS_2023", text=MURDER,
                       heading="Murder", source="BNS 2023", source_file="BNS.pdf", page=125,
                       status="in_force")
CHAT = {"model": "rag-semantic", "stream": True,
        "messages": [{"role": "user", "content": "What is murder?"}],
        "rag_options": {"pattern": "semantic", "limit": 5}}
QUERY = {"question": "What is murder?", "pattern": "semantic", "generate_answer": True}


def make_settings(**kw) -> Settings:
    base = {"generation_api_base_url": "http://llm.test/v1", "generation_api_key": "gen-secret"}
    return Settings(_env_file=None, **{**base, **kw})


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    app.dependency_overrides[get_settings] = lambda: make_settings()
    yield
    app.dependency_overrides.clear()
    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT", None)


def provider(*replies, calls=None):
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        parts = [text[i:i + 12] for i in range(0, len(text), 12)]
        frames = "".join("data: " + json.dumps({"choices": [{"delta": {"content": p}}]}) + "\n\n"
                         for p in parts)
        return httpx.Response(200, content=(frames + "data: [DONE]\n\n").encode())
    return httpx.MockTransport(handler)


@pytest.fixture
def searches(monkeypatch):
    seen = []

    def fake_search(request, settings):
        seen.append(request)
        return QueryResult(pattern="semantic", status="ok", message="m", results=[CHUNK])

    monkeypatch.setattr("building_with_rag.retrieval.semantic_search", fake_search)
    return seen


def chunks_of(text: str):
    frames = [f for f in text.split("\n\n") if f]
    assert frames[-1] == "data: [DONE]"
    return [json.loads(f[len("data: "):]) for f in frames[:-1]]


def content_of(chunks) -> str:
    return "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)


def test_chat_streams_the_same_operation_query_returns(searches, monkeypatch):
    calls = []
    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT", provider(GOOD, calls=calls))
    r = client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    chunks = chunks_of(r.text)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert len(chunks) > 5  # role, several text pieces, closing block, stop
    text = content_of(chunks)
    assert text.startswith("DRAFT — checking evidence\n\n" + GOOD)
    assert "**Confidence: high**" in text and "(attempt 1 of 2)" in text
    assert "- E1 · bns:103 · Murder · BNS 2023 · p.125" in text
    assert len(calls) == 1 and len(searches) == 1  # one provider call, one retrieval
    assert searches[0].generate_answer is True and searches[0].caller_id == "demo-public"

    gen = client.post("/v1/query", json=QUERY).json()["generation"]
    assert (gen["outcome"], gen["confidence"], gen["attempts"]) == ("answered", "high", 1)
    assert [c["section_id"] for c in gen["citations"]] == ["bns:103"]
    assert len(calls) == 2  # /v1/query made its own single call

    plain = client.post("/v1/chat/completions", json={**CHAT, "stream": False})
    assert plain.json()["choices"][0]["message"]["content"] == text


def test_low_confidence_and_provider_failure_still_close(searches, monkeypatch):
    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT", provider(UNKNOWN))
    text = content_of(chunks_of(client.post("/v1/chat/completions", json=CHAT).text))
    assert "Retrying (attempt 2 of 2)" in text and "DRAFT — low confidence" in text
    gen = client.post("/v1/query", json=QUERY).json()["generation"]
    assert (gen["outcome"], gen["confidence"], gen["answer"]) == ("malformed", "low", None)
    assert gen["draft_answer"] == UNKNOWN and gen["citations"] == [] and gen["issues"]

    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT",
                        httpx.MockTransport(lambda req: httpx.Response(502)))
    chunks = chunks_of(client.post("/v1/chat/completions", json=CHAT).text)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert "Answer generation unavailable" in content_of(chunks)


def test_placeholder_modes_do_no_retrieval_or_generation(searches, monkeypatch):
    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT",
                        httpx.MockTransport(lambda req: pytest.fail("provider called")))
    r = client.post("/v1/chat/completions", json={**CHAT, "model": "rag-hybrid-reranked"})
    assert "not implemented" in content_of(chunks_of(r.text)) and searches == []


def test_retrieval_failure_is_openai_error_before_stream(monkeypatch):
    def boom(request, settings):
        raise RetrievalUnavailable("MongoDB is unreachable")

    monkeypatch.setattr("building_with_rag.retrieval.semantic_search", boom)
    r = client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 503 and r.json()["error"]["code"] == "retrieval_unavailable"
    q = client.post("/v1/query", json={"question": "x", "pattern": "semantic"})
    assert q.status_code == 503 and "detail" in q.json()


def test_capstone_key_auth(monkeypatch):
    query = {"question": "x", "pattern": "hybrid-reranked"}
    chat = {**CHAT, "model": "rag-hybrid-reranked"}
    assert client.post("/v1/query", json=query).status_code == 200  # no key configured: open
    app.dependency_overrides[get_settings] = lambda: make_settings(capstone_api_key="s3cret")
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret"}):
        assert client.post("/v1/query", json=query, headers=headers).status_code == 401
        r = client.post("/v1/chat/completions", json=chat, headers=headers)
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
        assert "s3cret" not in r.text
    ok = {"Authorization": "Bearer s3cret"}
    assert client.post("/v1/query", json=query, headers=ok).status_code == 200
    assert client.post("/v1/chat/completions", json=chat, headers=ok).status_code == 200
    assert client.get("/healthz").status_code == 200 and client.get("/v1/models").status_code == 200
