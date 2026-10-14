"""Offline tests for grounded generation. The provider is faked; no network."""

import json

import httpx
import pytest

from building_with_rag import registry
from building_with_rag.config import Settings
from building_with_rag.generation import assemble_context, generate_answer
from building_with_rag.generation.context import MAX_CONTEXT_CHARS, MAX_CONTEXT_PASSAGES
from building_with_rag.models import QueryRequest, QueryResult, RetrievedChunk


def passage(i: int, text: str = "Whoever commits murder shall be punished.") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"ipc:{i}#v1#0", section_id=f"ipc:{i}", act="IPC_1860",
                          text=text, heading="Murder", source="IPC 1860",
                          source_file="IPC.pdf", page=10 + i, status="repealed")


def settings(**kw) -> Settings:
    base = {"generation_api_base_url": "http://llm.test/v1/", "generation_api_key": "secret-key",
            "generation_model_name": "m"}
    return Settings(_env_file=None, **{**base, **kw})


def fake(content=None, status=200, exc=None, calls=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if exc:
            raise exc
        body = {"choices": [{"message": {"content": content}}]}
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handler)


def reply(**d) -> str:
    return json.dumps(d)


def test_context_bounds_and_labels():
    ctx = assemble_context([passage(i) for i in range(8)])
    assert list(ctx.passages) == [f"E{i}" for i in range(1, MAX_CONTEXT_PASSAGES + 1)]
    assert ctx.passages["E2"].section_id == "ipc:1"
    assert [o["reason"] for o in ctx.omitted] == ["passage_limit"] * 3
    big = assemble_context([passage(1, "x" * (MAX_CONTEXT_CHARS - 5)), passage(2, "y" * 10)])
    assert list(big.passages) == ["E1"] and big.omitted[0]["reason"] == "char_limit"
    assert assemble_context([]).outcome == "empty"


def test_answered_resolves_citations():
    out = reply(outcome="answered", answer="Death or life.",
                claims=[{"text": "Punished", "evidence": ["E2", "E2"]}])
    gen = generate_answer("q", [passage(1), passage(2)], settings(), transport=fake(out))
    assert gen.outcome == "answered" and gen.answer == "Death or life."
    assert gen.citations == [{"label": "E2", "chunk_id": "ipc:2#v1#0", "section_id": "ipc:2",
                              "act": "IPC_1860", "heading": "Murder", "source": "IPC 1860",
                              "source_file": "IPC.pdf", "page": 12}]
    assert [p.chunk_id for p in gen.supporting_passages] == ["ipc:2#v1#0"]
    assert gen.attempts == 1 and gen.context_outcome == "assembled"


def test_code_fence_tolerated():
    out = "```json\n" + reply(outcome="insufficient_evidence", answer="guess", reason="no") + "\n```"
    gen = generate_answer("q", [passage(1)], settings(), transport=fake(out))
    assert gen.outcome == "insufficient_evidence" and gen.answer is None
    assert gen.citations == [] and gen.trace[0]["reason"] == "no"


@pytest.mark.parametrize("content", [
    "not json",
    reply(outcome="answered", answer="x", claims=[{"text": "t", "evidence": ["E9"]}]),
    reply(outcome="answered", answer="x", claims=[]),
    reply(outcome="answered", answer="x", claims=[{"text": "t"}]),
    reply(outcome="maybe"),
])
def test_malformed(content):
    gen = generate_answer("q", [passage(1)], settings(), transport=fake(content))
    assert gen.outcome == "malformed" and gen.answer is None
    assert gen.claims == [] and gen.citations == [] and gen.supporting_passages == []


@pytest.mark.parametrize("kw", [{"generation_api_base_url": ""}, {"generation_api_key": ""}])
def test_unconfigured_is_unavailable_without_call(kw):
    calls = []
    gen = generate_answer("q", [passage(1)], settings(**kw), transport=fake("{}", calls=calls))
    assert gen.outcome == "unavailable" and gen.answer is None and not calls and gen.attempts == 0


def test_provider_failures_are_unavailable_and_safe():
    for transport in (fake(exc=httpx.ReadTimeout("slow")), fake("{}", status=502)):
        gen = generate_answer("q", [passage(1)], settings(), transport=transport)
        assert gen.outcome == "unavailable" and gen.answer is None
        assert "secret-key" not in gen.model_dump_json()


def test_request_is_standard_chat_completions():
    calls = []
    generate_answer("q", [passage(1)], settings(),
                    transport=fake(reply(outcome="insufficient_evidence"), calls=calls))
    req = calls[0]
    assert str(req.url) == "http://llm.test/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer secret-key"
    body = json.loads(req.content)
    assert set(body) == {"model", "messages", "temperature", "stream"}
    assert "untrusted evidence" in body["messages"][0]["content"]


def test_empty_retrieval_is_insufficient_without_call():
    calls = []
    gen = generate_answer("q", [], settings(), transport=fake("{}", calls=calls))
    assert gen.outcome == "insufficient_evidence" and gen.context_outcome == "empty"
    assert not calls and gen.attempts == 0


def test_generate_answer_flag_controls_generation(monkeypatch):
    base = QueryResult(pattern="semantic", status="ok", message="m", results=[passage(1)])
    monkeypatch.setattr("building_with_rag.retrieval.semantic_search", lambda r, s: base)
    off = registry.run_pattern(QueryRequest(question="q", pattern="semantic"), settings())
    assert off.generation is None
    on = registry.run_pattern(
        QueryRequest(question="q", pattern="semantic", generate_answer=True),
        settings(generation_api_key=""))
    assert on.generation.outcome == "unavailable" and on.results == base.results
