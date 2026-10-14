"""Offline tests for grounded, streamed generation. The provider is faked; no network."""

import json

import httpx

from building_with_rag import registry
from building_with_rag.config import Settings
from building_with_rag.generation import (
    assemble_context,
    generate_answer,
    generate_events,
    render,
    with_generation,
)
from building_with_rag.generation.checks import split_claims
from building_with_rag.generation.context import MAX_CONTEXT_CHARS, MAX_CONTEXT_PASSAGES
from building_with_rag.generation.events import AttemptStarted, CheckFailed, Final, Text
from building_with_rag.models import QueryRequest, QueryResult, RetrievedChunk

MURDER = ("Whoever commits murder shall be punished with death or imprisonment for life, "
          "and shall also be liable to fine.")
GOOD = "Murder is punished with death or imprisonment for life [E1]."
FAKED = "Murder is punished with 99 years of rigorous imprisonment [E1]."
UNKNOWN = "Murder is punished with death or imprisonment for life [E9]."
OFF_TOPIC = "Pedestrians crossing roads carelessly receive traffic warnings [E1]."


def passage(i: int, text: str = "Whoever commits murder shall be punished.") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"ipc:{i}#v1#0", section_id=f"ipc:{i}", act="IPC_1860",
                          text=text, heading="Murder", source="IPC 1860",
                          source_file="IPC.pdf", page=10 + i, status="repealed")


def p1() -> list[RetrievedChunk]:
    return [passage(1, MURDER), passage(2)]


def settings(**kw) -> Settings:
    base = {"generation_api_base_url": "http://llm.test/v1/", "generation_api_key": "secret-key",
            "generation_model_name": "m"}
    return Settings(_env_file=None, **{**base, **kw})


class Cut(httpx.SyncByteStream):
    """A provider stream that dies after its first frame."""

    def __iter__(self):
        yield sse_frame("Murder is punished ")
        raise httpx.ReadError("cut")


def sse_frame(piece: str) -> bytes:
    return b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n\n"


def sse(text: str) -> bytes:
    third = max(1, len(text) // 3)
    parts = [text[k:k + third] for k in range(0, len(text), third)]
    return b"".join(sse_frame(p) for p in parts) + b"data: [DONE]\n\n"


def fake(*replies, status=200, exc=None, calls=None):
    """One reply per provider call (the last repeats): text, or a ready httpx.Response."""
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if exc:
            raise exc
        reply = queue.pop(0) if len(queue) > 1 else (queue[0] if queue else "")
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(status, content=sse(reply))
    return httpx.MockTransport(handler)


def run(question, results, transport, **kw):
    return list(generate_events(question, results, settings(**kw), transport=transport))


def registry_message(gen) -> str:
    base = QueryResult(pattern="semantic", status="ok", message="m", results=p1())
    return with_generation(base, gen).message


def test_context_bounds_and_labels():
    ctx = assemble_context([passage(i) for i in range(8)])
    assert list(ctx.passages) == [f"E{i}" for i in range(1, MAX_CONTEXT_PASSAGES + 1)]
    assert ctx.passages["E2"].section_id == "ipc:1"
    assert [o["reason"] for o in ctx.omitted] == ["passage_limit"] * 3
    big = assemble_context([passage(1, "x" * (MAX_CONTEXT_CHARS - 5)), passage(2, "y" * 10)])
    assert list(big.passages) == ["E1"] and big.omitted[0]["reason"] == "char_limit"
    assert assemble_context([]).outcome == "empty"


def test_answered_streams_and_resolves_citations():
    events = run("q", p1(), fake(GOOD))
    assert "".join(e.text for e in events if isinstance(e, Text)) == GOOD
    gen = events[-1].result
    assert isinstance(events[0], AttemptStarted) and isinstance(events[-1], Final)
    assert gen.outcome == "answered" and gen.answer == GOOD and gen.confidence == "high"
    assert gen.claims == [{"text": GOOD, "evidence": ["E1"]}] and gen.issues == []
    assert gen.citations == [{"label": "E1", "chunk_id": "ipc:1#v1#0", "section_id": "ipc:1",
                              "act": "IPC_1860", "heading": "Murder", "source": "IPC 1860",
                              "source_file": "IPC.pdf", "page": 11}]
    assert [p.chunk_id for p in gen.supporting_passages] == ["ipc:1#v1#0"]
    assert gen.attempts == 1 and gen.context_outcome == "assembled"
    text = "".join(render(events))
    assert text.startswith("DRAFT — checking evidence\n\n" + GOOD)
    assert "**Confidence: high**" in text and "- E1 · ipc:1 · Murder" in text


def test_split_claims_rule():
    claims = split_claims("Sec. 103 applies [E1]. It says so [E2, E3].\n- Third one [E1]")
    assert [c["evidence"] for c in claims] == [["E1"], ["E2", "E3"], ["E1"]]
    assert claims[0]["text"].startswith("Sec. 103")


def test_citation_and_support_failures_are_detected():
    unknown = run("q", p1(), fake(UNKNOWN))
    kinds = [(e.check, e.detail) for e in unknown if isinstance(e, CheckFailed)]
    assert any(c == "citation" and "E9" in d for c, d in kinds)
    assert unknown[-1].result.issues[0]["labels"] == ["E9"]  # not stripped
    faked = [e.check for e in run("q", p1(), fake(FAKED)) if isinstance(e, CheckFailed)]
    assert faked.count("support") == 2 and "citation" not in faked
    off = [e.check for e in run("q", p1(), fake(OFF_TOPIC)) if isinstance(e, CheckFailed)]
    assert "support" in off


def test_retry_then_pass_keeps_first_failure():
    calls = []
    events = run("q", p1(), fake(UNKNOWN, GOOD, calls=calls))
    gen = events[-1].result
    assert gen.outcome == "answered" and gen.attempts == 2 and len(calls) == 2
    assert {i["attempt"] for i in gen.issues} == {1} and gen.issues[0]["check"] == "citation"
    text = "".join(render(events))
    assert text.count("DRAFT — checking evidence") == 2
    assert "Retrying (attempt 2 of 2)" in text and UNKNOWN in text and "Checks passed" in text
    assert "which was not supplied" in json.loads(calls[1].content)["messages"][-1]["content"]


def test_two_failed_attempts_end_as_low_confidence_draft():
    events = run("q", p1(), fake(UNKNOWN, FAKED))
    gen = events[-1].result
    assert gen.outcome == "malformed" and gen.confidence == "low" and gen.answer is None
    assert gen.draft_answer == FAKED and gen.attempts == 2
    assert gen.claims == [] and gen.citations == [] and gen.supporting_passages == []
    assert [i["attempt"] for i in gen.issues] == [1, 1, 2]
    assert gen.issues[0]["labels"] == ["E9"] and gen.low_confidence_reason
    text = "".join(render(events))
    assert "DRAFT — low confidence, not the final answer" in text and "Sources" not in text
    assert text.index("low confidence") > text.index(FAKED)
    assert "failed evidence checks" in registry_message(gen)


def test_empty_output_is_malformed_without_confidence():
    gen = run("q", p1(), fake(""))[-1].result
    assert gen.outcome == "malformed" and gen.confidence is None and gen.draft_answer is None
    assert [i["check"] for i in gen.issues] == ["no_answer_text"] * 2


def test_insufficient_marker_is_not_a_draft_and_not_retried():
    calls = []
    events = run("q", p1(), fake("INSUFFICIENT_EVIDENCE: nothing on jaywalking", calls=calls))
    gen = events[-1].result
    assert gen.outcome == "insufficient_evidence" and gen.answer is None and gen.confidence is None
    assert len(calls) == 1 and not any(isinstance(e, Text) for e in events)
    text = "".join(render(events))
    assert "DRAFT" not in text and "nothing on jaywalking" in text


def test_unconfigured_is_unavailable_without_call():
    for kw in ({"generation_api_base_url": ""}, {"generation_api_key": ""}):
        calls = []
        gen = run("q", p1(), fake(GOOD, calls=calls), **kw)[-1].result
        assert gen.outcome == "unavailable" and gen.answer is None and not calls
        assert gen.attempts == 0 and gen.confidence is None


def test_provider_failures_are_unavailable_and_safe():
    for transport in (fake(exc=httpx.ReadTimeout("slow")), fake(GOOD, status=502)):
        events = run("q", p1(), transport)
        gen = events[-1].result
        assert gen.outcome == "unavailable" and gen.answer is None and gen.confidence is None
        assert "secret-key" not in gen.model_dump_json()
        assert "".join(render(events)).startswith("⚠ Answer generation unavailable")


def test_stream_cut_keeps_partial_draft_and_labels_it():
    events = run("q", p1(), fake(httpx.Response(200, stream=Cut())))
    gen = events[-1].result
    assert gen.outcome == "unavailable" and gen.draft_answer == "Murder is punished "
    assert gen.answer is None and gen.confidence is None
    text = "".join(render(events))
    assert text.startswith("DRAFT") and "incomplete DRAFT, not a final answer" in text
    assert "interrupted" in registry_message(gen)


def test_request_is_standard_streaming_chat_completions():
    calls = []
    run("q", p1(), fake("INSUFFICIENT_EVIDENCE: no", calls=calls))
    req = calls[0]
    assert str(req.url) == "http://llm.test/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer secret-key"
    body = json.loads(req.content)
    assert set(body) == {"model", "messages", "temperature", "stream"} and body["stream"] is True
    assert "untrusted evidence" in body["messages"][0]["content"]


def test_empty_retrieval_is_insufficient_without_call():
    calls = []
    gen = generate_answer("q", [], settings(), transport=fake(GOOD, calls=calls))
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
