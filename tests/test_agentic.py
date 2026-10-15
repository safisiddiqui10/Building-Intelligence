"""Offline tests for the bounded agentic controller (Story 6.1): adapters and clock are faked."""

import json

import pytest
from fastapi.testclient import TestClient

from building_with_rag.config import Settings, get_settings
from building_with_rag.main import app
from building_with_rag.models import (
    GenerationResult,
    QueryRequest,
    QueryResult,
    RetrievedChunk,
    SemanticFilters,
)
from building_with_rag.retrieval import RetrievalUnavailable, agentic
from building_with_rag.retrieval.agentic import agentic_search

HYPO = "SECRET hypothetical text about dishonestly receiving stolen property."
MISMATCH_Q = "What happens to a person who knowingly buys a phone that a thief stole?"
ORDINARY_Q = "What is the punishment for murder?"


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def chunk(cid="bns:103#0", score=0.85, act="BNS_2023"):
    return RetrievedChunk(chunk_id=cid, section_id=cid.split("#")[0], act=act, text="Murder text.",
                          score=score, access_level="public", status="in_force")


def result(pattern, status="ok", chunks=(), **kw):
    return QueryResult(pattern=pattern, status=status, message="m", results=list(chunks),
                       trace=[{"ignored_fields": []}], **kw)


@pytest.fixture
def adapters(monkeypatch):
    """Fake adapters: set `table[tool]` to a result or an exception."""
    calls, table = [], {}

    def make(tool):
        def fake(request, s):
            calls.append((tool, request))
            out = table[tool]
            if isinstance(out, Exception):
                raise out
            return out
        return fake

    for tool in agentic.ADAPTERS:
        monkeypatch.setitem(agentic.ADAPTERS, tool, make(tool))
    return calls, table


def run(question, req=None, **kw):
    request = QueryRequest(question=question, pattern="agentic", **(req or {}))
    return agentic_search(request, settings(**kw))


def ticker(*values):
    """A fake clock that returns each value once, then the last one forever."""
    it = iter(values)
    last = [values[-1]]

    def clock():
        last[0] = next(it, last[0])
        return last[0]
    return clock


def names(calls):
    return [c[0] for c in calls]


@pytest.mark.parametrize("question,reason", [
    ("What does section 103 say?", "bare_section_number"), ("103", "bare_section_number"),
    ("???", "malformed"), ("BNS 103 and IPC 302", "ambiguous_request"),
    ("list all sections about theft", "unsupported_intent")])
def test_validation_rejects_before_any_call(adapters, question, reason):
    calls, _ = adapters
    res = run(question)
    t = res.trace[0]
    assert res.status == "clarify" and names(calls) == []
    assert t["stop_reason"] == "validation_rejected" and t["call_count"] == 0
    assert t["validation"] == {"outcome": "rejected", "reason": reason}
    assert question not in json.dumps(res.model_dump())


def test_exact_path_one_structured_call(adapters):
    calls, table = adapters
    table["structuredLookup"] = result("structured", chunks=[chunk()])
    res = run("BNS section 103")
    assert names(calls) == ["structuredLookup"] and res.status == "ok"
    assert res.trace[0]["stop_reason"] == "ready_for_synthesis" and res.trace[0]["path"] == "exact"
    table["structuredLookup"] = result("structured", status="not_found")
    res = run("BNS section 999")
    assert res.status == "clarify" and res.trace[0]["stop_reason"] == "record_not_found"
    assert names(calls) == ["structuredLookup"] * 2 and res.results == []


def test_ordinary_path_semantic_once(adapters):
    calls, table = adapters
    table["semanticRetrieve"] = result("semantic",
                                       chunks=[chunk(score=0.9), chunk("bns:104#0", 0.5)])
    res = run(ORDINARY_Q)
    t = res.trace[0]
    assert names(calls) == ["semanticRetrieve"] and res.status == "ok"
    assert [r.chunk_id for r in res.results] == ["bns:103#0"]  # the weak passage is not evidence
    assert t["adequacy"][0]["adequate"] and t["calls"][0]["weak"][0]["chunk_id"] == "bns:104#0"
    table["semanticRetrieve"] = result("semantic", chunks=[chunk(score=0.5)])
    res = run(ORDINARY_Q)
    assert res.status == "partial_result" and res.results == []
    table["semanticRetrieve"] = result("semantic", status="no_results")
    assert run(ORDINARY_Q).status == "clarify"
    assert "hydeRetrieve" not in names(calls)


def test_mismatch_calls_hyde_only_after_inadequate_semantic_and_once(adapters):
    calls, table = adapters
    table["semanticRetrieve"] = result("semantic", chunks=[chunk(score=0.5)])
    table["hydeRetrieve"] = result("hyde", chunks=[chunk("ipc:411#0", 0.03, "IPC_1860")],
                                   hyde_hypothetical_text_debug=HYPO)
    res = run(MISMATCH_Q)
    t = res.trace[0]
    assert names(calls) == ["semanticRetrieve", "hydeRetrieve"]
    assert t["call_count"] == 2 and t["hyde_calls"] == 1 and t["mismatch_rule"]["applied"]
    assert res.status == "ok" and [r.section_id for r in res.results] == ["ipc:411"]
    assert HYPO not in json.dumps(res.model_dump()) and res.hyde_hypothetical_text_debug is None
    table["semanticRetrieve"] = result("semantic", chunks=[chunk(score=0.9)])
    run(MISMATCH_Q)
    assert names(calls)[2:] == ["semanticRetrieve"]  # adequate: no HyDE


def test_budget_allows_no_third_call(adapters):
    calls, table = adapters
    table["semanticRetrieve"] = result("semantic", chunks=[chunk(score=0.5)])
    table["hydeRetrieve"] = result("hyde", status="no_results")
    res = run(MISMATCH_Q)
    assert res.status == "partial_result" and len(calls) == 2
    assert res.trace[0]["stop_reason"] == "no_usable_evidence"


@pytest.mark.parametrize("bad,stop", [
    (RetrievalUnavailable("down"), "tool_unavailable"),
    (result("hyde", chunks=[chunk()]), "tool_invalid"),  # wrong pattern for semanticRetrieve
    (result("semantic", status="weird"), "tool_invalid"),
    (result("semantic", chunks=[chunk().model_copy(update={"access_level": "private"})]),
     "tool_invalid")])
def test_bad_tool_stops_explicitly(adapters, bad, stop):
    calls, table = adapters
    table["semanticRetrieve"] = bad
    res = run(ORDINARY_Q)
    assert res.status == "unavailable" and res.trace[0]["stop_reason"] == stop
    assert res.results == [] and len(calls) == 1


def test_deadline_before_and_after_call(adapters):
    calls, table = adapters
    table["semanticRetrieve"] = result("semantic", chunks=[chunk(score=0.5)])
    request = QueryRequest(question=ORDINARY_Q, pattern="agentic")
    res = agentic_search(request, settings(), clock=ticker(0, 100))  # guard already past 45 s
    assert res.trace[0]["stop_reason"] == "deadline_expired" and calls == []
    table["semanticRetrieve"] = result("semantic", chunks=[chunk()])
    res = agentic_search(request, settings(), clock=ticker(0, 1, 100))  # expires mid-call
    assert res.trace[0]["stop_reason"] == "deadline_expired" and res.results == []
    with pytest.raises(RetrievalUnavailable, match="AGENTIC_DEADLINE_SECONDS"):
        run(ORDINARY_Q, agentic_deadline_seconds=0)


def test_scope_propagates_unchanged_and_answer_flag_off(adapters):
    calls, table = adapters
    table["semanticRetrieve"] = result("semantic", chunks=[chunk(score=0.5)])
    table["hydeRetrieve"] = result("hyde", status="no_results")
    filters = SemanticFilters(act=["BNS_2023"], status=["in_force"])
    run(MISMATCH_Q, req={"filters": filters, "generate_answer": True, "limit": 3})
    assert len(calls) == 2
    for _, req in calls:
        assert req.filters == filters and req.limit == 3 and not req.generate_answer


def test_wiring_models_and_generation_only_for_ok(adapters, monkeypatch):
    _calls, table = adapters
    from building_with_rag.generation import events as ev

    seen = []

    def fake_events(question, passages, s):
        seen.append((question, [p.chunk_id for p in passages]))
        yield ev.Text("ok")
        yield ev.Final(GenerationResult(outcome="answered", answer="ok"))
    monkeypatch.setattr("building_with_rag.generation.generate_events", fake_events)
    app.dependency_overrides[get_settings] = lambda: settings()
    try:
        client = TestClient(app)
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
        assert len(ids) == 7 and ids[-1] == "rag-agentic"
        table["semanticRetrieve"] = result("semantic", chunks=[chunk(score=0.9)])
        body = {"question": ORDINARY_Q, "pattern": "agentic", "generate_answer": True}
        assert client.post("/v1/query", json=body).json()["status"] == "ok"
        assert seen == [(ORDINARY_Q, ["bns:103#0"])]
        r = client.post("/v1/query", json={**body, "question": "What does section 103 say?"})
        assert r.json()["status"] == "clarify" and len(seen) == 1
        chat = client.post("/v1/chat/completions", json={
            "model": "rag-agentic", "messages": [{"role": "user", "content": "103"}]})
        assert chat.status_code == 200
        assert "unclear" in chat.json()["choices"][0]["message"]["content"]
    finally:
        app.dependency_overrides.clear()
