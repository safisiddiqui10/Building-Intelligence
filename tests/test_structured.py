"""Offline tests for structured exact retrieval (Story 5.1): the collection and LLM are faked."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pymongo.errors import PyMongoError

from building_with_rag.config import Settings, get_settings
from building_with_rag.generation import answer as answer_module
from building_with_rag.main import app
from building_with_rag.models import QueryRequest, RetrievedChunk
from building_with_rag.retrieval import RetrievalUnavailable, structured
from building_with_rag.retrieval.structured import classify, structured_search

client = TestClient(app)


def section(sid="bns:103", act="BNS_2023", number=103, status="in_force", version="retrieved; in force"):
    return {"section_id": sid, "act": act, "act_label": "BNS 2023", "status": status,
            "chapter": "VI", "section_number": number, "heading": "Punishment for murder",
            "text": "Whoever commits murder shall be punished with death or imprisonment for life.",
            "source_pdf": "BNS.pdf", "source_status_version": version, "page_start": 9,
            "access_level": "public"}


class FakeSections:
    def __init__(self, rows, calls, fail=False):
        self.rows, self.calls, self.fail = rows, calls, fail

    def find(self, pred, projection=None):
        self.calls.append(pred)
        if self.fail:
            raise PyMongoError("boom")
        return self

    def limit(self, n):
        return self.rows[:n]


def fake_db(monkeypatch, rows=(), fail=False):
    calls = []
    monkeypatch.setattr(structured, "get_db", lambda s: {"sections": FakeSections(list(rows), calls, fail)})
    return calls


def no_db(monkeypatch):
    def boom(s):
        pytest.fail("database touched")

    monkeypatch.setattr(structured, "get_db", boom)


def settings(**kw) -> Settings:
    base = {"generation_api_base_url": "http://llm.test/v1", "generation_api_key": "gen-secret"}
    return Settings(_env_file=None, **{**base, **kw})


def req(q="BNS section 103", **kw) -> QueryRequest:
    return QueryRequest(question=q, pattern="structured", **kw)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    yield
    app.dependency_overrides.clear()
    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT", None)


@pytest.mark.parametrize("q,act,num", [("BNS section 103", "BNS", "103"),
                                       ("What does section 302 of the IPC say?", "IPC", "302"),
                                       ("IPC 304A", "IPC", "304A"),
                                       ("Section 103, Bharatiya Nyaya Sanhita", "BNS", "103")])
def test_classifier_exact_signals(q, act, num):
    c = classify(q)
    assert c.status == "ok" and c.signals.intent == "exact_lookup"
    assert (c.signals.act, c.signals.section_number) == (act, num)


@pytest.mark.parametrize("q,status,reason", [
    ("What does section 103 say?", "clarification_needed", "missing_act"),
    ("BNS section 103 and IPC section 302", "clarification_needed", "multiple_acts"),
    ("BNS sections 103 and 104", "clarification_needed", "multiple_sections"),
    ("how many sections are in the IPC?", "unsupported", "unsupported_aggregation"),
    ("list the sections in chapter VI of the BNS", "unsupported", "unsupported_filter"),
    ("What is the punishment for murder?", "unsupported", "no_supported_intent"),
    ('{"$ne": null}', "unsupported", "no_supported_intent")])
def test_classifier_not_exact(q, status, reason):
    c = classify(q)
    assert (c.status, c.reason) == (status, reason)
    assert c.signals is None or c.signals.intent != "exact_lookup" or c.signals.act is None \
        or c.signals.section_number is None


def test_valid_lookup_predicate_and_result(monkeypatch):
    calls = fake_db(monkeypatch, [section()])
    res = structured_search(req("BNS section 103; {\"$where\": 1}"), settings())
    assert calls == [{"act": "BNS_2023", "section_number": 103, "status": "in_force",
                      "access_level": "public"}]
    assert res.pattern == "structured" and res.status == "ok" and len(res.results) == 1
    r = res.results[0]
    assert (r.chunk_id, r.section_id, r.score) == ("bns:103#section", "bns:103", None)
    assert r.source_status_version == "retrieved; in force" and r.heading
    t = res.trace[0]
    assert t["mongodb_called"] is True and t["matched"] == 1 and "murder" not in json.dumps(t)
    assert "current" not in res.message.replace("not current", "")


def test_ipc_lookup_uses_repealed_status_and_chapter(monkeypatch):
    calls = fake_db(monkeypatch, [section("ipc:302", "IPC_1860", 302, "repealed")])
    res = structured_search(req("IPC section 302 in chapter XVI"), settings())
    assert calls[0] == {"act": "IPC_1860", "section_number": 302, "status": "repealed",
                        "access_level": "public", "chapter": "XVI"} and res.status == "ok"


def test_missing_record_is_not_found(monkeypatch):
    fake_db(monkeypatch, [])
    res = structured_search(req("BNS section 999"), settings())
    assert res.status == "not_found" and res.results == [] and res.trace[0]["mongodb_called"]


def test_ambiguous_number_and_suffix(monkeypatch):
    rows = [section("ipc:304", "IPC_1860", 304, "repealed"), section("ipc:304A", "IPC_1860", 304, "repealed")]
    fake_db(monkeypatch, rows)
    res = structured_search(req("IPC section 304"), settings())
    assert res.status == "clarification_needed" and res.results == []
    assert res.trace[0]["candidate_section_ids"] == ["ipc:304", "ipc:304A"]
    one = structured_search(req("IPC section 304A"), settings())
    assert one.status == "ok" and one.results[0].section_id == "ipc:304A"


def test_unsupported_and_clarification_never_touch_database(monkeypatch):
    no_db(monkeypatch)
    for q in ("What does section 103 say?", "What is the punishment for murder?"):
        res = structured_search(req(q), settings(mongodb_uri=""))
        assert res.status in ("clarification_needed", "unsupported")
        assert res.results == [] and res.trace[0]["mongodb_called"] is False
    f = {"act": ["IPC_1860"]}
    res = structured_search(req(filters=f), settings())  # filters only narrow
    assert res.status == "not_found" and res.trace[0]["mongodb_called"] is False
    assert structured_search(req(chapter="nope"), settings()).status == "clarification_needed"


def test_mongo_error_is_retrieval_unavailable(monkeypatch):
    fake_db(monkeypatch, fail=True)
    with pytest.raises(RetrievalUnavailable, match="MongoDB lookup failed"):
        structured_search(req(), settings())


def test_source_status_version_defaults_null():
    assert RetrievedChunk(chunk_id="c", section_id="s", act="A", text="t").source_status_version is None


def test_query_and_chat_answer_boundary(monkeypatch):
    fake_db(monkeypatch, [section()])
    app.dependency_overrides[get_settings] = lambda: settings()
    body = client.post("/v1/query", json={"question": "BNS section 103", "pattern": "structured"}).json()
    assert body["status"] == "ok" and body["generation"] is None and body["trace"][0]["predicate"]

    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        frame = "data: " + json.dumps({"choices": [{"delta": {"content":
            "Murder is punished with death or imprisonment for life [E1]."}}]}) + "\n\n"
        return httpx.Response(200, content=(frame + "data: [DONE]\n\n").encode())

    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT", httpx.MockTransport(handler))
    gen = client.post("/v1/query", json={"question": "BNS section 103", "pattern": "structured",
                                         "generate_answer": True}).json()["generation"]
    assert gen["outcome"] == "answered" and len(calls) == 1
    assert calls[0]["messages"][1]["content"].count("<passage ") == 1
    assert gen["citations"][0]["chunk_id"] == "bns:103#section"
    chat = {"model": "rag-structured", "stream": False,
            "messages": [{"role": "user", "content": "BNS section 103"}]}
    text = client.post("/v1/chat/completions", json=chat).json()["choices"][0]["message"]["content"]
    assert "Confidence: high" in text and len(calls) == 2

    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT", httpx.MockTransport(
        lambda r: pytest.fail("provider called")))
    for q, status in (("What does section 103 say?", "clarification_needed"),
                      ("What is the punishment for murder?", "unsupported")):
        r = client.post("/v1/query", json={"question": q, "pattern": "structured",
                                           "generate_answer": True}).json()
        assert r["status"] == status and r["generation"] is None
        chat["messages"][0]["content"] = q
        out = client.post("/v1/chat/completions", json=chat).json()["choices"][0]["message"]["content"]
        assert out == r["message"]
    fake_db(monkeypatch, [])
    missing = client.post("/v1/query", json={"question": "BNS section 999", "pattern": "structured",
                                             "generate_answer": True}).json()
    assert missing["status"] == "not_found" and missing["generation"] is None
