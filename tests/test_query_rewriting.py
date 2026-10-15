"""Offline tests for decomposition and HyDE (Story 5.2): the model, embeddings and search are faked."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from building_with_rag.config import Settings, get_settings
from building_with_rag.generation import answer as answer_module
from building_with_rag.main import app
from building_with_rag.models import QueryRequest, RetrievedChunk, SemanticFilters
from building_with_rag.retrieval import RetrievalUnavailable, decomposition, hyde, model_step
from building_with_rag.retrieval.decomposition import decomposition_search, parse_split
from building_with_rag.retrieval.hyde import hyde_search

client = TestClient(app)
QUESTION = "Compare the punishment for murder under the BNS and the IPC"
HYPO = "Whoever dishonestly receives or retains stolen property knowing it to be stolen shall be punished."
ANSWER = "Murder is punished with death or imprisonment for life [E1]."


def make_settings(**kw) -> Settings:
    base = {"generation_api_base_url": "http://llm.test/v1", "generation_api_key": "gen-secret",
            "voyage_api_key": "v", "mongodb_uri": "mongodb://x"}
    return Settings(_env_file=None, **{**base, **kw})


def chunk(cid, score=0.8, act="BNS_2023", text="Whoever commits murder shall be punished with death or imprisonment for life."):
    sid = cid.split("#")[0]
    return RetrievedChunk(chunk_id=cid, section_id=sid, act=act, text=text, heading="Murder",
                          score=score, source="BNS 2023", source_file="x.pdf", page=1)


def model(*replies, calls=None):
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(json.loads(request.content))
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})
    return httpx.MockTransport(handler)


def split(*items):
    return json.dumps({"subquestions": list(items)})


BNS_Q = "What is the punishment for murder under the BNS?"
IPC_Q = "What is the punishment for murder under the IPC?"


@pytest.fixture
def searches(monkeypatch):
    """Fake semantic_search keyed by subquestion text; records each request."""
    seen, table = [], {}

    def fake(request, settings):
        seen.append(request)
        from building_with_rag.models import QueryResult
        return QueryResult(pattern="semantic", status="ok", message="m",
                           results=table.get(request.question, []))

    monkeypatch.setattr(decomposition, "semantic_search", fake)
    fake.table, fake.seen = table, seen
    return fake


# --- model step ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["generation_api_base_url", "generation_api_key",
                                  "generation_model_name"])
def test_missing_generation_setting_is_unavailable_before_any_work(name, searches):
    with pytest.raises(RetrievalUnavailable, match=name.upper()):
        decomposition_search(QueryRequest(question=QUESTION, pattern="decomposition"),
                             make_settings(**{name: ""}))
    assert searches.seen == []


@pytest.mark.parametrize("transport", [
    httpx.MockTransport(lambda r: httpx.Response(502)),
    httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow"))),
    httpx.MockTransport(lambda r: httpx.Response(200, json={"nope": 1}))])
def test_provider_failure_is_unavailable_and_safe(transport):
    with pytest.raises(RetrievalUnavailable) as exc:
        model_step.complete_text("s", "q", make_settings(), transport)
    assert "gen-secret" not in str(exc.value) and "Bearer" not in str(exc.value)


# --- split validation ----------------------------------------------------------------------


@pytest.mark.parametrize("items", [[BNS_Q], [BNS_Q, IPC_Q],
                                   [BNS_Q, IPC_Q, "Is murder punished with a fine in either act?"]])
def test_valid_splits_are_accepted(items):
    assert parse_split(split(*items), QUESTION) == (items, None)


@pytest.mark.parametrize("reply,reason", [
    ("", "empty"), (split(), "empty"), ("not json", "malformed"), ('["a"]', "malformed"),
    (json.dumps({"subquestions": [3]}), "malformed"),
    (split(BNS_Q, IPC_Q, BNS_Q + "?", "Murder punishment fine in BNS?"), "too_many"),
    (split("Murder?"), "invalid_length"), (split(BNS_Q, BNS_Q.upper()), "duplicate"),
    (split("How do I register a company in India?"), "out_of_scope")])
def test_rejected_splits_are_clarify_with_no_search(reply, reason, searches):
    res = decomposition_search(QueryRequest(question=QUESTION, pattern="decomposition"),
                               make_settings(), transport=model(reply))
    assert res.status == "clarify" and res.subquestions == [] and res.results == []
    assert res.trace[0]["decomposition"]["rejection_reason"] == reason
    assert searches.seen == []


# --- decomposition -------------------------------------------------------------------------


def run(searches, reply, req_kw=None):
    req = QueryRequest(question=QUESTION, pattern="decomposition", **(req_kw or {}))
    return decomposition_search(req, make_settings(), transport=model(reply))


def test_all_evidenced_is_ok_with_round_robin_union(searches):
    searches.table[BNS_Q] = [chunk("bns:103#v#0", 0.85), chunk("bns:104#v#0", 0.8)]
    searches.table[IPC_Q] = [chunk("ipc:302#v#0", 0.84, "IPC_1860"), chunk("bns:104#v#0", 0.8)]
    res = run(searches, split(BNS_Q, IPC_Q))
    assert res.status == "ok" and [s.status for s in res.subquestions] == ["evidenced"] * 2
    ids = [c.chunk_id for c in res.results]
    assert ids == ["bns:103#v#0", "ipc:302#v#0", "bns:104#v#0"]  # once each, round-robin
    sub_ids = {c.chunk_id for s in res.subquestions for c in s.results}
    assert sub_ids == set(ids)
    assert [r.limit for r in searches.seen] == [3, 3]
    assert searches.seen[0].filters.act == ["BNS_2023"] and searches.seen[1].filters.act == ["IPC_1860"]


def test_one_weak_subquestion_is_partial(searches):
    searches.table[BNS_Q] = [chunk("bns:103#v#0", 0.85)]
    searches.table[IPC_Q] = [chunk("ipc:302#v#0", 0.3, "IPC_1860")]
    res = run(searches, split(BNS_Q, IPC_Q))
    assert res.status == "partial_answer"
    weak = res.subquestions[1]
    assert (weak.status, weak.reason, weak.results) == ("no_evidence", "below_score_threshold", [])
    assert [c.chunk_id for c in res.results] == ["bns:103#v#0"]
    assert "IPC" in res.trace[0]["outcome_reason"]
    assert res.trace[0]["steps"][1]["below_threshold"][0]["chunk_id"] == "ipc:302#v#0"


def test_none_evidenced_is_clarify(searches):
    res = run(searches, split(BNS_Q, IPC_Q))
    assert res.status == "clarify" and res.results == []
    assert {s.reason for s in res.subquestions} == {"no_passages"}


def test_act_narrowing_never_widens_filters(searches):
    searches.table[BNS_Q] = [chunk("bns:103#v#0")]
    res = run(searches, split(BNS_Q, IPC_Q),
              {"filters": SemanticFilters(act=["BNS_2023"], status=["in_force"])})
    assert [r.filters.act for r in searches.seen] == [["BNS_2023"]]  # IPC step never searched
    assert searches.seen[0].filters.status == ["in_force"]
    assert res.subquestions[1].reason == "outside_filters" and res.status == "partial_answer"


# --- decomposition / hyde wiring -----------------------------------------------------------


def provider_stream(text, calls):
    def handler(request):
        calls.append(json.loads(request.content))
        frames = "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n\ndata: [DONE]\n\n"
        return httpx.Response(200, content=frames.encode())
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _settings():
    app.dependency_overrides[get_settings] = lambda: make_settings()
    yield
    app.dependency_overrides.clear()


def content_of(text):
    frames = [f for f in text.split("\n\n") if f]
    assert frames[-1] == "data: [DONE]"
    return "".join(json.loads(f[6:])["choices"][0]["delta"].get("content") or ""
                   for f in frames[:-1])


def test_decomposition_generation_paths(searches, monkeypatch):
    gen_calls = []
    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT", provider_stream(ANSWER, gen_calls))
    searches.table[BNS_Q] = [chunk("bns:103#v#0", 0.85)]
    body = {"question": QUESTION, "pattern": "decomposition", "generate_answer": True}
    chat = {"model": "rag-decomposition", "stream": True,
            "messages": [{"role": "user", "content": QUESTION}]}

    def go(reply, path="/v1/query", payload=body):
        monkeypatch.setattr(model_step, "DEFAULT_TRANSPORT", model(reply))
        return client.post(path, json=payload)

    # ok: original question, evidenced passages
    searches.table[IPC_Q] = [chunk("ipc:302#v#0", 0.84, "IPC_1860")]
    r = go(split(BNS_Q, IPC_Q))
    assert r.json()["status"] == "ok" and r.json()["generation"]["outcome"] == "answered"
    assert QUESTION in gen_calls[-1]["messages"][-1]["content"]
    assert len(gen_calls) == 1

    # partial: only evidenced subquestions are asked, notice is shown in chat
    searches.table[IPC_Q] = []
    r = go(split(BNS_Q, IPC_Q))
    assert r.json()["status"] == "partial_answer" and r.json()["generation"]["outcome"] == "answered"
    asked = gen_calls[-1]["messages"][-1]["content"]
    assert BNS_Q in asked and IPC_Q not in asked and len(gen_calls) == 2
    text = content_of(go(split(BNS_Q, IPC_Q), "/v1/chat/completions", chat).text)
    assert text.startswith("Partial answer — no corpus evidence found for: " + IPC_Q)
    assert len(gen_calls) == 3

    # clarify: no provider call; chat streams the message
    r = go("not json")
    assert r.json()["status"] == "clarify" and r.json()["generation"] is None
    text = content_of(go("not json", "/v1/chat/completions", chat).text)
    assert "could not be split" in text and len(gen_calls) == 3


def test_decomposition_config_failure_is_chat_error_envelope(monkeypatch):
    app.dependency_overrides[get_settings] = lambda: make_settings(generation_api_key="")
    r = client.post("/v1/chat/completions", json={
        "model": "rag-decomposition", "messages": [{"role": "user", "content": QUESTION}]})
    assert r.status_code == 503 and "GENERATION_API_KEY" in r.json()["error"]["message"]
    assert r.json()["error"]["code"] == "retrieval_unavailable"
    assert [m["id"] for m in client.get("/v1/models").json()["data"]].count("rag-hyde") == 1


# --- HyDE ----------------------------------------------------------------------------------


class FakeHydeEnv:
    def __init__(self, monkeypatch):
        self.embedded, self.by_text = [], {}
        monkeypatch.setattr(hyde, "get_db", lambda s: object())
        monkeypatch.setattr(hyde, "check_quick", lambda db: None)
        monkeypatch.setattr(hyde, "embed_query", self.embed)
        monkeypatch.setattr(hyde, "_search", self.search)

    def embed(self, text, settings):
        self.embedded.append(text)
        return [0.0]

    def search(self, db, settings, text, clauses, limit):
        self.embedded.append(text)
        return self.by_text.get(text, []), len(self.by_text.get(text, [])), 0, 0


def hyde_run(env, reply, **kw):
    req = QueryRequest(question="phone bought from a thief", pattern="hyde", **kw)
    return hyde_search(req, make_settings(), transport=model(reply))


def test_hyde_fuses_real_passages_and_keeps_text_in_debug_only(monkeypatch):
    env = FakeHydeEnv(monkeypatch)
    env.by_text["phone bought from a thief"] = [chunk("ipc:379#v#0", 0.7), chunk("ipc:411#v#0", 0.6)]
    env.by_text[HYPO] = [chunk("ipc:411#v#0", 0.8), chunk("bns:317#v#0", 0.7)]
    res = hyde_run(env, HYPO)
    assert res.status == "ok"
    assert [c.chunk_id for c in res.hyde_direct_candidates] == ["ipc:379#v#0", "ipc:411#v#0"]
    assert [c.chunk_id for c in res.hyde_query_candidates] == ["ipc:411#v#0", "bns:317#v#0"]
    assert res.results[0].chunk_id == "ipc:411#v#0"  # in both routes
    assert len({c.chunk_id for c in res.results}) == len(res.results) == 3
    assert res.hyde_hypothetical_text_debug == HYPO
    dumped = res.model_dump()
    dumped.pop("hyde_hypothetical_text_debug")
    assert HYPO not in json.dumps(dumped)
    assert res.trace[0]["hypothetical"]["chars"] == len(HYPO) and res.trace[0]["overlap"] == 1


@pytest.mark.parametrize("reply,reason", [("", "empty"), ("   ", "empty"), ("short", "malformed"),
                                          ('{"a": "' + "x" * 60 + '"}', "malformed"),
                                          ("I cannot help with that request, sorry about it.", "malformed")])
def test_bad_hypothetical_is_unavailable_with_no_search(monkeypatch, reply, reason):
    env = FakeHydeEnv(monkeypatch)
    res = hyde_run(env, reply)
    assert res.status == "hyde_unavailable" and res.results == []
    assert res.hyde_hypothetical_text_debug is None and res.hyde_direct_candidates == []
    assert res.trace[0]["hypothetical"]["rejection_reason"] == reason and env.embedded == []


@pytest.mark.parametrize("kw", [{"voyage_api_key": ""}, {"mongodb_uri": ""},
                                {"generation_api_key": ""}])
def test_hyde_missing_config_fails_before_model_call(monkeypatch, kw):
    if "mongodb_uri" not in kw:  # an empty MONGODB_URI must hit the real get_db check
        FakeHydeEnv(monkeypatch)
    called = []
    req = QueryRequest(question="x", pattern="hyde")
    with pytest.raises(RetrievalUnavailable):
        hyde_search(req, make_settings(**kw), transport=model(HYPO, calls=called))
    assert called == []


def test_hyde_generation_gets_original_question_and_real_passages_only(monkeypatch):
    env = FakeHydeEnv(monkeypatch)
    env.by_text["phone bought from a thief"] = [chunk("ipc:411#v#0", 0.7)]
    env.by_text[HYPO] = [chunk("ipc:411#v#0", 0.8)]
    gen_calls = []
    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT", provider_stream(ANSWER, gen_calls))
    monkeypatch.setattr(model_step, "DEFAULT_TRANSPORT", model(HYPO))
    payload = {"model": "rag-hyde", "stream": True,
               "messages": [{"role": "user", "content": "phone bought from a thief"}]}
    r = client.post("/v1/chat/completions", json=payload)
    text = content_of(r.text)
    assert "Confidence: high" in text and HYPO not in text
    assert len(gen_calls) == 1
    sent = json.dumps(gen_calls[0])
    assert HYPO not in sent and "phone bought from a thief" in sent

    monkeypatch.setattr(model_step, "DEFAULT_TRANSPORT", model(""))  # hyde_unavailable
    r = client.post("/v1/chat/completions", json=payload)
    assert "no HyDE search ran" in content_of(r.text) and len(gen_calls) == 1
