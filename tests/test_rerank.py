"""Offline tests for hybrid-reranked retrieval (Story 4.2): hybrid and the provider are faked."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from building_with_rag.config import Settings, get_settings
from building_with_rag.generation import answer as answer_module
from building_with_rag.main import app
from building_with_rag.models import QueryRequest, RetrievedChunk
from building_with_rag.retrieval import RetrievalUnavailable, rerank
from building_with_rag.retrieval.rerank import rerank_search

client = TestClient(app)
TEXT = "Whoever commits murder shall be punished with death or imprisonment for life."


def settings(**kw) -> Settings:
    base = {"generation_api_base_url": "http://llm.test/v1", "generation_api_key": "gen-secret",
            "rerank_api_key": "rr-secret", "rerank_api_base_url": "http://rr.test/v1"}
    return Settings(_env_file=None, **{**base, **kw})


def req(**kw) -> QueryRequest:
    return QueryRequest(question="murder", pattern="hybrid-reranked", **kw)


def fake_hybrid(monkeypatch, n=6, calls=None):
    def candidates(request, s, keep):
        if calls is not None:
            calls.append(keep)
        rows, items = [], []
        for i in range(1, n + 1):
            cid = f"c{i}"
            rows.append({"chunk_id": cid, "section_id": f"bns:{100 + i}", "routes": ["semantic"],
                         "semantic_rank": i, "semantic_score": 1 - i / 10, "keyword_rank": None,
                         "keyword_score": None, "fused_rank": i, "fused_score": 1 / (60 + i),
                         "returned": i <= keep})
            items.append(RetrievedChunk(
                chunk_id=cid, section_id=f"bns:{100 + i}", act="BNS_2023", text=f"{TEXT} {cid}",
                heading="H", score=1 / (60 + i), source="BNS 2023", source_file="BNS.pdf",
                page=7, status="in_force", semantic_rank=i, semantic_score=1 - i / 10,
                fused_rank=i, fused_score=1 / (60 + i)))
        return items, {"step": "hybrid_search", "mode": "hybrid", "query": "murder",
                       "candidates": rows, "returned": 0, "dropped_unresolved": 0}

    monkeypatch.setattr(rerank, "hybrid_candidates", candidates)


def provider(scores, seen=None):
    """Fake re-ranker: `scores` maps document position -> relevance_score."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if seen is not None:
            seen.append((request, body))
        data = [{"index": i, "relevance_score": scores[i]} for i in range(len(body["documents"]))]
        return httpx.Response(200, json={"object": "list", "data": data, "model": "m",
                                         "usage": {"total_tokens": 42}})
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    yield
    app.dependency_overrides.clear()
    monkeypatch.setattr(rerank, "DEFAULT_TRANSPORT", None)
    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT", None)


def use(monkeypatch, scores, **kw):
    monkeypatch.setattr(rerank, "DEFAULT_TRANSPORT", provider(scores, **kw))


def test_order_follows_provider_scores_and_keeps_hybrid_fields(monkeypatch):
    fake_hybrid(monkeypatch)
    seen = []
    # c4 best; c1 and c2 tie (fused rank breaks it); c3 lowest
    use(monkeypatch, [0.5, 0.5, 0.1, 0.9, 0.2, 0.3], seen=seen)
    res = rerank_search(req(), settings())
    assert res.pattern == "hybrid-reranked" and res.status == "ok"
    assert [r.chunk_id for r in res.results] == ["c4", "c1", "c2", "c6", "c5"]
    top = res.results[0]
    assert (top.rerank_rank, top.rerank_score, top.score) == (1, 0.9, 0.9)
    assert (top.fused_rank, top.semantic_rank) == (4, 4) and top.fused_score is not None
    body = seen[0][1]
    assert body["model"] == "rerank-2.5" and body["top_k"] == 6 and len(body["documents"]) == 6
    assert seen[0][0].headers["authorization"] == "Bearer rr-secret"
    t = res.trace[0]
    assert t["reranked_order"] == ["c4", "c1", "c2", "c6", "c5", "c3"]
    assert [c["chunk_id"] for c in t["candidates"]] == [f"c{i}" for i in range(1, 7)]
    c3 = t["candidates"][2]
    assert (c3["rerank_rank"], c3["final_rank"], c3["outcome"]) == (6, None, "rerank_return_limit")
    assert [(o.chunk_id, o.omitted_reason) for o in res.omitted_candidates] == [
        ("c3", "rerank_return_limit")]
    assert t["rerank"]["usage"] == {"total_tokens": 42} and "rr-secret" not in json.dumps(t)


def test_bounds_and_omitted_reasons(monkeypatch):
    fake_hybrid(monkeypatch, n=6)
    seen = []
    use(monkeypatch, [0.1, 0.9, 0.2, 0.8, 0.3, 0.7], seen=seen)
    s = settings(rerank_candidate_limit=5, rerank_send_limit=4, rerank_return_limit=3)
    res = rerank_search(req(limit=2), s)  # request.limit narrows below the return limit
    assert len(seen[0][1]["documents"]) == 4
    assert [r.chunk_id for r in res.results] == ["c2", "c4"]
    assert [(o.chunk_id, o.omitted_reason) for o in res.omitted_candidates] == [
        ("c6", "rerank_candidate_limit"), ("c5", "rerank_send_limit"),
        ("c3", "rerank_return_limit"), ("c1", "rerank_return_limit")]
    t = res.trace[0]
    assert t["rerank"]["sent"] == 4 and t["rerank"]["effective_candidates"] == 5
    assert {c["chunk_id"]: c["outcome"] for c in t["candidates"]}["c5"] == "rerank_send_limit"
    # the return limit caps a larger request.limit
    assert len(rerank_search(req(limit=10), s).results) == 3


def test_no_candidates_is_no_results_without_provider_call(monkeypatch):
    monkeypatch.setattr(rerank, "hybrid_candidates", lambda r, s, k: ([], {"candidates": []}))
    monkeypatch.setattr(rerank, "DEFAULT_TRANSPORT", httpx.MockTransport(
        lambda r: pytest.fail("provider called")))
    res = rerank_search(req(), settings())
    assert res.status == "no_results" and res.results == [] and res.omitted_candidates == []


def test_missing_config_fails_before_any_retrieval(monkeypatch):
    calls = []
    fake_hybrid(monkeypatch, calls=calls)
    monkeypatch.setattr(rerank, "DEFAULT_TRANSPORT", httpx.MockTransport(
        lambda r: pytest.fail("provider called")))
    with pytest.raises(RetrievalUnavailable, match="RERANK_API_KEY is not set."):
        rerank_search(req(), settings(rerank_api_key=""))
    for kw, name in (({"rerank_model_name": ""}, "RERANK_MODEL_NAME"),
                     ({"rerank_send_limit": 30}, "RERANK_SEND_LIMIT"),
                     ({"rerank_return_limit": 0}, "RERANK_RETURN_LIMIT"),
                     ({"rerank_request_timeout_seconds": 0}, "RERANK_REQUEST_TIMEOUT_SECONDS")):
        with pytest.raises(RetrievalUnavailable, match=name):
            rerank_search(req(), settings(**kw))
    assert calls == []


@pytest.mark.parametrize("kind", ["timeout", "http", "duplicate", "range", "missing", "json"])
def test_provider_failures_never_return_hybrid_order(monkeypatch, kind):
    fake_hybrid(monkeypatch, n=3)

    def handler(request):
        if kind == "timeout":
            raise httpx.ReadTimeout("slow")
        if kind == "http":
            return httpx.Response(429, text="secret provider body")
        if kind == "json":
            return httpx.Response(200, text="not json secret provider body")
        data = {"duplicate": [{"index": 0, "relevance_score": 1}] * 3,
                "range": [{"index": i + 1, "relevance_score": 1} for i in range(3)],
                "missing": [{"index": 0, "relevance_score": 1}]}[kind]
        return httpx.Response(200, json={"data": data})

    monkeypatch.setattr(rerank, "DEFAULT_TRANSPORT", httpx.MockTransport(handler))
    with pytest.raises(RetrievalUnavailable) as exc:
        rerank_search(req(), settings())
    text = str(exc.value)
    assert text.startswith("Re-ranking failed") and "secret" not in text
    if kind == "http":
        assert "HTTP 429" in text


def test_rerank_fields_default_null():
    c = RetrievedChunk(chunk_id="c", section_id="s", act="A", text="t")
    assert c.rerank_score is None and c.rerank_rank is None


def test_query_generation_and_chat_use_only_final_evidence(monkeypatch):
    fake_hybrid(monkeypatch, n=6)
    use(monkeypatch, [0.1, 0.2, 0.3, 0.4, 0.5, 0.9])  # c6 first, c1 last (cut)
    app.dependency_overrides[get_settings] = lambda: settings()
    body = client.post("/v1/query", json={"question": "murder",
                                          "pattern": "hybrid-reranked"}).json()
    assert body["pattern"] == "hybrid-reranked" and body["trace"][0]["candidates"]
    assert body["omitted_candidates"] == [
        {"chunk_id": "c1", "omitted_reason": "rerank_return_limit"}]

    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        text = "Murder is punished with death or imprisonment for life [E1]."
        frame = "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n\n"
        return httpx.Response(200, content=(frame + "data: [DONE]\n\n").encode())

    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT", httpx.MockTransport(handler))
    gen = client.post("/v1/query", json={"question": "murder", "pattern": "hybrid-reranked",
                                         "generate_answer": True}).json()["generation"]
    assert gen["outcome"] == "answered" and len(calls) == 1
    prompt = calls[0]["messages"][1]["content"]
    assert prompt.count("<passage ") == 5 and "bns:101" not in prompt  # c1 was cut
    assert gen["citations"][0]["chunk_id"] == "c6"
    chat = {"model": "rag-hybrid-reranked", "stream": False,
            "messages": [{"role": "user", "content": "murder"}]}
    text = client.post("/v1/chat/completions", json=chat).json()["choices"][0]["message"]["content"]
    assert "Confidence: high" in text and len(calls) == 2


def test_missing_key_is_503_on_query_and_chat():
    app.dependency_overrides[get_settings] = lambda: settings(rerank_api_key="")
    q = client.post("/v1/query", json={"question": "x", "pattern": "hybrid-reranked"})
    assert q.status_code == 503 and "RERANK_API_KEY is not set." in q.json()["detail"]
    chat = {"model": "rag-hybrid-reranked", "messages": [{"role": "user", "content": "x"}]}
    c = client.post("/v1/chat/completions", json=chat)
    assert c.status_code == 503 and c.json()["error"]["code"] == "retrieval_unavailable"
