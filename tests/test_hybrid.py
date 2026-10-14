"""Offline tests for hybrid retrieval (Story 4.1): both routes are faked."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from building_with_rag.config import Settings, get_settings
from building_with_rag.generation import answer as answer_module
from building_with_rag.ingestion.mongodb_schema import CHUNKS, keyword_index_definition
from building_with_rag.main import app
from building_with_rag.models import QueryRequest, RetrievedChunk
from building_with_rag.retrieval import RetrievalUnavailable, hybrid, semantic_search
from building_with_rag.retrieval.hybrid import RRF_K, check_keyword, fuse, hybrid_search

client = TestClient(app)
SETTINGS = Settings(_env_file=None, generation_api_base_url="http://llm.test/v1",
                    generation_api_key="k")
TEXT = "Whoever commits murder shall be punished with death or imprisonment for life."


def hit(cid, score):
    return {"chunk_id": cid, "score": score}


def test_fuse_is_reciprocal_rank_fusion():
    rows = fuse([hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)], [hit("b", 9.0), hit("d", 5.0)])
    by = {r["chunk_id"]: r for r in rows}
    assert [r["chunk_id"] for r in rows] == ["b", "a", "d", "c"]  # both routes first
    assert by["b"]["fused_score"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert by["a"]["fused_score"] == pytest.approx(1 / (RRF_K + 1))
    assert (by["a"]["keyword_rank"], by["d"]["semantic_rank"]) == (None, None)
    assert [r["fused_rank"] for r in rows] == [1, 2, 3, 4]
    # a (semantic 1) and d (keyword 2) differ; equal scores break by best rank, then chunk_id
    tie = fuse([hit("y", 1.0)], [hit("x", 1.0)])
    assert [r["chunk_id"] for r in tie] == ["x", "y"]


def test_new_fields_default_to_null():
    c = RetrievedChunk(chunk_id="c", section_id="s", act="A", text="t")
    assert all(getattr(c, f) is None for f in (
        "semantic_score", "semantic_rank", "keyword_score", "keyword_rank", "fused_score",
        "fused_rank"))


class FakeColl:
    def __init__(self, indexes):
        self.indexes = indexes

    def list_search_indexes(self, name=None):
        return iter(self.indexes)


def test_keyword_index_not_queryable_is_unavailable_not_no_results():
    for idx in ([], [{"type": "search", "queryable": False}],
                [{"type": "vectorSearch", "queryable": True}]):
        with pytest.raises(RetrievalUnavailable):
            check_keyword({CHUNKS: FakeColl(idx)})
    check_keyword({CHUNKS: FakeColl([{"type": "search", "queryable": True}])})
    assert keyword_index_definition()["mappings"]["fields"]["text"]["analyzer"] == "lucene.standard"


@pytest.fixture
def fake_routes(monkeypatch):
    chunks = {c: {"_id": c, "chunk_id": c, "section_id": s, "act": "BNS_2023", "text": TEXT,
                  "chapter": "VI", "status": "in_force", "access_level": "public"}
              for c, s in (("c1", "bns:103"), ("c2", "bns:104"), ("c3", "bns:101"))}
    sections = {s: {"_id": s, "section_id": s, "heading": "H", "act_label": "BNS 2023",
                    "source_pdf": "BNS.pdf", "page_start": 7}
                for s in ("bns:103", "bns:104", "bns:101")}
    monkeypatch.setattr(hybrid, "get_db", lambda s: object())
    monkeypatch.setattr(hybrid, "check_quick", lambda db: None)
    monkeypatch.setattr(hybrid, "check_keyword", lambda db: None)
    monkeypatch.setattr(hybrid, "embed_query", lambda q, s: [0.0])
    monkeypatch.setattr(hybrid, "vector_search",
                        lambda *a: ([hit("c1", 0.9), hit("c2", 0.8)], 3))
    monkeypatch.setattr(hybrid, "keyword_search", lambda *a: [hit("c3", 4.0), hit("c1", 3.0)])
    monkeypatch.setattr(hybrid, "load_evidence", lambda db, ids: (chunks, sections))


def test_hybrid_search_result_and_trace(fake_routes):
    res = hybrid_search(QueryRequest(question="murder", pattern="hybrid", limit=2), SETTINGS)
    assert res.pattern == "hybrid" and res.status == "ok"
    first, second = res.results
    assert first.chunk_id == "c1" and first.score == first.fused_score
    assert (first.semantic_rank, first.keyword_rank, first.fused_rank) == (1, 2, 1)
    assert second.chunk_id == "c3" and second.semantic_rank is None  # keyword-only
    t = res.trace[0]
    assert (t["semantic_hits"], t["keyword_hits"], t["in_both_routes"]) == (2, 2, 1)
    assert (t["semantic_only"], t["keyword_only"], t["dropped_model_mismatch"]) == (1, 1, 1)
    assert [(c["chunk_id"], c["routes"], c["returned"]) for c in t["candidates"]] == [
        ("c1", ["semantic", "keyword"], True), ("c3", ["keyword"], True),
        ("c2", ["semantic"], False)]
    assert t["fusion"]["rrf_k"] == RRF_K and "text" not in json.dumps(t["candidates"])


def test_semantic_results_leave_hybrid_fields_null(monkeypatch):
    import building_with_rag.retrieval.semantic as sem

    monkeypatch.setattr(sem, "get_db", lambda s: object())
    monkeypatch.setattr(sem, "check_quick", lambda db: None)
    monkeypatch.setattr(sem, "embed_query", lambda q, s: [0.0])
    monkeypatch.setattr(sem, "vector_search", lambda *a: ([hit("c1", 0.9)], 1))
    chunk = {"_id": "c1", "chunk_id": "c1", "section_id": "s", "act": "A", "text": "t"}
    monkeypatch.setattr(sem, "load_evidence",
                        lambda db, ids: ({"c1": chunk}, {"s": {"section_id": "s"}}))
    res = semantic_search(QueryRequest(question="q", pattern="semantic"), SETTINGS)
    assert res.results[0].score == 0.9 and res.results[0].fused_score is None


def test_hybrid_through_query_and_chat(fake_routes, monkeypatch):
    app.dependency_overrides[get_settings] = lambda: SETTINGS
    body = client.post("/v1/query", json={"question": "murder", "pattern": "hybrid"}).json()
    assert body["pattern"] == "hybrid" and body["trace"][0]["candidates"]

    calls = []

    def handler(request):
        calls.append(request)
        text = "Murder is punished with death or imprisonment for life [E1]."
        frame = "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n\n"
        return httpx.Response(200, content=(frame + "data: [DONE]\n\n").encode())

    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT", httpx.MockTransport(handler))
    gen = client.post("/v1/query", json={"question": "murder", "pattern": "hybrid",
                                         "generate_answer": True}).json()["generation"]
    assert gen["outcome"] == "answered" and len(calls) == 1
    chat = {"model": "rag-hybrid", "stream": False,
            "messages": [{"role": "user", "content": "murder"}]}
    text = client.post("/v1/chat/completions", json=chat).json()["choices"][0]["message"]["content"]
    assert "Confidence: high" in text and len(calls) == 2
    app.dependency_overrides.clear()
    monkeypatch.setattr(answer_module, "DEFAULT_TRANSPORT", None)
