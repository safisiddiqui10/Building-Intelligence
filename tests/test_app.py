import json

from fastapi.testclient import TestClient

from building_with_rag.main import app

client = TestClient(app)

MODELS = [
    "rag-semantic",
    "rag-hybrid",
    "rag-hybrid-reranked",
    "rag-structured",
    "rag-decomposition",
    "rag-hyde",
]

PIPE_REQUEST = {
    "model": "rag-semantic",
    "stream": True,
    "messages": [{"role": "user", "content": "What is murder?"}],
    "rag_options": {
        "pattern": "semantic",
        "act": ["BNS"],
        "status": [],
        "access_level": [],
        "limit": 5,
        "required_acts": [],
        "chapter": None,
    },
}


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_query_placeholder():
    r = client.post("/v1/query", json={"question": "What is murder?", "pattern": "semantic"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "not_implemented"
    assert body["results"] == []


def test_query_validation_and_caller():
    assert client.post("/v1/query", json={"question": "", "pattern": "semantic"}).status_code == 422
    assert client.post("/v1/query", json={"question": "x", "pattern": "nope"}).status_code == 422
    r = client.post("/v1/query", json={"question": "x", "pattern": "hyde", "caller_id": "evil"})
    assert r.status_code == 403


def test_models():
    r = client.get("/v1/models")
    assert [m["id"] for m in r.json()["data"]] == MODELS


def test_chat_non_stream():
    r = client.post("/v1/chat/completions", json={**PIPE_REQUEST, "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert "not implemented" in body["choices"][0]["message"]["content"]


def test_chat_stream():
    r = client.post("/v1/chat/completions", json=PIPE_REQUEST)
    assert r.status_code == 200
    frames = [f for f in r.text.split("\n\n") if f]
    assert frames[-1] == "data: [DONE]"
    chunks = [json.loads(f[len("data: "):]) for f in frames[:-1]]
    assert len(chunks) == 3
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["content"]
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"


def test_chat_errors_use_openai_envelope():
    r = client.post("/v1/chat/completions", json={**PIPE_REQUEST, "model": "gpt-4"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"
    r = client.post("/v1/chat/completions", json={**PIPE_REQUEST, "rag_options": {"surprise": 1}})
    assert r.status_code == 400
    assert "error" in r.json()
