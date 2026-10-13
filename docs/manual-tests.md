# Manual tests

## Story 1.1 — Architecture and Project Seed

**What it adds:** a FastAPI seed with `/healthz`, `/v1/query`, `/v1/models` and OpenAI-compatible `/v1/chat/completions`, where all six RAG modes return an honest `not_implemented` placeholder.

Prerequisite: start the API with `uv run uvicorn building_with_rag.main:app --port 8000` (no credentials needed). Commands use Git Bash/POSIX quoting.

```bash
# 1. Health
curl http://127.0.0.1:8000/healthz

# 2. Query (success path)
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder?", "pattern": "semantic"}'

# 3. Every mode returns its placeholder
for m in semantic hybrid hybrid-reranked structured decomposition hyde; do
  curl -s -X POST http://127.0.0.1:8000/v1/query \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"test\", \"pattern\": \"$m\"}"; echo
done

# 4. Failure: unknown caller
curl -i -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "pattern": "semantic", "caller_id": "someone-else"}'

# 5. Models
curl http://127.0.0.1:8000/v1/models

# 6. Chat, non-streaming
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-semantic", "stream": false, "messages": [{"role": "user", "content": "What is the punishment for murder?"}]}'

# 7. Chat, streaming
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid", "stream": true, "messages": [{"role": "user", "content": "What is the punishment for murder?"}], "rag_options": {"pattern": "hybrid", "limit": 5}}'

# 8. Chat failure: unknown model
curl -i -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-unknown", "messages": [{"role": "user", "content": "test"}]}'
```

Expected:
1. `200` with `{"status":"ok"}`.
2. `200`, `pattern: "semantic"`, `status: "not_implemented"`, empty `results`, message "The 'semantic' pattern is not implemented yet."
3. Six results, one per mode, each echoing its own `pattern` with `status: "not_implemented"` and empty `results`.
4. `403` with `{"detail":"Unknown caller."}`.
5. Object `list` with exactly six ids: `rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked`, `rag-structured`, `rag-decomposition`, `rag-hyde`.
6. `200` `chat.completion` JSON; `choices[0].message.content` is the `semantic` not-implemented placeholder.
7. SSE frames: role frame, content frame (the `hybrid` placeholder text), stop frame, then `data: [DONE]`.
8. `404` OpenAI error envelope with `error.code: "model_not_found"`.
