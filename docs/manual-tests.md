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
  -d '{"model": "rag-hybrid-reranked", "stream": true, "messages": [{"role": "user", "content": "What is the punishment for murder?"}], "rag_options": {"pattern": "hybrid-reranked", "limit": 5}}'

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
7. SSE frames: role frame, content frame (the `hybrid-reranked` placeholder text), stop frame, then `data: [DONE]`.
8. `404` OpenAI error envelope with `error.code: "model_not_found"`.

## Story 2.3 — Semantic Retrieval

**What it adds:** the real `semantic` mode on `/v1/query`, returning ranked source passages in `results` (no generated answer).

Prerequisite: API running (see Story 1.1) with `MONGODB_URI` and `VOYAGE_API_KEY` set in `.env` and the Story 2.2 load complete. The free Voyage key is rate limited, so pause a minute between bursts.

```bash
# 1. Success: semantic query
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "limit": 3}'

# 2. No-result: valid question, filters that match nothing
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "filters": {"act": ["BNS_2023"], "status": ["repealed"]}}'

# 3. Failure: unknown filter field is refused
curl -i -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "pattern": "semantic", "filters": {"$or": []}}'

# 4. Selectable model is still listed
curl http://127.0.0.1:8000/v1/models

# 5. Chat stays a placeholder (does not call retrieval)
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-semantic", "stream": false, "messages": [{"role": "user", "content": "What is the punishment for murder?"}]}'
```

Expected:
1. `200`, `pattern: "semantic"`, `status: "ok"`, up to 3 `results` in descending `score`, each with `chunk_id`, `section_id` (for example `ipc:302`), `heading`, `text`, `source_file` and `page`; `trace[0]` shows the query, limit and filters applied. Scores are similarity, not proof of correctness.
2. `200`, `status: "no_results"`, empty `results`.
3. `422` validation error.
4. Six ids including `rag-semantic`.
5. `200` `chat.completion` whose content is the existing "not implemented yet" placeholder.

If `MONGODB_URI` or `VOYAGE_API_KEY` is missing, or Voyage rate-limits, command 1 returns `503` with `{"detail":"Retrieval unavailable: ..."}`, not `no_results`.

## Story 3.1 — Grounded Answer Generation

**What it adds:** `POST /v1/query` with `"generate_answer": true` on the `semantic` mode returns a grounded, cited answer in `generation`, next to the retrieved `results` (non-streaming; chat stays a placeholder).

Prerequisite: API running (see Story 1.1) with the Story 2.3 setup plus `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` set in `.env` (`GENERATION_MODEL_NAME` defaults to `gpt-4o-mini`). Run the commands one at a time because the free Voyage key is rate limited.

```bash
# 1. Success: grounded answer
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "limit": 5, "generate_answer": true}'

# 2. Edge: question the corpus cannot answer
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the penalty for jaywalking in Delhi under the Motor Vehicles Act?", "pattern": "semantic", "limit": 5, "generate_answer": true}'

# 3. Default: generation is off, so the Story 2.3 shape is unchanged
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "limit": 3}'
```

Expected:
1. `200`, `status: "ok"`, `results` still returned, `generation.outcome: "answered"` with `provider: "openai-compatible"`, a non-empty `answer`, `claims` each citing labels (`E1`…), and `citations` resolved to act-qualified `section_id`s (for example `ipc:302`, `bns:103`) with matching `supporting_passages`. `message` is "Answer generated from N cited of 5 retrieved passages."
2. `200`, `results` still hold the nearest passages, `generation.outcome: "insufficient_evidence"`, `answer: null`, empty `citations`; `message` is "The retrieved passages do not support an answer."
3. `200`, `status: "ok"`, `generation: null`.

If `GENERATION_API_BASE_URL` or `GENERATION_API_KEY` is empty, or the provider times out or errors, commands 1 and 2 still return `200` with `generation.outcome: "unavailable"` and `answer: null` (reason in `generation.trace`); retrieval failures remain `503`.

## Story 3.2 — Streamed Answer, Confidence and Open WebUI

**What it adds:** chat for `rag-semantic` runs the same retrieval and generation as `/v1/query` and streams the answer with DRAFT labels, a confidence block and sources. Run these **in Open WebUI** (`http://127.0.0.1:8080`, model **Building with RAG**, `semantic`), one question at a time (the free Voyage key is rate limited). Not Postman or curl.

| # | Check | Expected |
|---|---|---|
| 1 | "What is the punishment for murder?" | Streams from `DRAFT — checking evidence`; closes with `Confidence: high`, `Checks passed`, Sources incl. `bns:103`. Equivalent: `answered`, `high`, attempts 1–2, citations ≥ 1. |
| 2 | "What is the penalty for jaywalking in Delhi under the Motor Vehicles Act?" | One plain "retrieved passages do not support an answer" line; no draft prefix, no confidence. Equivalent: `insufficient_evidence`, citations 0, results 5. |
| 3 | Force a failure (temporary bad `GENERATION_MODEL_NAME` in the shell, restart, ask check 1) | Labelled `⚠ Answer generation unavailable` line (or `DRAFT — low confidence, not the final answer`); the chat ends. Equivalent: `unavailable`, or `malformed` + `low` + `draft_answer` + citations 0. |
| 4 | Watch check 1 | Text appears incrementally; clean finish; no raw JSON or error toast. |
| 5 | Auth (when `CAPSTONE_API_KEY` is set) | Matching Pipe valve works; wrong/missing shows `Capstone API error (HTTP 401)`. |
| 6 | Choose `hybrid-reranked` | The "not implemented" placeholder. (`hybrid` is real since Story 4.1.) |

## Story 4.1 — Hybrid Search

**What it adds:** the real `hybrid` mode (`rag-hybrid`), which fuses semantic and keyword candidates by reciprocal rank and returns them with per-route ranks and scores; it uses the same `/v1/query` and chat path as `semantic`.

Prerequisite: API running (see Story 1.1) with the Story 2.3 setup, the keyword index created once by `uv run python -m building_with_rag.ingestion.load_mongodb --skip-embed`, and, for answers, `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` in `.env`. Run the commands one at a time because the free Voyage key is rate limited.

```bash
# 1. Success: hybrid retrieval
curl -X POST http://127.0.0.1:8000/v1/query   -H "Content-Type: application/json"   -d '{"question": "culpable homicide not amounting to murder", "pattern": "hybrid", "limit": 5}'

# 2. No-result: valid question, filters that match nothing
curl -X POST http://127.0.0.1:8000/v1/query   -H "Content-Type: application/json"   -d '{"question": "culpable homicide not amounting to murder", "pattern": "hybrid", "filters": {"act": ["BNS_2023"], "status": ["repealed"]}}'

# 3. Failure: unknown filter field is refused
curl -i -X POST http://127.0.0.1:8000/v1/query   -H "Content-Type: application/json"   -d '{"question": "test", "pattern": "hybrid", "filters": {"$or": []}}'

# 4. Grounded answer from the fused passages
curl -X POST http://127.0.0.1:8000/v1/query   -H "Content-Type: application/json"   -d '{"question": "What is culpable homicide not amounting to murder?", "pattern": "hybrid", "limit": 5, "generate_answer": true}'

# 5. Selectable model is listed
curl http://127.0.0.1:8000/v1/models

# 6. Chat, streaming, selecting the hybrid model
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{"model": "rag-hybrid", "stream": true, "messages": [{"role": "user", "content": "What is culpable homicide not amounting to murder?"}], "rag_options": {"pattern": "hybrid", "limit": 5}}'
```

Expected:
1. `200`, `pattern: "hybrid"`, `status: "ok"`, up to 5 `results` ordered by `fused_rank`, each with `score` equal to `fused_score` plus `semantic_rank`/`semantic_score` and `keyword_rank`/`keyword_score` (a `null` rank means that route did not return it). `trace[0].candidates` shows each candidate's `routes`, ranks, scores and whether it was `returned`, with the fusion rule (`reciprocal_rank_fusion`, `rrf_k` 60). Scores are not proof of correctness.
2. `200`, `status: "no_results"`, empty `results`.
3. `422` validation error.
4. `200`, `results` as in 1, `generation.outcome: "answered"` with `citations` resolved to act-qualified `section_id`s and `confidence: "high"` (or the same `insufficient_evidence`, `malformed` or `unavailable` outcomes as Story 3.1/3.2).
5. Six ids including `rag-hybrid`.
6. `200` `text/event-stream`: a role frame, content frames starting with `DRAFT — checking evidence` and closing with a `Confidence` block and Sources, a stop frame, then `data: [DONE]`.

If `MONGODB_URI` or `VOYAGE_API_KEY` is missing, Voyage rate-limits, or the keyword index is missing or not queryable, command 1 returns `503` with `{"detail":"Retrieval unavailable: ..."}`, never `no_results` and never a semantic-only result.
