# Manual tests

## Story 1.1 — Architecture and Project Seed

**What it adds:** a FastAPI seed with `/healthz`, `/v1/query`, `/v1/models` and OpenAI-compatible `/v1/chat/completions`, where all six RAG modes returned an honest `not_implemented` placeholder. *Historical: every mode is real since Story 5.2, so the placeholder expectations below no longer hold; later sections are current.*

Prerequisite: start the API with `uv run uvicorn building_with_rag.main:app --port 8000` (no credentials needed). Commands use Git Bash/POSIX quoting.

```bash
# 1. Health
curl http://127.0.0.1:8000/healthz

# 2. Query (success path)
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder?", "pattern": "semantic"}'

# 3. (Historical: all six modes were placeholders here; none is since Story 5.2.)

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
  -d '{"model": "rag-decomposition", "stream": true, "messages": [{"role": "user", "content": "What is the punishment for murder?"}], "rag_options": {"pattern": "decomposition", "limit": 5}}'

# 8. Chat failure: unknown model
curl -i -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-unknown", "messages": [{"role": "user", "content": "test"}]}'
```

Expected:
1. `200` with `{"status":"ok"}`.
2. `200`, `pattern: "semantic"`, `status: "not_implemented"`, empty `results`, message "The 'semantic' pattern is not implemented yet."
3. Historical: all six modes returned `not_implemented` here; since Story 5.2 none do.
4. `403` with `{"detail":"Unknown caller."}`.
5. Object `list` with exactly six ids: `rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked`, `rag-structured`, `rag-decomposition`, `rag-hyde`.
6. `200` `chat.completion` JSON; `choices[0].message.content` is the `semantic` not-implemented placeholder.
7. SSE frames: role frame, content frame (the `decomposition` placeholder text), stop frame, then `data: [DONE]`.
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
| 6 | Choose `decomposition` | A real answer or message (Story 5.2). No mode is a placeholder any more. |

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

## Story 4.2 — Re-ranking

**What it adds:** the real `hybrid-reranked` mode (`rag-hybrid-reranked`): hybrid candidates, one re-rank call, then the same grounded answer path. Selected only by model ID or `pattern`; no new endpoint.

Prerequisite: everything from Story 4.1 plus `RERANK_API_KEY` in `.env` (defaults for the other `RERANK_*` names are in `.env.example`). Run the commands one at a time.

```bash
# 1. Success: hybrid candidates re-ranked
curl -X POST http://127.0.0.1:8000/v1/query   -H "Content-Type: application/json"   -d '{"question": "What happens to someone who kills another person on purpose?", "pattern": "hybrid-reranked", "limit": 5}'

# 2. No-result: filters that match nothing (no re-rank call)
curl -X POST http://127.0.0.1:8000/v1/query   -H "Content-Type: application/json"   -d '{"question": "culpable homicide", "pattern": "hybrid-reranked", "filters": {"act": ["BNS_2023"], "status": ["repealed"]}}'

# 3. Missing key: run with RERANK_API_KEY empty (for example, before adding it to .env)
curl -i -X POST http://127.0.0.1:8000/v1/query   -H "Content-Type: application/json"   -d '{"question": "test", "pattern": "hybrid-reranked"}'

# 4. Grounded answer from the final selected passages
curl -X POST http://127.0.0.1:8000/v1/query   -H "Content-Type: application/json"   -d '{"question": "What happens to someone who kills another person on purpose?", "pattern": "hybrid-reranked", "limit": 5, "generate_answer": true}'

# 5. Chat, streaming
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{"model": "rag-hybrid-reranked", "stream": true, "messages": [{"role": "user", "content": "What happens to someone who kills another person on purpose?"}], "rag_options": {"pattern": "hybrid-reranked", "limit": 5}}'
```

Expected:
1. `200`, `pattern: "hybrid-reranked"`, up to 5 `results` ordered by `rerank_rank`, each with `rerank_score` (and `score` equal to it) plus the hybrid fields. `trace[0].candidates` lists every candidate in fused order with its `rerank_rank`/`rerank_score`, `final_rank` and `outcome`; `trace[0].reranked_order` is the order after re-ranking; `omitted_candidates` lists each cut candidate with `rerank_candidate_limit`, `rerank_send_limit` or `rerank_return_limit`. Scores are not proof of correctness.
2. `200`, `status: "no_results"`, empty `results`.
3. `503` with `{"detail":"Retrieval unavailable: RERANK_API_KEY is not set."}`: never hybrid results and never `no_results`.
4. `200`, `generation.outcome: "answered"` with citations that are among the returned (re-ranked) passages, or the same `insufficient_evidence`, `malformed` or `unavailable` outcomes as before.
5. `200` `text/event-stream` as for `rag-hybrid`: DRAFT start, closing `Confidence` block with Sources, stop frame, `data: [DONE]`. A missing key or provider failure is a `503` OpenAI error envelope before any stream starts.

A provider timeout, non-2xx (only the status code is shown) or unusable reply also gives `503` ("Re-ranking failed (...)"), never silently hybrid order.

## Story 5.1 — Structured Exact Retrieval

**What it adds:** the real `structured` mode (`rag-structured`), which reads a question like "BNS section 103" with fixed rules and fetches that one section from MongoDB, with no LLM unless `generate_answer` is true.

Prerequisite: start the API (`uv run uvicorn building_with_rag.main:app --port 8000`) with `MONGODB_URI` set in `.env`; answers also need `GENERATION_API_BASE_URL` and `GENERATION_API_KEY`.

```bash
# 1. Success: exact lookup
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "BNS section 103", "pattern": "structured"}'
```
Expected: `200`, `pattern: "structured"`, `status: "ok"`, one result `bns:103` (`score` null, `source_status_version` set), `generation: null`; `trace[0]` shows the signals and the predicate with `mongodb_called: true`.

```bash
# 2. Clarification: section number without an act (no MongoDB call)
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What does section 103 say?", "pattern": "structured"}'

# 3. No match: well-formed request for a section that is not stored
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "BNS section 999", "pattern": "structured"}'

# 4. Unsupported question (recommendation, no MongoDB call)
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What is the punishment for murder?", "pattern": "structured"}'
```
Expected: 2 returns `200`, `clarification_needed`, empty `results`, `mongodb_called: false`; 3 returns `200`, `not_found`, empty `results`; 4 returns `200`, `unsupported`, empty `results`, a message recommending `rag-hybrid` or `rag-semantic`, `mongodb_called: false`.

```bash
# 5. Grounded answer from the section
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "BNS section 103", "pattern": "structured", "generate_answer": true}'
```
Expected: `200`, `generation.outcome: "answered"` with citations to `bns:103`. With the questions from 2, 3 or 4 there is no generation call (`generation: null`).

```bash
# 6. Chat, streaming, selecting the structured model
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json"   -d '{"model": "rag-structured", "stream": true, "messages": [{"role": "user", "content": "BNS section 103"}], "rag_options": {"pattern": "structured"}}'
```
Expected: `200` `text/event-stream` with a role frame, the streamed answer (DRAFT start, `Confidence` block, Sources) and a stop frame, ending with `data: [DONE]`. For a clarification, no-match or unsupported question the stream carries only the message text.

## Story 5.2 — Query Decomposition and HyDE

**What it adds:** the real `decomposition` mode (`rag-decomposition`: one model split into 1 to 3 required subquestions, each searched on its own) and the real `hyde` mode (`rag-hyde`: a hypothetical passage is embedded only to find real passages).

Prerequisite: start the API (`uv run uvicorn building_with_rag.main:app --port 8000`) with `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` set in `.env`. The free Voyage key is rate limited: wait about a minute between requests.

```bash
# 1. Success: decomposition, BNS-versus-IPC comparison
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "Compare the punishment for murder under the BNS and under the IPC.", "pattern": "decomposition"}'
```
Expected: `200`, `pattern: "decomposition"`, `status: "ok"`, two or three `subquestions` all `evidenced` (BNS passages such as `bns:103`, IPC passages such as `ipc:302`), every passage also in `results`; `trace[0]` shows the model reply, each step and `min_evidence_score`.

```bash
# 2. Edge case: one part the corpus cannot support
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "Compare the punishment for murder under the BNS with income tax evasion under the Income Tax Act.", "pattern": "decomposition", "generate_answer": true}'
```
Expected: `200`, `status: "partial_answer"` (or `clarify`); the unsupported subquestions are `no_evidence` with a `reason` (also in `trace[0].outcome_reason`); any `generation` covers only the evidenced part, never a complete comparison.

```bash
# 3. Success: hyde, vocabulary-mismatch question
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What happens to a person who knowingly buys a phone that a thief stole?", "pattern": "hyde"}'
```
Expected: `200`, `pattern: "hyde"`, `status: "ok"`, `hyde_direct_candidates`, `hyde_query_candidates` and deduplicated real `results` (for example `ipc:411`, `bns:317`); the hypothetical text appears only in `hyde_hypothetical_text_debug`.

```bash
# 4. Grounded answer from real passages (hyde)
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What happens to a person who knowingly buys a phone that a thief stole?", "pattern": "hyde", "generate_answer": true}'

# 5. Chat, streaming, selecting the decomposition model (use "rag-hyde" for HyDE)
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json"   -d '{"model": "rag-decomposition", "stream": true, "messages": [{"role": "user", "content": "Compare the punishment for murder under the BNS and under the IPC."}], "rag_options": {"pattern": "decomposition"}}'
```
Expected: 4 returns `200` with `generation.outcome: "answered"` and citations to real corpus sections. 5 returns `200` `text/event-stream`: role frame, `DRAFT` start, `Confidence` block with Sources, stop frame, `data: [DONE]` (a partial case starts with "Partial answer — no corpus evidence found for: ..."). A missing `GENERATION_*` or `VOYAGE_API_KEY` value is a `503` (chat: OpenAI error envelope before any stream).

## Session 5 smoke check — all six modes

Run one at a time (rate limits); each should keep its earlier contract. `/v1/models` must list exactly the six IDs.

```bash
Q='{"question": "What is the punishment for murder?", "pattern": "%s"}'
for m in semantic hybrid hybrid-reranked; do curl -s -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d "$(printf "$Q" $m)" | head -c 300; echo; sleep 20; done
curl -s -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "BNS section 103", "pattern": "structured"}' | head -c 300; echo
curl -s -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "Compare the punishment for murder under the BNS and under the IPC.", "pattern": "decomposition"}' | head -c 300; echo
sleep 60
curl -s -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What happens to a person who knowingly buys a phone that a thief stole?", "pattern": "hyde"}' | head -c 300; echo
curl -s http://127.0.0.1:8000/v1/models
for m in semantic hybrid hybrid-reranked structured decomposition hyde; do curl -s -N -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"rag-$m\", \"stream\": true, \"messages\": [{\"role\": \"user\", \"content\": \"BNS section 103\"}]}" | tail -n 2; sleep 60; done
```
Expected: `semantic` and `hybrid` `status: "ok"` with passages; `hybrid-reranked` re-ranked passages (or `503` "RERANK_API_KEY is not set." when that key is absent); `structured` `ok` for `bns:103`; `decomposition` and `hyde` as above; every chat stream ends with `data: [DONE]`. Then pick each of the six in Open WebUI once.
