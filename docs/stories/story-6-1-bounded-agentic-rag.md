# Story 6.1: Bounded Agentic RAG

## Purpose

Make `agentic` (model ID `rag-agentic`) the seventh and final selectable course mode. It is a small, deterministic, read-only controller that picks one of three fixed internal paths and calls at most two of three existing retrieval adapters. It is **not** an LLM planner and **not** a router.

- It runs only when the user explicitly selects `agentic` (`rag-agentic` or `pattern: "agentic"`). Its internal path choice never routes a request from any other mode, and no other mode routes to it.
- Allowlist, fixed and closed: `structuredLookup` (existing `structured_search`), `semanticRetrieve` (existing `semantic_search`) and `hydeRetrieve` (existing `hyde_search`). No other tool, no dynamic registration, no raw database, shell or HTTP tool, no write.
- It validates before it calls, inspects every returned result, makes at most **two** adapter calls and at most **one** HyDE call, and stops at a short configurable application deadline with an explicit trace stop reason.
- Real retrieved corpus passages are the only evidence, answer and citation source. A HyDE hypothetical text is never kept, traced, logged, cited or answered from.

All six existing modes, `POST /v1/query`, `GET /v1/models`, OpenAI-compatible `POST /v1/chat/completions`, response shapes, citations and Open WebUI streaming keep working. The seven modes are additive: nothing existing is replaced.

**Out of scope:** an LLM planner, free-form or dynamic tools, raw DB/shell/HTTP tools, writes, browsing, GraphRAG, controller re-ranking, retries, parallel/background/recursive work, memory, automatic routing, authorization systems, production security or observability, deployment, broad tests, an evaluation harness, any change to the supplied Open WebUI Functions, `provision_open_webui.py`, the corpus, indexes, embedding model or chunking.

A retrieved passage is the stored text of the supplied BNS or IPC document. An answer is not a statement about current law.

## Prerequisites

- Stories 1.1 to 5.2 are complete. Use the repository root (no `docs/config.yaml`).
- Read before you start: `docs/architecture.md`; every story in `docs/stories/` with its completion record (5.1 is the pattern for a new mode and the exact-lookup classifier; 5.2 is the pattern for a mode that gates retrieval, for `MIN_EVIDENCE_SCORE` and for HyDE's answer boundary); and the code: `src/building_with_rag/{config,models,registry,main}.py`, `retrieval/{semantic,structured,hyde,decomposition,model_step,__init__,__main__}.py`, `generation/{context,answer,events,render,checks}.py`, `tests/`, `docs/manual-tests.md`, `.env.example` and `open_webui_functions/{capstone_retrieval_controls,building_with_rag_ui_preview}.py`.
- Read, do not edit, the Open WebUI Functions. The Filter's `UserValves` already offers `agentic` and the Pipe already sends `rag-<pattern>`, so no Function change is needed; confirm this and say so in the completion record. Do not touch `graph` or `reranked`, which stay unsupported there.
- Preserve uncommitted work, the shared registry and every existing test (changed only where step 5 requires).
- Live checks need the same prerequisites as Story 5.2: loaded MongoDB data, `VOYAGE_API_KEY` and the generation proxy. If one is missing, mark that check **pending participant** and do not fake results. The Voyage key is rate limited: space requests out, and treat a rate-limit reply as the adapter failure it is.
- One new setting, and only one: `AGENTIC_DEADLINE_SECONDS` (not a provider variable). Add it to `Settings` (default `45`) and to `.env.example` with that default; no other variable is added, renamed or removed, and no dependency is added. Use named code constants for every other fixed value. Never print, log or return a secret.

## Work to do

### 1. Contract (additive)

- Add `"agentic"` to `Mode` in `models.py`. Registry model IDs, `/v1/models` (now seven IDs) and `MODE_BY_MODEL_ID` follow from it. No other contract change: no new top-level result field, no `QueryRequest` change. `QueryResult.status` is already a free string.
- `QueryResult.pattern` is `"agentic"`. `results` holds the retained real passages. `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates` and `hyde_hypothetical_text_debug` stay empty/null for this mode. The controller trace is `trace[0]`, a plain dict.

### 2. Controller (new `retrieval/agentic.py`, `agentic_search(request, settings) -> QueryResult`)

Plain synchronous code. No LLM call of its own, no thread, no loop beyond the fixed two-step path. A module-level allowlist maps the three tool names to the existing functions; nothing else can be called. Constants: `MAX_CALLS` = 2, `MAX_HYDE_CALLS` = 1, the allowlist, the lay-term list (below), and the score floor (reuse `MIN_EVIDENCE_SCORE` from `decomposition.py`; do not retune it).

**Rules, in this order.**

1. **Validate first, before any tool call.** No MongoDB, embedding or model call happens here. Reject with status `clarify` and stop reason `validation_rejected`:
   - a **bare section number**: the question names a section number with no act (reuse `structured.classify`, which is pure; its `missing_act` result), or is only digits;
   - a **malformed or ambiguous** request: no letters after trimming, two acts, several section numbers (`classify`'s `multiple_acts`, `multiple_sections`), or an aggregation or listing intent (`unsupported_aggregation`, `unsupported_filter`: this mode serves neither).
   The trace records only the validation outcome and a fixed reason code (for example `bare_section_number`, `ambiguous_request`, `unsupported_intent`). The clarify message is a fixed sentence and never echoes the question.
2. **Exact path.** An act-qualified section request (`classify` returns `ok`, `exact_lookup`) calls `structuredLookup` once. Inspect the result: `ok` with one real section is adequate and stops as `ready_for_synthesis`. `not_found`, `clarification_needed` or `unsupported` is `clarify` with stop reason `record_not_found` (or `no_usable_evidence`); never a guessed section, never a semantic fallback.
3. **Vocabulary-mismatch path.** For any other valid question, apply the prepared rule: a small named tuple of everyday words the statute does not use (about 8 to 12 lowercase stems chosen from the Story 5.2 HyDE question and one or two similar cases, for example thief, stole, stolen, phone), matched on word boundaries. Only when it matches, call `semanticRetrieve` once and apply the adequacy rule below. If the direct result is inadequate, and calls and deadline allow, call `hydeRetrieve` **once** after it and apply the HyDE adequacy rule. The trace records that the rule applied and a match count, never the matched request words.
4. **Ordinary path.** Otherwise call `semanticRetrieve` once and apply the adequacy rule. Never call HyDE here.

**Adequacy rules (one per source, stated in code and in the architecture).**
- `semanticRetrieve`: adequate if at least one returned passage is a real passage inside the effective scope with `score >= MIN_EVIDENCE_SCORE`. Only those passages are retained; weaker nearest passages are recorded as ids and scores in the trace and never reach the writer.
- `hydeRetrieve`: adequate if its status is `ok` and it returns at least one real in-scope passage. HyDE's score is a fused rank score, not a similarity, so the floor does not apply to it; say that plainly as a limitation. Retain only those returned real passages, deduplicated with any retained semantic passages by `chunk_id`.

**Outcomes.** Adequate evidence stops as `ready_for_synthesis` (status `ok`). Otherwise:
- `clarify`: validation rejection, structured not-found/ambiguous, or an adapter returned nothing at all (ask the user to rephrase or name a section).
- `partial_result`: an adapter returned real in-scope passages but none was usable (nearest-only; `results` empty, ids and scores only in the trace; message suggests rephrasing or choosing `rag-semantic` to see them).
- `unavailable` (HTTP 200, no generation): the controller stopped on an adapter failure, an invalid result, deadline expiry or the call budget. If usable evidence was already retained, the status is `partial_result` and it is returned; otherwise it is `unavailable`. The reason is the stop reason, never a provider body. This is deliberate and different from other modes (which return 503): the controller's whole purpose is an inspectable trace, and a 503 would discard it. State this in the architecture.

**Per-call guard (before every call).** Allowlisted name; the request already validated; `call_count < MAX_CALLS`; the call is not a second HyDE call; the application deadline has not expired. A failed guard stops with `call_budget_exhausted` or `deadline_expired` and no call is made.

**Scope and constraints.** Every adapter receives a copy of the original request (same `filters`, `caller_id`, `limit`, `chapter`, `required_acts`) with only `generate_answer` forced false. The controller never widens or rewrites `filters`; the existing demo/public scope (`build_filter`'s effective `access_level`) applies unchanged to every call. Adapters keep reporting their own `ignored_fields`; the controller records the union.

**Inspect every result (before it can become evidence).** The result's `pattern` matches the tool called, its `status` is one the tool documents, and every passage has a non-empty `chunk_id`, `section_id` and `text`, is inside the request's filter scope and has `access_level` `public`. Anything else is `tool_invalid` and stops the controller; the passages are dropped.

**Adapter failure.** `RetrievalUnavailable` (or any adapter exception) stops the controller with `tool_unavailable`. No retry, no fallback to another adapter.

**Deadline.** `AGENTIC_DEADLINE_SECONDS` (a positive number; a non-positive or invalid value is a `RetrievalUnavailable` 503 "AGENTIC_DEADLINE_SECONDS must be greater than 0.", checked first). Measured with a monotonic clock from controller start. Checked before every call and again after each call returns; a result that returns after expiry is discarded and the stop reason is `deadline_expired`. A call already running is **not** cancelled (no threads), so the true bound is the deadline plus one adapter's existing timeout. Say so honestly in the architecture; do not claim a hard cap.

**Hypothetical text.** The controller discards `hyde_hypothetical_text_debug` and the adapter's own trace immediately on return. It builds its own trace and result from the retained real passages only. The hypothetical text appears in no trace, evidence, context, answer, citation, log or diagnostic output.

### 3. Controller trace (`trace[0]`, plain dict, non-sensitive)

`step` ("agentic_controller"), `mode`, `validation` (`outcome`, `reason`), `path` (`validation_rejected`, `exact`, `vocabulary_mismatch` or `ordinary`), `mismatch_rule` (`applied`, `matches`), `calls` (one row per call: order, adapter name, `outcome`, adapter status, `passages`, `usable`, retained and weak `chunk_id`s/scores), `call_count`, `hyde_calls`, `adequacy` (one row per decision: after which call, rule name, `adequate`), `stop_reason` (`validation_rejected`, `ready_for_synthesis`, `record_not_found`, `no_usable_evidence`, `tool_invalid`, `tool_unavailable`, `deadline_expired`, `call_budget_exhausted`), `deadline_seconds`, `elapsed_ms`, `returned`, `ignored_fields`. It must **not** contain the question or any text derived from it, `caller_id`, credentials, provider messages or bodies, passage text, or any HyDE hypothetical text. A reader can reconstruct the sequence from `/v1/query` alone.

### 4. Registry, `/v1/query` and chat (no new path)

- `registry.start_pattern` dispatches `agentic` to `retrieval.agentic_search` through the existing retrieval branch (export it from `retrieval/__init__.py`). `_generation_plan` generates for `agentic` only when the status is `ok` (`ready_for_synthesis`), with the original question and `result.results`. Every other status never calls the LLM; chat streams `result.message`.
- One request is one controller run and at most one generation operation. The existing context assembly, grounded answer, citation and support checks, confidence, `DRAFT` rendering, `/v1/query` diagnostics and chat streaming are unchanged. `main.py`, `generation/`, the Pipe, the Filter and `provision_open_webui.py` stay unchanged; if a tiny additive change is truly unavoidable, name it in the record.

### 5. Diagnostic command and tests

- Extend `uv run python -m building_with_rag.retrieval` with `--mode agentic`: compact output only (status, path, stop reason, call sequence, adequacy decisions, retained `section_id`s), never the question, passage text or hypothetical text.
- A few focused offline tests with fake adapters and a fake clock (no network, MongoDB or LLM): validation rejects a bare section number and an ambiguous request with no adapter call and no question in the trace; the act-qualified request calls only `structuredLookup` once, and a missing record is `clarify` with no semantic call; the ordinary question calls only `semanticRetrieve` once and stops `ready_for_synthesis` or `partial_result`; the mismatch rule calls HyDE only after an inadequate semantic result and never twice; no third call is possible; `tool_invalid`, `tool_unavailable` and `deadline_expired` stop with the right reason; the effective filters are the original ones on every call; the hypothetical text appears nowhere in the result or trace; `/v1/models` lists seven IDs and `generation` runs only for `ok`. Update the existing assertions that count six IDs; keep every other test.

### 6. Docs

- **`docs/architecture.md`:** extend, do not rewrite. Course-modes table (`agentic` real since Story 6.1); the six-to-seven ID wording under API surface and the streamed-answer sentence; Configuration (`AGENTIC_DEADLINE_SECONDS` is the only addition, not a provider variable); a short "Bounded agentic RAG" section with the allowlist, fixed rules and order, adequacy rules, outcomes (including the deliberate 200 `unavailable`), scope propagation, trace fields, deadline behaviour and the answer boundary. **Limitations:** deterministic and narrow, not an agent that plans; the mismatch rule is a hand-prepared word list and misses other phrasings; the semantic floor is a chosen guard, not correctness; HyDE adequacy is weaker (no similarity floor); the deadline is cooperative and does not cancel a running call; HyDE can add nothing or add noise; extra latency and Voyage rate-limit exposure; explicit selection only; no routing, retries, loops, memory, graph or evaluation; one demo caller, `public` only. **Commands:** the diagnostic command above.
- **`README.md`:** one line for the `--mode agentic` command, in the existing style; update the "all six modes" sentence.
- **`docs/manual-tests.md`:** add `## Story 6.1 — Bounded Agentic RAG` in the existing style (the four cases below plus one chat stream), and change the smoke section to `## Final smoke check — all seven modes`.

## Verification rules

The assistant may run `uv run ruff check src tests`, `uv run pytest -q` and `git status`. Live checks run only if credentials are present, print compact output (ids, statuses, scores, trace fields) and never show a secret. Open WebUI checks are run by the participant.

## Completion checks

1. `uv run ruff check src tests` and `uv run pytest -q` pass, including every existing test. Record the pass count.
2. **Act-qualified lookup:** `/v1/query` with `pattern: "agentic"` and "BNS section 103". Expect status `ok`, one `bns:103` passage, `calls` = `structuredLookup` only, `call_count` 1, `stop_reason` `ready_for_synthesis`, no embedding or model call.
3. **Ordinary explanation:** an ordinary question with adequate semantic evidence ("What is the punishment for murder?"). Expect only `semanticRetrieve`, one adequacy row `adequate: true`, `ready_for_synthesis`, real `results` only.
4. **Vocabulary mismatch:** "What happens to a person who knowingly buys a phone that a thief stole?". Expect `mismatch_rule.applied: true` and `semanticRetrieve` first. Record honestly whether the direct result was adequate. If it was, the sequence correctly ends there and HyDE was not called. To see the HyDE branch live, inspect for a second lay-vocabulary question whose direct result falls below the floor; expect `semanticRetrieve` then `hydeRetrieve`, `call_count` 2, `hyde_calls` 1. If none can be found, record that the branch was proven by the offline test only. Either way: no hypothetical text in `trace`, `results`, `hyde_hypothetical_text_debug`, `generation`, or the console.
5. **Ambiguous input:** "What does section 103 say?". Expect status `clarify`, `validation` rejected with a fixed reason, **no adapter calls** (`call_count` 0), `stop_reason` `validation_rejected`, and no question text in the trace.
6. **Answer boundary and streaming:** for one answerable case, `generate_answer: true` returns `generation.outcome: "answered"` with citations that are real corpus passages. Chat with `rag-agentic`, `stream: true` ends with `data: [DONE]` and presents grounded answer text with the `Confidence` block and Sources. Participant: select **agentic** in the RAG-options chip in Open WebUI and confirm the same.
7. **Final smoke, all seven modes:** one short question each through `/v1/query` and chat for `semantic`, `hybrid`, `hybrid-reranked`, `structured`, `decomposition`, `hyde` and `agentic`, plus the existing diagnostic command (`--mode` each). Every earlier contract still holds (`hybrid-reranked` may be pending on a valid `RERANK_API_KEY`); `GET /v1/models` lists exactly seven IDs; every chat stream ends `data: [DONE]`. Space requests for the Voyage limit; use only the curl and UI steps in `docs/manual-tests.md`.
8. **Code and test inspection:** the only callable tools are the three allowlisted adapters; at most two calls and one HyDE call on every path; validation precedes any call; scope is propagated unchanged; every result is inspected; no LLM planner, retry, loop, thread, write, routing or memory; the hypothetical text is dropped on return; `main.py`, `generation/`, the Pipe, the Filter and `provision_open_webui.py` unchanged.
9. `git status`: `.env` untracked and unchanged; `data/raw/`, `data/processed/`, indexes and the Open WebUI Functions unchanged; the only `.env.example` change is `AGENTIC_DEADLINE_SECONDS`.

## Handover

When done, add a **Completion record** to the end of this file with:

- Files created or changed; the exact commands run with results (counts and statuses only).
- The final lay-term list, the validation reason codes, the adequacy rules, the score floor used (the reused `MIN_EVIDENCE_SCORE` value) and the `AGENTIC_DEADLINE_SECONDS` default.
- Tests added and which assertions changed for seven modes.
- Live results for checks 2 to 7 that were run (section ids, statuses, call sequence, stop reasons, with an honest note on whether HyDE was ever invoked live); anything not run listed as **pending participant**, with no results written.
- Whether any file outside the list in step 4 needed a change, and exactly what.
- Known limitations as in step 6.

**Course end:** all seven modes are real and additive. A later extension reuses `start_pattern`, `RetrievedChunk` and the unchanged grounded-answer and streaming path, and adds no provider variable without its own story.

## Completion record

**Files created:** `retrieval/agentic.py`, `tests/test_agentic.py`. **Changed:** `models.py` (`Mode` + `agentic`), `config.py` and `.env.example` (`AGENTIC_DEADLINE_SECONDS`, default 45), `registry.py` (dispatch; generation only for `ok`), `retrieval/{__init__,__main__}.py` (export, `--mode agentic`), `tests/test_app.py` (seven model IDs), `docs/{architecture,manual-tests}.md`, `README.md`. Unchanged: `main.py`, `generation/`, the Pipe, the Filter, `provision_open_webui.py` (the Filter already offers `agentic`), data, indexes, `pyproject.toml`.

**Commands:** `uv run ruff check src tests` passed; `uv run pytest -q` 119 passed (103 before, 9 test functions / 16 cases added in `tests/test_agentic.py`).

**Built rules.** Lay terms: thief, thieves, stole, stolen, steal, phone, mobile, scam, scammer, cheated, conned, robbed. Validation reasons: `bare_section_number`, `malformed`, `ambiguous_request`, `unsupported_intent`. Semantic adequacy: a real in-scope passage with score >= 0.74 (reused `MIN_EVIDENCE_SCORE`); HyDE adequacy: status `ok` with at least one real in-scope passage. Deadline default 45 s (cooperative).

**Live (diagnostic command, compact output).**
- Check 2: "BNS section 103" -> `ok`, `structuredLookup` only, 1 call, `ready_for_synthesis`, retained `bns:103`.
- Check 3: "What is the punishment for murder?" -> `ok`, `semanticRetrieve` only, adequate; retained `ipc:302, 303, bns:103, 104, ipc:396`.
- Check 4: the prepared phone question -> rule applied; direct semantic scored below 0.74 (inadequate), so `hydeRetrieve` ran once: 2 calls, 1 HyDE, retained `ipc:411, 413, 412, bns:317, ipc:250` (`ipc:250` looks like noise). A first attempt hit a Voyage rate limit inside HyDE and correctly stopped `tool_unavailable` (status `unavailable`); the retry after a pause succeeded. The HyDE branch was therefore seen live. No hypothetical text was printed.
- Check 5: "What does section 103 say?" -> `clarify`, 0 calls, `validation_rejected` (`bare_section_number`).

**Pending participant:** check 6 (`generate_answer` / chat streaming / Open WebUI), check 7 (seven-mode `/v1/query` and chat smoke, `hybrid-reranked` also needs a valid `RERANK_API_KEY`); the API was not started and no live HTTP calls were made. A transient local OpenBLAS `MemoryError` hit one run; it passed with `OPENBLAS_NUM_THREADS=1`.

**Unfinished:** none in code. **Known limitations:** as in step 6 of this story.

**Update: live HTTP checks (API on port 8001, since an older server held 8000).**
- Check 6: `/v1/query` agentic with `generate_answer` on the murder question returned `answered`, `high`, cited `bns:103`; the ambiguous question returned `clarify` with `generation: null`. Chat `rag-agentic` with `stream: true` ended with `data: [DONE]` and presented `Confidence: high` and Sources (confirmed on a re-run; one earlier streamed call in the smoke run had no Confidence block, probably a transient provider or Voyage rate-limit reply, not reproduced).
- Check 7 (one pass, 25 s spacing): `/v1/query` and chat ok for `semantic`, `hybrid`, `structured`, `agentic`; `decomposition` query ok but its chat call returned 503; `hyde` chat ok but its query returned 503; `hybrid-reranked` 503 on both (the configured `RERANK_API_KEY` is rejected, as in Story 5.2). The 503s on `decomposition` and `hyde` were not diagnosed here and are likely Voyage rate limits; they were not re-run. `/v1/models` lists seven IDs.
- Still **pending participant:** the Open WebUI check, a clean re-run of the `decomposition` chat and `hyde` query, and `hybrid-reranked` with a valid key.

**Update 2: rate-limit re-run (60 to 70 s spacing, API on port 8001).** `hyde` `/v1/query` returned `ok`; `decomposition` chat and `agentic` chat (phone question) each returned 200, ended with `data: [DONE]` and showed `Confidence` and Sources. The earlier 503s were Voyage rate limits. Still **pending participant:** `hybrid-reranked` (the configured `RERANK_API_KEY` is rejected with 401; needs a valid key) and the Open WebUI check.
