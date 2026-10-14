# Story 4.2: Re-ranking

## Purpose

Make `hybrid-reranked` the third real course mode. Hybrid (Story 4.1) fuses two retrieval routes by rank, which is cheap but never reads the question and a passage together. A re-ranker does: it scores each candidate against the question and reorders them. Retrieval casts a wide net quickly; re-ranking spends one extra model call on a small, bounded set to decide which few passages the answer generator should see.

The mode takes a bounded candidate set from the existing hybrid retrieval, applies **one** documented re-ranking step, and passes only the final selected passages on. It changes nothing else: the same `RetrievedChunk` evidence shape, context assembly, grounded answer, citations, confidence checks, `/v1/query` route, `rag-hybrid-reranked` model ID, Open WebUI streaming path and `.env.example` contract. `semantic` and `hybrid` keep working exactly as they do today.

Out of scope: automatic choice between modes, a comprehensive evaluation or benchmark, advanced RAG patterns (query rewriting, decomposition, HyDE, multi-stage or iterative re-ranking), a new endpoint or UI, a re-ranking framework or library, a different provider contract, re-embedding or re-chunking, a second application, and any change to the Pipe or Filter.

A re-rank score says how relevant the re-ranker judged a passage to the question. It is not proof that the passage is correct or that an answer built from it is right. Say so in the docs and never present it as confidence.

## Prerequisites

- Stories 1.1, 2.1, 2.2, 2.3, 3.1, 3.2 and 4.1 are complete. No `docs/config.yaml` exists, so use the repository root.
- Read before you start: `docs/architecture.md`; Stories 1.1 to 4.1 including their completion records; and the code: `src/building_with_rag/{config,models,registry,main}.py`, `retrieval/{semantic,hybrid,__main__}.py`, `generation/{context,answer,render}.py`, `tests/`, `docs/manual-tests.md`, and `open_webui_functions/capstone_retrieval_controls.py` (read only). Preserve all existing routes, models, settings, tests, the shared registry and uncommitted work.
- Use **exactly** the existing `.env.example` names, defaults and meanings: `RERANK_API_BASE_URL` (`https://api.voyageai.com/v1`), `RERANK_API_KEY` (empty), `RERANK_MODEL_NAME` (`rerank-2.5`), `RERANK_REQUEST_TIMEOUT_SECONDS` (30), `RERANK_CANDIDATE_LIMIT` (20), `RERANK_SEND_LIMIT` (10), `RERANK_RETURN_LIMIT` (5). They are already fields on `Settings`. Add, rename or remove no variable, edit no `.env.example`, add no setting and add no dependency (`httpx` is already a runtime dependency). Fixed bounds go in named code constants. Never print, log or return a key, header, request body or provider response body.
- Do not change the embedding model, dimensions, chunking, `chunks` or `embeddings` data, or either search index. Do not run the embedding load or re-ingest.
- Live checks need the Story 4.1 prerequisites (loaded data, both indexes queryable, `MONGODB_URI`, `VOYAGE_API_KEY`), plus `RERANK_API_KEY` and, for answers, `GENERATION_API_BASE_URL` and `GENERATION_API_KEY`. If any is missing, stop, say which, and mark live verification pending. Do not fake results. The free Voyage key is rate limited, so run live questions one at a time.

## Work to do

### 1. Meaning of the three limits

Fix the stages and what each setting bounds. All three are existing settings; the order is fixed.

1. **Hybrid candidates.** Hybrid retrieval returns its fused list. Take the top `RERANK_CANDIDATE_LIMIT` (default 20) in fused order as the candidate set. Fused candidates beyond that are cut **before** re-ranking.
2. **Send.** Of the candidate set, send the top `RERANK_SEND_LIMIT` (default 10) in fused order to the re-ranker. Candidates beyond that are cut **before** re-ranking and are never sent.
3. **Return.** The re-ranker orders what was sent. Keep the top `min(request.limit, RERANK_RETURN_LIMIT)` (default 5) in re-rank order as the final selected evidence. The rest are cut **after** re-ranking. `request.limit` can only narrow, never exceed `RERANK_RETURN_LIMIT`.

Validate the settings before any retrieval work: `RERANK_CANDIDATE_LIMIT`, `RERANK_SEND_LIMIT` and `RERANK_RETURN_LIMIT` are at least 1, `RERANK_SEND_LIMIT` is at most `RERANK_CANDIDATE_LIMIT`, `RERANK_RETURN_LIMIT` is at most `RERANK_SEND_LIMIT`, and the timeout is positive. A violation is a clear safe 503 naming the variable (see step 5), never a silent clamp. The hybrid fused list is bounded by its own pool (2 × `HYBRID_ROUTE_CANDIDATES`), so the effective candidate count is the smaller of the setting and what hybrid produced; record the effective count in the trace.

### 2. Reuse hybrid retrieval, unchanged in behaviour

Re-ranking consumes the hybrid candidates; it does not search again and does not duplicate hybrid code.

- Extract from `retrieval/hybrid.py`, with the smallest refactor, the part of `hybrid_search` that produces the resolved, fused candidate list and its trace, so that `hybrid_search` is the same function plus the existing `limit` cut and returns exactly the same results, trace and message as today. Its existing tests pass unchanged. The extracted function takes the number of candidates to keep instead of `request.limit`.
- The same readiness checks (vector and keyword indexes), the same server-built filter and the same failure rules apply. Hybrid failure is 503 and the re-ranker is not called.
- If hybrid finds no candidates, the result is `no_results` and the re-ranker is **not** called (nothing to rank). A missing key still fails first (step 5).

### 3. The re-ranking step

Put the code in `retrieval/rerank.py`. One call to the re-ranker per request, no retry, no second pass.

- **Provider contract (do not invent another).** The default provider is Voyage. `POST {RERANK_API_BASE_URL}/rerank` with `Authorization: Bearer <RERANK_API_KEY>`, JSON body `{"query": <question>, "documents": [<candidate texts>], "model": <RERANK_MODEL_NAME>, "top_k": <number sent>}`, timeout `RERANK_REQUEST_TIMEOUT_SECONDS`. The response lists `data` entries with `index` and `relevance_score`. Confirm the exact request and response fields against the current Voyage rerank documentation before coding, and use only documented fields. Use `httpx` and a seam like `DEFAULT_TRANSPORT` in `generation/answer.py` so tests inject a fake transport. Do not add the Voyage SDK call path, a re-ranking library or any provider-specific parameter beyond the documented ones.
- **What is sent.** The query is the request question. Each document is the candidate's chunk `text`, the same text that is shown as evidence, in fused order. Nothing else from the passage goes out. The corpus is public classroom data; say in the docs that passage text and the question leave the app to the re-rank provider.
- **Validating the reply.** Strictly check that every returned `index` is an integer in range and unique, every score is a finite number, and that exactly the number sent came back. Any other reply (wrong shape, duplicates, out-of-range index, missing entries, non-JSON) is a provider failure (step 5), not a partial result. Do not invent, default or fill in a score.
- **Ordering.** Sort by `relevance_score` descending. Break ties by the better (lower) `fused_rank`, then `chunk_id`, so the order is deterministic. `rerank_rank` is the 1-based position among everything that was sent (including those later cut by the return limit); `rerank_score` is the provider's `relevance_score` unchanged.
- **Final evidence.** The top `min(request.limit, RERANK_RETURN_LIMIT)` in that order. Each returned `RetrievedChunk` keeps every field hybrid set (`semantic_*`, `keyword_*`, `fused_*`), gains `rerank_score` and `rerank_rank`, and has `score` set to `rerank_score` (the score that decided its place, as `score` equals `fused_score` in hybrid). Never invent fields. Passage text is never altered.

### 4. Contract: two additive fields and the existing omitted list

- In `models.py`, add to `RetrievedChunk` only `rerank_score` (float) and `rerank_rank` (1-based int), both default `None`. `QueryResult`, `QueryRequest`, `OmittedCandidate` and every other model are unchanged. `semantic` and `hybrid` results leave both `null`.
- Use the existing `QueryResult.omitted_candidates` (`chunk_id`, `omitted_reason`) for **every** candidate cut at any stage, with exactly these reasons: `rerank_candidate_limit` (beyond `RERANK_CANDIDATE_LIMIT`), `rerank_send_limit` (beyond `RERANK_SEND_LIMIT`, not sent), `rerank_return_limit` (sent and re-ranked but outside the return bound, including a smaller `request.limit`). Order: cut-before candidates in fused order, then cut-after in re-rank order. Candidates dropped inside hybrid as unresolved are not candidates; they stay counted in the hybrid trace fields as today.
- `generation/` is not edited. The answer generator receives only `result.results`, the final selected evidence; omitted candidates never reach it. Context assembly's own passage and character limits still apply to what it receives and are still recorded in the generation trace.

### 5. Failure is a clear safe outcome, never a silent fallback

`hybrid-reranked` must never return hybrid order while claiming it was re-ranked, and must never fall back silently to hybrid or semantic order. Use the existing `RetrievalUnavailable` (HTTP 503 on `/v1/query`, OpenAI error envelope on chat, before any stream starts) with a short safe message:

- **Missing key (the explicit outcome):** empty `RERANK_API_KEY` is `RetrievalUnavailable("RERANK_API_KEY is not set.")`, checked first, before any MongoDB or embedding call, mirroring how `VOYAGE_API_KEY` is handled. The default is empty, so a fresh `.env` yields this outcome for `hybrid-reranked` only.
- Empty `RERANK_API_BASE_URL` or `RERANK_MODEL_NAME`, or invalid limits (step 1): a message naming the variable.
- Provider failure: timeout, connection error, non-2xx (include only the status code, for example "HTTP 429"), or an unusable reply, each a distinct short message such as "Re-ranking failed (timed out)." Never include the provider body, headers or key.
- The mode is not a degraded `hybrid`. If a caller wants hybrid order, they choose `rag-hybrid`; nothing chooses for them. There is no automatic routing and no "fall back to hybrid" flag.

### 6. Diagnostic response

`QueryResult.pattern` is `"hybrid-reranked"`, `status` is `ok` or `no_results`, and `message` is one plain sentence with no answer text and no claim of correctness, for example "5 passages selected from 10 re-ranked hybrid candidates; re-rank scores are not proof of correctness."

`trace[0]` is a plain dict (no vectors, no passage text, no key, no URL) that keeps the hybrid trace fields and adds a `rerank` block:

- `rerank`: `model`, `candidate_limit`, `send_limit`, `return_limit` (the configured values), `effective_candidates`, `sent`, `returned`, `request_limit`, and provider `usage` only if the provider reports it as a plain count.
- `candidates`: one compact row per candidate, in **fused order** (the order before re-ranking): `chunk_id`, `section_id`, `routes`, `fused_rank`, `fused_score`, `semantic_rank`/`semantic_score`, `keyword_rank`/`keyword_score`, `sent` (bool), `rerank_rank`, `rerank_score` (null when not sent), `final_rank` (null when not returned), `outcome` (`returned`, `rerank_candidate_limit`, `rerank_send_limit` or `rerank_return_limit`).
- `reranked_order`: the sent candidates' `chunk_id`s in re-rank order (the order after re-ranking, including those cut by the return limit).

From `/v1/query` alone a reader can see the order and scores before and after re-ranking, which passages moved, which were cut and where, and the final selected evidence, without reading code.

### 7. Registry, API and chat wiring (no new path)

- `registry.start_pattern` dispatches `hybrid-reranked` to the new search function through the same branch as `semantic` and `hybrid` and, when `generate_answer` is true, returns the same unstarted `generate_events` iterator over `result.results`. `run_pattern`, the chat handler and `_stream` do not change: one retrieval pipeline (hybrid plus one re-rank call), one generation operation, one chat path. Grounded answer, citations, citation and support checks, confidence, DRAFT rendering and Sources work on the final selected evidence unchanged, and the answer must still cite it.
- Selection is only by the existing registry: model `rag-hybrid-reranked` or `pattern: "hybrid-reranked"`. No new endpoint, no UI change. `GET /v1/models` still lists the same six IDs. The Open WebUI Filter already offers `hybrid-reranked`; do not edit it or the Pipe.
- `structured`, `decomposition` and `hyde` stay placeholders. Tests that use `hybrid-reranked` as the placeholder example (`tests/test_app.py`, `tests/test_chat_streaming.py`) must switch to `structured`; say so in the completion record. `docs/manual-tests.md` row 6 of Story 3.2 and the placeholder loop and chat example that use `hybrid-reranked` change the same way.

### 8. Diagnostic command

Extend `uv run python -m building_with_rag.retrieval` with `--mode hybrid-reranked` (the default stays `semantic`; existing usage is unchanged). `--check` also reports whether `RERANK_API_KEY` is set (`set` / `missing`), never its value. The question run prints the trace counts, then one line per **candidate** in fused order: fused rank/score, rerank rank/score, outcome, `chunk_id`, `section_id`, heading. Compact output only, no passage text beyond the existing `--inspect` excerpt.

### 9. Update `docs/architecture.md`

Extend; do not rewrite earlier sections. Update the **Course modes** table (hybrid-reranked is real since Story 4.2), and the one sentence in the Streamed-answer section that lists which modes are real. In **Configuration**, note that the `RERANK_*` settings are now used by `hybrid-reranked`, with names and defaults unchanged. Add a short "Re-ranking" section recording:

- The flow: hybrid candidates → top `RERANK_CANDIDATE_LIMIT` → top `RERANK_SEND_LIMIT` sent → one re-rank call → top `min(limit, RERANK_RETURN_LIMIT)` → context assembly and generation unchanged.
- The provider contract used (Voyage `/rerank`, fields, strict reply validation), what leaves the app (question and chunk text), and that the key is never logged or returned.
- The result contract: `rerank_score`, `rerank_rank`, `score` meaning, `omitted_candidates` reasons, the trace.
- Outcomes: missing key and provider failure are 503, never a silent hybrid fallback; no results is `no_results` without a re-rank call.
- **Limitations:** the re-rank score is relevance judged by the provider, not correctness or confidence; it can only reorder or cut what hybrid already found, so a passage hybrid missed stays missed; only the top `RERANK_SEND_LIMIT` fused candidates are sent, so a good passage ranked below that is never seen; chunk text only (heading and act label are not sent); one extra network call adds latency and provider cost and can fail or be rate limited; limits were chosen, not tuned; the Voyage reply format is relied on; explicit selection only; no evaluation. Replace "No re-ranking" in the Story 4.1 limitations only by adding a note beside it; do not delete the Story 4.1 text.
- The command `uv run python -m building_with_rag.retrieval "<question>" --mode hybrid-reranked`.

Add one line to `README.md` for the command. Update `docs/manual-tests.md` with a short Story 4.2 section in the style of Story 4.1: the success, no-result, missing-key and answer commands, the model list, the chat command, and the expected results, including the `503` for a missing key. Do not edit `.env.example`.

### 10. Offline tests (few, targeted)

No network, MongoDB, Voyage or re-rank provider. Hybrid candidates and the provider are faked (fake transport for the HTTP call).

- Re-rank order: the final order follows the fake provider's scores, not fused order; `rerank_rank` and `rerank_score` are set, `score == rerank_score`, fused and route fields are kept; ties are deterministic.
- Bounds: with limits smaller than the candidate count, each cut candidate appears once in `omitted_candidates` with the right reason (`rerank_candidate_limit`, `rerank_send_limit`, `rerank_return_limit`); only the send-limit number of documents reach the provider; `request.limit` below the return limit narrows the result.
- Failure: empty `RERANK_API_KEY` raises `RetrievalUnavailable` before any retrieval or provider call; a timeout, non-2xx and a malformed reply (duplicate or out-of-range index, missing entries) each raise `RetrievalUnavailable`, never return hybrid order; no key or provider body appears in the message.
- Contract: `rerank_*` default `null`; `hybrid_search` still returns its existing results and passes its existing tests unchanged.
- `/v1/query` with `pattern: "hybrid-reranked"` returns the trace candidates and `omitted_candidates`; with `generate_answer`, the generator's provider call receives only the final selected passages (check the labelled passages in the request body) and the answer cites them; chat for `rag-hybrid-reranked` streams the real rendered answer; chat for `structured` still streams the placeholder with no retrieval.

## Verification rules

Offline checks may be run by the assistant: `uv run ruff check .`, `uv run pytest -q`, `git status`. The live steps below call MongoDB, Voyage, the re-rank provider and the generation provider. Run them only if those credentials are present and the participant has asked for live verification; otherwise record them as **pending participant**. When run, use the diagnostic command and the in-process or local app, print compact output only, and never show a secret.

## Completion checks

1. `uv run ruff check .` and `uv run pytest -q` pass, including every existing test (updated only as step 7 requires). Record the pass count.
2. **Hybrid and semantic unchanged (live):** `retrieval "What is the punishment for murder?" --limit 5` in `semantic` and in `hybrid` returns the same passage order as the Story 4.1 record (small score drift is not a failure).
3. **Re-ranking works (live):** one `hybrid-reranked` question through `/v1/query` shows `pattern: "hybrid-reranked"`, `rerank_score` and `rerank_rank` on every returned passage, trace `candidates` in fused order with their re-rank ranks and scores, `omitted_candidates` with a reason for every cut candidate, and the number sent equal to `min(RERANK_SEND_LIMIT, effective candidates)`.
4. **Lightweight comparison (live), one question, `hybrid` against `hybrid-reranked`.** Choose a question where ordering plausibly matters: one whose best passage is not clearly first in hybrid, for example a paraphrase such as "What happens to someone who kills another person on purpose?" (in Story 4.1 `bns:101` ranked first in hybrid only because both routes found it, with semantic rank 15). Run each mode once at the same `limit`.
   - **Inspect** the order of returned sections (hybrid `fused_rank` against `rerank_rank`), the sources (which passages each returned, which were cut and why, whether re-ranking brought in a passage hybrid ranked below the final cut, and which moved most), and run one answer in each mode (`generate_answer: true` or one chat message): note outcome, confidence and the cited sections.
   - **Latency and cost, qualitative and only if available:** note roughly how much longer the re-ranked request took (a single wall-clock observation, say it is one run), the number of documents sent, and the provider's reported usage if it returns one. Say plainly if a figure was not available. This is not a benchmark and not a cost model.
   - Record, compactly, the question, both orders, what changed, whether the answers differed in cited sections or outcome, and what you saw. State only what you saw on this one question. Do **not** claim re-ranking is better, or worse, in general, and do not run more questions to build a case. Agreement with hybrid, or a worse order, is a valid finding recorded as is. Do not judge legal correctness beyond whether the cited sections are plausible.
5. **Failure is distinct (live, `.env` untouched):** with `RERANK_API_KEY` empty for that one command, `hybrid-reranked` returns 503 with `RERANK_API_KEY is not set.`, not `no_results`, not hybrid results and not a generated answer. With an invalid `RERANK_API_KEY` or an unreachable `RERANK_API_BASE_URL` (shell-only override), it returns a 503 provider-failure message with no key or body in it. A filter that matches nothing (with a valid key) returns 200 `no_results` and makes no re-rank call.
6. **Chat (live or pending participant):** in Open WebUI, `rag-hybrid-reranked` streams a DRAFT start and a closing `Confidence` block with Sources as `hybrid` does; `semantic` and `hybrid` still behave as before.
7. `GET /v1/models` still lists the six `rag-*` IDs. By code and test inspection: `/v1/query` and chat both go through `start_pattern` / `run_pattern`, one retrieval pipeline and one generation iterator per request, one re-rank call at most, only existing `.env.example` names, no new setting, `structured`, `decomposition` and `hyde` still placeholders, no automatic routing, no fallback to hybrid order on failure, the generator receives only the final selected evidence, `generation/` unchanged.
8. `git status`: `.env` untracked and unchanged, `.env.example`, `data/raw/`, `data/processed/`, the Pipe and the Filter unchanged, no secret in any file or output, no corpus, chunk, embedding or index change, no new dependency.

## Handover

When done, add a **Completion record** to the end of this file with:

- Files created or changed, and the exact commands run with results (counts and statuses only).
- The provider request and response fields as implemented and the documentation page you confirmed them against; the three limits and how `request.limit` interacts; the tie-break; the `omitted_reason` values.
- Tests updated for the `hybrid-reranked` placeholder change, and why; how many tests were added.
- The one-question comparison: the question, both orders (section ids with ranks and scores), what was cut, both answers' outcome and cited sections, and the qualitative latency and cost note with what was unavailable, without a universal claim. Live checks you did not run are listed as **pending participant** with no results written for them.
- Anything unfinished, with the reason.
- Known limitations: re-rank score is not confidence or correctness; it can only reorder or cut hybrid's candidates; only the top send-limit candidates are seen; chunk text only; one extra call adds latency, cost and a failure point with no fallback by design; limits chosen not tuned; explicit selection only; no evaluation, routing or advanced patterns; one demo caller, `public` only.

**For the next story:** `structured`, `decomposition` and `hyde` remain placeholders. Any later mode should reuse `start_pattern`, `RetrievedChunk`, `omitted_candidates` and the unchanged generation path, as `hybrid-reranked` does.

## Completion record

**Files created:** `retrieval/rerank.py`, `tests/test_rerank.py`. **Changed:** `models.py` (`rerank_score`, `rerank_rank` on `RetrievedChunk`), `retrieval/hybrid.py` (candidate step extracted as `hybrid_candidates`; `hybrid_search` is it plus the `limit` cut), `retrieval/{__init__,__main__}.py` (`rerank_search` export, `--mode hybrid-reranked`, `--check` reports `RERANK_API_KEY` set/missing), `registry.py` (dispatch through the same `start_pattern` branch), `tests/{test_app,test_chat_streaming}.py`, `docs/{architecture,manual-tests}.md`, `README.md`. `generation/`, `main.py`, `.env.example`, `pyproject.toml`, the Pipe and the Filter are unchanged.

**Provider contract as implemented:** `POST {RERANK_API_BASE_URL}/rerank`, Bearer `RERANK_API_KEY`, body `query`, `documents` (chunk text, fused order), `model`, `top_k` (number sent); reply `data[].index` / `relevance_score`, optional `usage.total_tokens`. Confirmed against the Voyage rerank API reference (docs.voyageai.com/reference/reranker-api) before coding. Strict reply validation; one attempt, no retry. Limits: candidates = top `RERANK_CANDIDATE_LIMIT`, send = top `RERANK_SEND_LIMIT`, final = `min(request.limit, RERANK_RETURN_LIMIT)`; settings validated before retrieval. Ties: better `fused_rank`, then `chunk_id`. `omitted_reason`: `rerank_candidate_limit`, `rerank_send_limit`, `rerank_return_limit`. `score` equals `rerank_score` in this mode.

**Tests updated:** the tests and manual-test rows that used `hybrid-reranked` as the placeholder example now use `structured`, because `hybrid-reranked` is real. Added 13 offline tests in `tests/test_rerank.py` (8 test functions; the parametrised failure test runs six cases).

**Commands and results:** `uv run ruff check .`: passed. `uv run pytest -q`: 49 passed (36 at Story 4.1 plus the 13 new tests). `uv run python -m building_with_rag.retrieval "test" --mode hybrid-reranked` with `RERANK_API_KEY` unset in `.env`: "Service/configuration problem: RERANK_API_KEY is not set." exit 2, before any MongoDB or Voyage call (check 5, missing-key part).

**Pending participant (no results written):** `RERANK_API_KEY` is missing from `.env`, so every live check that calls the re-rank provider was not run: check 3 (re-ranking through `/v1/query`), check 4 (the one-question comparison, including latency and cost notes), the provider-failure and filter-no-results parts of check 5, and check 6 (Open WebUI chat). Check 2 (semantic and hybrid unchanged live) was run later on 2026-10-14 at the participant's request; see "Live check 2" below. Suggested comparison question: "What happens to someone who kills another person on purpose?".

**Live check 2 (run 2026-10-14, one query at a time):** `semantic`, "What is the punishment for murder?", `--limit 5`: ipc:302, ipc:303, bns:103, bns:104, ipc:396, the same order as the Story 2.3 record (scores 0.8362 to 0.7994). `hybrid`, "What happens to someone who kills another person on purpose?", `--limit 5`: bns:101 (semantic rank 15, keyword rank 6), bns:100, ipc:307, ipc:299, ipc:302, the same order and ranks as the Story 4.1 record. The 4.1 record holds no hybrid order for the murder question, so the paraphrase question was used for hybrid. Result: semantic and hybrid unchanged. Also rechecked offline: `ruff check src tests` passes; `pytest -q` 49 passed. `ruff check .` reports errors only in the untracked tooling folders (`.agents`, `.claude`, `.cline`, `_bmad`), none in project code.

**Still pending participant (needs `RERANK_API_KEY`, which is missing from `.env`):** checks 3, 4, the provider-failure and filter-no-results parts of check 5, and the re-rank part of check 6. No results are written for them.

**Update 2026-10-14 (key now set):** `--check` reports `RERANK_API_KEY: set` and readiness `ready`. `hybrid` on the paraphrase question at `--limit 5` again returned bns:101, bns:100, ipc:307, ipc:299, ipc:302 (about 5 s wall clock, one run). `hybrid-reranked` on the same question returned `Re-ranking failed (HTTP 401).` (exit 2, about 6 s): the provider rejected the key. This is the provider-failure outcome of check 5 working as designed: a clear safe message, no key or body shown, no hybrid fallback. Checks 3, 4, 6 and the filter-no-results part of check 5 stay **pending participant** until a valid re-rank key is in `.env`; no comparison results are written.

**Unfinished:** none in code; the pending live checks above.

**Known limitations:** re-rank score is not confidence or correctness; it can only reorder or cut hybrid's candidates; only the top send-limit candidates are seen; chunk text only; one extra call adds latency, cost and a failure point with no fallback by design; limits chosen not tuned; explicit selection only; no evaluation, routing or advanced patterns; one demo caller, `public` only.
