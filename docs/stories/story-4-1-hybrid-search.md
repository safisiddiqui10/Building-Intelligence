# Story 4.1: Hybrid Search

## Purpose

Make `hybrid` the second real course mode. Semantic similarity finds passages that mean the same thing as the question, even in different words. It can miss a passage whose value is an exact legal term or section wording (for example "culpable homicide" or a cross-reference such as "section 300"). Keyword search finds those exact words but not paraphrases. Hybrid runs both routes against the same stored chunks and fuses the two candidate lists with one explicit rule, so a passage that either route finds can reach the answer.

Hybrid is a retrieval change only. It reuses everything semantic RAG already uses: the same corpus and `chunks`, the same MongoDB Atlas deployment, the same `RetrievedChunk` evidence shape, the same context assembly, citations and confidence checks, the same `/v1/query` route, the same `rag-hybrid` model ID, the same Open WebUI streaming path and the same `.env.example` contract. `semantic` keeps working exactly as it does today.

Out of scope: re-ranking (a later story), automatic routing between modes, an evaluation harness, a broad regression suite, GraphRAG, agentic behaviour, a new embedding model or re-embedding, regenerating or re-chunking the corpus, a second application, and any change to the Pipe or Filter.

A fused score says how well a passage ranked across two retrieval routes. It is not proof that the passage is correct or answers the question. Say so in the docs and never present it as confidence.

## Prerequisites

- Stories 1.1, 2.1, 2.2, 2.3, 3.1 and 3.2 are complete. No `docs/config.yaml` exists, so use the repository root.
- Read before you start: `docs/architecture.md`; Stories 1.1 to 3.2 including their completion records; and the code: `src/building_with_rag/{config,models,registry,main}.py`, `retrieval/{semantic,__main__}.py`, `ingestion/{mongodb_schema,load_mongodb}.py`, `generation/{context,answer,render,__main__}.py`, `tests/`, and `open_webui_functions/capstone_retrieval_controls.py` (read only). Preserve all existing routes, models, settings, tests, the shared registry and uncommitted work.
- Live checks need the Story 2.3 prerequisites (loaded `chunks` and `embeddings`, queryable `embeddings_vector_index`, `MONGODB_URI`, `VOYAGE_API_KEY`) and, for answers, the `GENERATION_*` values. If any is missing, stop and say which; mark live verification pending. Do not fake results. The free Voyage key is rate limited, so run live questions one at a time.
- Use **exactly** the existing `.env.example` names. Add, rename or remove no variable, edit no `.env.example`, add no setting and add no dependency. Bounds go in named code constants. Never print, log or return a secret, URI or key. The `RERANK_*` variables stay unused here.
- Do not change the embedding model, dimensions, chunking, `chunks` or `embeddings` data, or the vector index. Do not run the embedding load or re-ingest.

## Work to do

### 1. Identify the keyword-search capability first

Before writing hybrid code, find out what the configured deployment supports, and record it. Do not assume.

- The deployment is MongoDB Atlas (Story 2.2 proved `list_search_indexes` and `$vectorSearch`). Keyword search on Atlas is **Atlas Search**: a `search`-type index and the `$search` aggregation stage with the `text` operator. This is a different index type from the `vectorSearch` index, built by the same `create_search_index` call and listed by the same `list_search_indexes`. Confirm against the current MongoDB documentation and PyMongo's `SearchIndexModel`, and use only documented syntax.
- With one compact script, confirm and print (never print documents or secrets): the deployment is reachable; `list_search_indexes()` works on `chunks`; the existing search indexes and their types (the M0 tier has a small cap on search indexes, so count them before creating one); and that a trivial `$search` can run once the index below exists.
- **If keyword search is unavailable** (not Atlas, missing privilege, search-index cap reached, `$search` not supported), stop, report the exact missing prerequisite and what the participant must do, and leave `semantic` unchanged. Do not build a fallback: no regex or `$text` substitute, no in-memory BM25, no second store. `hybrid` then stays the existing `not_implemented` placeholder and the story is recorded as blocked.
- Record the finding in `docs/architecture.md` (step 8).

### 2. Keyword index on the stored chunk text

The keyword route searches **`chunks.text`**, the clean chunk text that is also shown as evidence. Not `embedding_input_text` (it adds the act label and heading, which would make hits depend on text the passage does not contain) and not the parent section text. Consequence to record: the section number and heading live on `sections`, not in `chunks.text`, so a question such as "section 302" matches only passages whose body text contains those words (cross-references do; the section's own number usually does not).

Add to `ingestion/mongodb_schema.py`, beside the vector index, with named constants:

- `KEYWORD_INDEX_NAME = "chunks_text_search_index"`, `KEYWORD_FIELD = "text"`.
- `keyword_index_definition()`: a `search`-type, static-mapping definition on `chunks`: `text` as `string` with the `lucene.standard` analyzer, and `act`, `chapter`, `status`, `access_level` as `token` so the same filters can be applied inside `$search`. These four fields are already denormalized onto `chunks`.
- `keyword_index_model()` returning `SearchIndexModel(definition=..., name=KEYWORD_INDEX_NAME, type="search")`.

Create or verify it with the same safe-repeat rules as the vector index: reuse a matching index, report exactly what differs from a mismatching one and stop, never delete or update it, wait a bounded time for `queryable`. Wire that into `ingestion/load_mongodb.py` as one more idempotent step, so the participant runs the existing command (`uv run python -m building_with_rag.ingestion.load_mongodb --skip-embed` is enough and makes no embedding call). Creating the index reads existing chunks only; it does not rewrite them.

### 3. Contract: add the six hybrid fields

In `models.py`, add to `RetrievedChunk` only these optional fields, default `None`: `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`. Scores are floats, ranks are 1-based integers. Add nothing else (no `rerank_*`, no route field). `QueryResult`, `QueryRequest`, `SemanticFilters` and every other model are unchanged. `semantic` results keep `score` as the vector similarity and leave the six new fields `null`.

Meaning in hybrid results: `score` equals `fused_score`. `semantic_*` is set only if the semantic route returned the chunk; `keyword_*` only if the keyword route did. A `null` rank means that route did not return it. The route that contributed a chunk is therefore readable from which ranks are non-null; also state it in plain words in the trace (step 5).

### 4. Hybrid retrieval and the fusion rule

Put the code in `retrieval/hybrid.py`; import names from `ingestion/mongodb_schema.py`; connect lazily as `semantic` does. Reuse the existing helpers (`get_db`, `embed_query`, `build_filter`, the chunk/section resolution) by extracting them from `semantic.py` with the smallest refactor, so `semantic_search` returns exactly the same results and trace as today and its existing tests pass unchanged. Do not duplicate them.

Flow: validate and resolve the caller (unchanged) → readiness (vector index queryable and embeddings exist, as semantic, plus keyword index `queryable`) → **semantic route**: embed once (`input_type="query"`, same model) and run the existing `$vectorSearch` with the database-side filter → **keyword route**: one `$search` on `KEYWORD_INDEX_NAME`, a `text` operator on `KEYWORD_FIELD` with the question, the same server-built filter applied as `compound.filter` on the token fields, projecting `chunk_id` and `{"$meta": "searchScore"}` → fuse → resolve to `chunks` and `sections` in batches → `RetrievedChunk`s.

- **Same filters.** Both routes use the server-built filter from the validated `act`/`status`/`access_level` lists, narrowing only. Nothing caller-shaped goes into either query. If the effective access list is empty, return `no_results` without searching.
- **Candidate pool.** Each route returns up to `HYBRID_ROUTE_CANDIDATES` (20, a named constant, never below `limit`) candidates with ranks 1…n in that route's score order. The vector route keeps the existing bounded `numCandidates`.
- **Fusion rule: Reciprocal Rank Fusion (RRF), computed in the application.** For each distinct `chunk_id`: `fused_score = Σ over routes that returned it of 1 / (RRF_K + rank_in_route)`, with `RRF_K = 60` and equal route weights (named constants). Rank-based, because a cosine similarity and a keyword (BM25-style) score are on different scales and cannot be added or averaged. Order by `fused_score` descending; break ties by the better (lower) single-route rank, then by `chunk_id` for a deterministic order. `fused_rank` is the 1-based position in that order. Return the top `limit` (1–20, default 5). Do not use `$rankFusion` or `$unionWith` pipelines: the rule stays visible and identical on every Atlas tier.
- **Evidence integrity.** A fused candidate is dropped and counted if its chunk or section cannot be resolved, or (semantic-route hits) its embedding model, version or dimensions differ from the fixed choice, exactly as in semantic. Never invent fields.
- **Failure is not no-results.** A missing credential, unreachable MongoDB, a missing or not-queryable index (vector **or** keyword), a failed `$search`, or an embedding failure is `RetrievalUnavailable`, HTTP 503 with a short safe message naming the problem (chat: the OpenAI error envelope, as today). Do **not** silently fall back to the semantic route alone: a hybrid result must never be semantic-only without saying so. `no_results` means both routes ran and nothing matched. One route legitimately returning zero while the other returns hits is normal and is shown in the trace.
- **Fields from stored data only** (`source`, `source_file`, `page`, `chapter`, `status`, `access_level`, `heading`, `chunk_id`, `section_id`), identical to semantic. `QueryResult.pattern` is `"hybrid"`, `status` `ok` or `no_results`, and `message` is one plain sentence that no answer text and no claim of correctness, for example "5 passages retrieved by semantic similarity and keyword match, fused by reciprocal rank; scores are not proof of correctness."

### 5. Diagnostic response before generation

`QueryResult.trace[0]` is the diagnostic for the combined evidence. Plain dict, no vectors and no passage text:

- query, mode, embedding model and dimensions, `input_type`, `limit`, filters applied, keyword index name and field, `fusion` (`"reciprocal_rank_fusion"`, `rrf_k`, route weights), `route_candidates` per route.
- Counts: semantic hits, keyword hits, in both routes (overlap), semantic only, keyword only, dropped (model mismatch, unresolved), fused total, returned.
- `candidates`: one compact row per fused candidate in fused order (including those cut by `limit`, marked `returned: false`): `chunk_id`, `section_id`, `routes` (for example `["semantic", "keyword"]`), `semantic_rank`, `semantic_score`, `keyword_rank`, `keyword_score`, `fused_rank`, `fused_score`.
- Ignored request fields noted as in semantic (`required_acts`, `chapter`).

A reader should be able to see from `/v1/query` alone which route found each passage, why a keyword-only or semantic-only passage is where it is, and what was cut, without reading code.

### 6. Registry, API and chat wiring (no new path)

- `registry.start_pattern` dispatches `hybrid` to `hybrid_search` and, when `generate_answer` is true, returns the same unstarted `generate_events` iterator over `result.results`, exactly as for `semantic`. `run_pattern`, the chat handler and `_stream` stay as they are: one retrieval, one generation operation, one chat path. Context assembly, the prompt, citations, the citation/support checks, confidence, the DRAFT rendering and the Sources list consume the fused results unchanged (fused order is retrieval order). Do not edit `generation/` unless inspection shows it assumes vector similarity; record anything you had to touch.
- `rag-hybrid` already exists in the registry and `/v1/models`; chat already maps it via `MODE_BY_MODEL_ID`. The mode is selected explicitly (model `rag-hybrid` or `pattern: "hybrid"`). **No automatic routing**: nothing chooses between semantic and hybrid for the caller. The Open WebUI Filter already offers `hybrid`; do not edit it or the Pipe.
- `hybrid-reranked`, `structured`, `decomposition` and `hyde` stay placeholders. Tests that used `hybrid` as the placeholder example (`tests/test_app.py`, `tests/test_chat_streaming.py`) must switch to `hybrid-reranked`; say so in the completion record.

### 7. Diagnostic command

Extend `uv run python -m building_with_rag.retrieval` with `--mode semantic|hybrid` (default `semantic`, so existing usage is unchanged). `--check` also reports the keyword index (exists, type `search`, `queryable`, field `text`). In hybrid mode the question run prints the trace counts, then one line per passage: final rank, fused score, `semantic_rank`/`semantic_score`, `keyword_rank`/`keyword_score`, `chunk_id`, `section_id`, heading, page, text length. Compact output only: no vectors, no passage text beyond the existing `--inspect` excerpt.

### 8. Update `docs/architecture.md`

Extend it; do not rewrite earlier sections or change settings. Update the **Course modes** table note (hybrid is real) and add a short "Hybrid retrieval" section recording:

- The capability found (Atlas Search, `$search` with `text`, `search`-type index) and the stop-and-report rule when unavailable.
- Exact field and index: `chunks.text`, index `chunks_text_search_index`, `lucene.standard`, token filter fields.
- The flow, the shared pieces (corpus, MongoDB, evidence shape, context, citations, `/v1/query`, chat path, `.env.example`), and the fact that nothing was re-embedded or re-chunked.
- **The fusion rule:** RRF, `k = 60`, equal weights, 20 candidates per route, tie-breaks, why rank-based, and why in the application.
- The six new `RetrievedChunk` fields and what `score` means in hybrid; the trace contents.
- **Limitation:** the keyword route sees only `chunks.text`, so section numbers and headings are not keyword-searchable; natural-language questions are matched term by term with no stop-word, phrase or synonym handling, so common words add noise; RRF discards score magnitude; the fused score is not confidence; `k` and pool size were chosen, not tuned; no re-ranking; hybrid costs one extra search call per query.
- Explicit selection only, no automatic routing; failures are 503, never a silent semantic-only result.
- The command `uv run python -m building_with_rag.retrieval "<question>" --mode hybrid`.

Add one line to `README.md` for the command. Do not edit `.env.example`.

### 9. Offline tests (few, targeted)

No network, MongoDB or Voyage. Add only a handful of tests, with the two routes faked:

- Fusion: a chunk in both routes outranks a chunk in one; a keyword-only and a semantic-only chunk both appear; ranks and `fused_score` equal the RRF formula; ties are deterministic; `score == fused_score`.
- Contract: the six fields exist, default `null`, and `semantic` results are otherwise unchanged; `semantic_search` still passes its existing tests.
- A keyword-index-not-queryable condition raises `RetrievalUnavailable` (HTTP 503), not `no_results` and not a semantic-only result.
- Hybrid through `/v1/query` returns `pattern: "hybrid"` with the trace `candidates`; hybrid with `generate_answer` goes through the same single generation iterator (one provider call on a passing attempt).
- Chat for `rag-hybrid` streams the real rendered answer; chat for `rag-hybrid-reranked` still streams the placeholder with no retrieval.

## Verification rules

Offline checks may be run by the assistant: `uv run ruff check .`, `uv run pytest -q`, `git status`. The live steps below call MongoDB, Voyage and the generation provider. Run them only if those credentials are present and the participant has asked for live verification; otherwise record them as **pending participant**. When run, use the diagnostic commands and the in-process or local app, print compact output only, and never show a secret.

## Completion checks

1. `uv run ruff check .` passes and `uv run pytest -q` passes, including every existing test (updated only as step 6 requires). Record the pass count.
2. **Capability and index (live):** the step 1 script's findings (Atlas, search-index count, `$search` runs), `load_mongodb --skip-embed` reports the keyword index created or reused, and `retrieval --check` shows `chunks_text_search_index` as `search` type, `queryable`, field `text`. If keyword search is unavailable, record the prerequisite, confirm `semantic` still works, and stop.
3. **Semantic unchanged (live):** `uv run python -m building_with_rag.retrieval "What is the punishment for murder?" --limit 5` returns the same passages and scores order as the Story 2.3 record (ipc:302, ipc:303, bns:103, bns:104, ipc:396 at the time; small score drift is not a failure).
4. **Small comparison (live), two questions, each run in `semantic` and `hybrid`:**
   - **A paraphrased question**, with no wording from the statute, for example "What happens to someone who kills another person on purpose?".
   - **An exact legal term or section-related question** whose key words you first confirm appear in a stored chunk's text, for example a defined term such as "culpable homicide" or a cross-reference such as "section 300". Do not pick a bare section number the chunk text does not contain.
   For each of the four runs, inspect the retrieved sources (section ids, route ranks/scores, fused rank) with the diagnostic command or `/v1/query`, and for hybrid inspect the trace `candidates`. For each question also run one answer (`generate_answer: true`, or one chat message) in each mode and note outcome, confidence and the cited sections.
   Record in the completion record, compactly: the two questions, which passages each mode returned, which hybrid passages came from one route only, whether the answers differed in cited sections or outcome, and what you observed. State only what you saw on these two questions. Do **not** claim hybrid is better, or semantic is, in general, and do not run more questions to build a case. A result where hybrid and semantic agree, or where hybrid is worse, is a valid finding and is recorded as is.
5. **Failure is distinct (live, one command, `.env` untouched):** with `VOYAGE_API_KEY` set empty for that command, hybrid returns 503 with a safe message, not `no_results` and not a keyword-only result. Optionally, a filter that matches nothing (`act: BNS_2023` with `status: repealed`) returns 200 `no_results`.
6. **Chat (live or pending participant):** in Open WebUI, `rag-hybrid` streams a DRAFT start and a closing `Confidence` block with Sources as `semantic` does; `semantic` still behaves as before.
7. `GET /v1/models` still lists the six `rag-*` IDs. By code and test inspection: `/v1/query` and chat both go through `start_pattern`/`run_pattern`, one retrieval and one generation iterator per request, only existing `.env.example` names, no new setting, `hybrid-reranked`, `structured`, `decomposition` and `hyde` still placeholders, no automatic routing.
8. `git status`: `.env` untracked and unchanged, `.env.example`, `data/raw/` and `data/processed/` unchanged, the Pipe and Filter unchanged, no secret in any file or output, and no corpus, chunk, embedding or vector-index change (only the new search index).

## Handover

When done, add a **Completion record** to the end of this file with:

- The keyword-search capability found, and the exact field and index created or reused (or the prerequisite that stopped the work).
- Files created or changed, and the exact commands run with results (counts and statuses only).
- The fusion rule and constants as implemented.
- Tests updated for the `hybrid` placeholder change, and why.
- The small comparison: both questions, what each mode returned and cited, and the observed difference, without a universal claim. Live checks you did not run are listed as **pending participant** with no results written for them.
- Anything unfinished, with the reason.
- Known limitations: fused score is not confidence; keyword route covers `chunks.text` only; no stop-word, phrase or synonym handling; RRF constants were chosen not tuned; explicit selection only; no re-ranking, evaluation, routing or agentic behaviour; one demo caller, `public` only.

**For the next story (re-ranking, `hybrid-reranked`):** call the same hybrid retrieval and take its fused `results` as the candidate list; add `rerank_score` and `rerank_rank` to `RetrievedChunk` additively, use the existing `RERANK_*` settings, and keep context assembly, citations, `/v1/query`, chat and the streaming path unchanged.
