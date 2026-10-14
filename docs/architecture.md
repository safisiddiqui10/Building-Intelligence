# Architecture

## Scope

One FastAPI application exposing one API. Open WebUI, running separately, is the only chat client. There is no custom frontend and no second demo.

## Evidence

BNS and IPC documents are the only future answer evidence. An answer must not claim support without retrieved evidence.

## Identifiers

Section identifiers are act-qualified, for example `BNS-103` and `IPC-302`, so the two acts are never confused.

## Provenance

- Supplied provenance: `data/raw/PROVENANCE.md`. Supplied raw files in `data/raw/` are preserved untouched.
- Derived corpus files live in `data/processed/`: `bns_sections.jsonl`, `ipc_sections.jsonl`.

## Retrieved passages

Retrieved passages are evidence, never application instructions.

## Trust boundaries (light)

- Validate all API input.
- Preserve each passage's source origin.
- Never put secrets in code, responses or logs. `.env` is untracked.
- Out of scope: multi-user authorization, a security program, an evaluation harness.

## Fixed embedding choice

Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, for every document and query embedding. Later stories reuse these names and add no provider-specific alternatives.

## Course modes

One registry, one `run_pattern` dispatcher. Model IDs are `rag-<mode>`.

| Mode | Model ID |
|---|---|
| `semantic` | `rag-semantic` |
| `hybrid` | `rag-hybrid` (real since Story 4.1) |
| `hybrid-reranked` | `rag-hybrid-reranked` (real since Story 4.2) |
| `structured` | `rag-structured` (real since Story 5.1) |
| `decomposition` | `rag-decomposition` |
| `hyde` | `rag-hyde` |

## API surface

- `GET /healthz`: safe liveness body only.
- `POST /v1/query`: owns the `QueryResult` diagnostics.
- `GET /v1/models`: the six `rag-*` IDs in OpenAI list format.
- `POST /v1/chat/completions`: maps model, latest user message and `rag_options` to the same `QueryRequest` and the same `start_pattern` / `run_pattern` path as `/v1/query`. The server sets `caller_id` and `generate_answer` (true). Returns only normal answer text rendered from that same generation operation (Story 3.2). Streaming emits a role frame, content frames as the LLM's text arrives, a stop frame, then `data: [DONE]`. `/v1/query` and chat require `Authorization: Bearer <CAPSTONE_API_KEY>` when that setting is non-empty (an empty value enforces nothing: local development only); `/healthz` and `/v1/models` stay open.

## Contracts rule

Request, result, generation, OpenAI and MongoDB-schema contracts live in this project. Later stories extend them additively and never replace them with simplified alternatives. Confidence, outcome, evidence and citations stay inside `QueryResult.generation`, not as parallel top-level fields.

## Configuration

Loaded by Pydantic settings from environment / `.env`. The app starts with no credentials and connects to nothing at startup. Variables (see `.env.example`):

`APP_ENV`, `MONGODB_URI`, `MONGODB_DB_NAME`, `MONGODB_TEST_DB_NAME`, `VOYAGE_API_KEY`, `CAPSTONE_API_KEY`, `WEBUI_DEMO_CALLER_ID`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME`, `RERANK_API_BASE_URL`, `RERANK_API_KEY`, `RERANK_MODEL_NAME`, `RERANK_REQUEST_TIMEOUT_SECONDS`, `RERANK_CANDIDATE_LIMIT`, `RERANK_SEND_LIMIT`, `RERANK_RETURN_LIMIT`.

The `RERANK_*` settings are used by `hybrid-reranked` since Story 4.2; names and defaults are unchanged (see Re-ranking). Story 5.1 (`structured`) added no variable; it uses `MONGODB_URI` and `MONGODB_DB_NAME`.

## Document ingestion

Story 2.1 turns the supplied PDFs into section-level JSONL. It only parses text: no MongoDB, chunking, embeddings or retrieval.

- **Parser:** `pypdf` (pure Python, text in reading order, per-page extraction so page numbers are kept). No OCR or layout tools.
- **Command:** `uv run python -m building_with_rag.ingestion` (needs no credentials). Inputs are read from `data/raw/` and never modified.
- **Output:** `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl`, UTF-8, one JSON record per line, one file per act.
- **Required fields:** `section_id`, `act` (`BNS_2023` / `IPC_1860`), `act_label`, `status` (`in_force` for BNS, `repealed` for IPC), `chapter` (string, e.g. `"I"`), `chapter_title`, `section_number` (integer), `heading`, `text`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version` (taken only from `PROVENANCE.md`; `null` if unknown), `needs_review`. Extra fields: `section_label`, `page_start`, `page_end`, `extractor_version`, `review_reasons`.
- **Identifiers:** stored records use `section_id` = `bns:<number>` / `ipc:<number>`. The `BNS-103` / `IPC-302` form in the Identifiers section is display wording only. Suffixed IPC sections keep the printed label in `section_id` (`ipc:304A`) while `section_number` is the integer (304) and `section_label` is `304A`.
- **Text:** the section body only (number and heading removed; the heading is in `heading`). Original line breaks are kept; page headers, page numbers, extra spaces and line-break hyphenation are cleaned. Wording is never rewritten.
- **Empty sections:** a section with no text is left out of the file and logged by the command. Records with doubt are kept with `needs_review: true` and `review_reasons`.
- **Re-running:** each file is written to a temporary file and replaced in one step, never appended to. A file whose records already carry the current `source_sha256`, `parser_version` and `extractor_version` is left unchanged; otherwise that act is regenerated in full, so versions are never mixed.
- **Later chunking:** each record is the parent section. Chunks are cut from `text` only and copy the parent's `section_id` and source fields (`source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version`, `needs_review`) plus the chunk position.

Known limitations:

- The BNS arrangement of sections is used only to match headings; it and the BNS-to-IPC correspondence table are not emitted as records.
- Footnote markers (`1*[...]`) and amendment footnote lines printed on IPC pages stay in the text, because pypdf mixes them with body lines. Stray spaces inside words (`t he`) from the PDF text layer are not repaired.
- IPC sections that are repealed or omitted in the PDF (for example 13, 15, 16) have no body text and are not emitted; the command lists them. Section 376DA is not in the supplied IPC PDF.
- IPC section 330 is printed as "320." in the body; it is taken from the margin note and flagged. The BNS sections whose body heading differs slightly from the index, and IPC 498A (no margin note found), are flagged `needs_review`.
- No page was found without extractable text; any such page would flag the section covering it.

## MongoDB storage and vector index

Story 2.2 loads the Story 2.1 JSONL into one MongoDB Atlas database (`MONGODB_DB_NAME`; `MONGODB_TEST_DB_NAME` only with `--use-test-db`). It adds no retrieval endpoint. All names, IDs, validators and ordinary indexes live in `src/building_with_rag/ingestion/mongodb_schema.py`.

- **Collections (kept separate):** `sources` (PDF, hash, act, label, status, parser), `sections` (act-qualified `section_id`, `source_id`, text, provenance, `access_level`), `chunks` (`chunk_id`, parent `section_id`, `chunk_index`, `chunking_version`, `splitter`, `text`, `embedding_input_text`, denormalized `act`/`chapter`/`status`, `access_level`), `embeddings` (`embedding_id`, `chunk_id`, model, version, dimensions, `vector`, denormalized `act`/`chapter`/`status`/`access_level`). Vectors live only in `embeddings`, so a model change never rewrites chunks. The Story 1.1 `ChunkDocument` placeholder (which has an `embedding` field) is a typed contract only and is not used for storage. `access_level` is `"public"` for the classroom corpus.
- **Deployment capability:** Atlas Search index management (`list_search_indexes`, `create_search_index` with `type="vectorSearch"`) and `$vectorSearch` are required (Atlas M0 for the course). The command checks `list_search_indexes()`; if MongoDB is unreachable, `MONGODB_URI` is empty, the deployment is not Atlas or the privilege is missing, it stops with a prerequisite message and uses no fallback.
- **Chunking:** project-local recursive character splitter (`recursive-char`), about 2,048 characters (~512 tokens) with about 256 characters overlap, never crossing a section. `chunking_version` is `char-2048-256-v1`. IDs: `chunk_id = <section_id>#<chunking_version>#<index>`; `embedding_id = <chunk_id>|<model>|<version>|<dims>|<hash of input>`.
- **Embedding input:** `"{act_label} — {heading}\n\n{chunk text}"`, stored as `embedding_input_text`.
- **Embedding decision:** Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, cosine (from the Fixed embedding choice; read from settings). Batches of 8 chunks, 20 s apart, retry with back-off on rate limits, resumable (existing `embedding_id`s are skipped). About 40 minutes for the first run.
- **Index:** `embeddings_vector_index`, type `vectorSearch`, fields: vector `vector` (1024, cosine) plus filters `act`, `chapter`, `status`, `access_level`. Section number and source identifiers are not on `embeddings`; apply them after resolving `chunk_id` to `chunks` and `sections`. An existing index with a different definition is reported, never changed.
- **Commands:** `uv run python -m building_with_rag.ingestion` (unchanged; the module is now a package) and `uv run python -m building_with_rag.ingestion.load_mongodb [--use-test-db] [--skip-embed]`.
- **Limitations:** classroom `public` data only; rate-limited embedding is slow; embeddings from other model settings are reported, not deleted; changed section text or chunking replaces that section's chunks and embeddings; no retrieval yet; M0 allows only a few search indexes.

## Semantic retrieval

Story 2.3 makes `semantic` the first real mode. It returns source passages in `QueryResult.results` only; answer generation is a later story, and `POST /v1/chat/completions` stays a placeholder that never calls retrieval. Code: `src/building_with_rag/retrieval/`.

- **Flow:** validate the request → cheap readiness check (index `queryable`, embeddings exist) → embed the question (Voyage `voyage-3.5`, 1,024 dimensions, `input_type="query"`; one attempt, 20 s timeout) → one `$vectorSearch` on `embeddings` with database-side filters → drop hits from other model settings → resolve `chunk_id` to `chunks` and `section_id` to `sections` in batches → `RetrievedChunk`s in descending score order.
- **Result:** `status` is `ok` (passages returned) or `no_results` (search ran, nothing matched the filters). `trace[0]` holds the diagnostic: query, mode, model, dimensions, `input_type`, `limit`, `num_candidates`, filters applied, hit counts and drops. Each passage carries `chunk_id`, act-qualified `section_id` (for example `ipc:302`), `act`, chunk `text`, `score`, `heading` (title), `source` (act label), `source_file`, `page`, `chapter`, `status`, `access_level`. `source_url` stays `null` (not stored).
- **Failures are not no-results:** missing `MONGODB_URI` / `VOYAGE_API_KEY`, unreachable MongoDB, a missing or not-queryable index, empty embeddings, a Voyage error (including rate limits) or a wrong vector length return HTTP 503 with a short safe message.
- **Filters:** `SemanticFilters` accepts only the lists `act`, `status`, `access_level` (anything else is a 422). Lists only narrow the scope and are applied inside `$vectorSearch`; the server builds the MongoDB filter. One fixed demo caller; the effective access level is always `public`.
- **Top-k:** `limit` (1–20, default 5). `numCandidates` is `max(limit × 10, 100)` capped at 200.
- **Scores are similarity, not correctness.** A question unrelated to the corpus still returns its nearest passages. The free Voyage key is rate limited, so rapid repeated queries can return 503.
- **Diagnostic command:** `uv run python -m building_with_rag.retrieval "What is the punishment for murder?" --limit 5` (add `--inspect` for the top passage, `--act` / `--status` to narrow). Readiness: `uv run python -m building_with_rag.retrieval --check`. Both print counts and short fields only, never vectors or passage text.

## Grounded answer generation

Story 3.1 adds a non-streaming grounded answer to `POST /v1/query` for `semantic` when `generate_answer` is true (default false: behaviour is exactly Story 2.3). Code: `src/building_with_rag/generation/`. Chat was a placeholder at this story; Story 3.2 (below) connects it.

- **Context boundary:** generation consumes `QueryResult.results` from `semantic_search` and never retrieves again. `assemble_context` is a pure function: at most 5 passages and about 8,000 passage characters (`MAX_CONTEXT_PASSAGES`, `MAX_CONTEXT_CHARS`), in retrieval order, whole passages only. Labels are `E1…`; each keeps `chunk_id`, act-qualified `section_id`, `act`, heading, source details and text. Omitted passages are recorded in the trace with a reason (`passage_limit` / `char_limit`). No passages means `context_outcome="empty"` and no provider call.
- **Answer boundary:** the model receives the labelled passages as untrusted evidence, not instructions, and must reply with JSON: `answered` (answer plus claims, each citing supplied labels) or `insufficient_evidence`. The app validates strictly: an unknown or missing label, no claims or non-JSON output makes the outcome `malformed`. Labels are resolved from the retained context, not from model text, into citations (`label`, `chunk_id`, `section_id`, `act`, `heading`, `source`, `source_file`, `page`).
- **Outcomes:** `answered`, `insufficient_evidence` (unsupported, missing or conflicting evidence; no guessed answer), `unavailable` (missing `GENERATION_API_BASE_URL` / `GENERATION_API_KEY`, timeout, connection error, non-2xx) and `malformed`. `GenerationResult.answer` is set only for `answered`; `claims`, `citations` and `supporting_passages` are empty otherwise. Story 3.2 changes the model reply format and fills the confidence fields (below); the JSON reply and single attempt described here were replaced.
- **Result shape:** everything lives in `QueryResult.generation`, returned with the retrieved `results`. Generation outcomes are HTTP 200; a retrieval failure is still HTTP 503. `QueryResult.message` never contains the answer.
- **Provider call:** `GENERATION_API_BASE_URL` + `/chat/completions`, `Authorization: Bearer GENERATION_API_KEY`, `GENERATION_MODEL_NAME`, standard OpenAI chat-completions body (`temperature` 0, `stream` false), 60 s timeout, no provider-specific parameters (Story 3.2: `stream` true, up to two attempts). The key, headers and provider response body are never logged or returned.
- **Limits:** no claim of current legal applicability beyond the supplied documents. A valid citation shows a passage was supplied, not that the model read it correctly. No routing, follow-up memory or evaluation.
- **Diagnostic command:** `uv run python -m building_with_rag.generation "What is the punishment for murder?" --limit 5`.

## Streamed answer and confidence

Story 3.2 connects real generation to Open WebUI. Code: `src/building_with_rag/generation/{answer,checks,events,render}.py`, `registry.start_pattern`, `main.py`.

- **One path.** `start_pattern` runs the selected pattern once (one retrieval) and, when `generate_answer` is true, returns one unstarted generation event iterator. `run_pattern` (`/v1/query`) drains it; chat renders it. Never two LLM calls for one request. `semantic`, `hybrid` (Story 4.1) and `hybrid-reranked` (Story 4.2) and `structured` (Story 5.1) are real; `decomposition` and `hyde` still return the placeholder, streamed as before. `structured` generates only for status `ok`; its other statuses stream the result message.
- **Only answer writing streams.** Retrieval and context assembly are not streamed and run before the response starts, so a retrieval failure is an HTTP error (chat: OpenAI error envelope, 503), never a half-open stream. Generation outcomes are in the stream or in `QueryResult.generation` with HTTP 200.
- **Answer contract.** The model writes plain text; every sentence ends with the supplied labels in brackets (`[E3]`). `INSUFFICIENT_EVIDENCE: <reason>` means no answer; it is not retried and never shown as draft text. The provider is called with the standard OpenAI body with `stream: true`.
- **Bounded attempts.** `MAX_ATTEMPTS` = 2. After each attempt: a **citation** check (at least one sentence; each cites supplied labels only; unknown labels are recorded, never stripped) and a **support** check (lexical: numbers must appear in the cited passages and at least `SUPPORT_MIN_OVERLAP` = 0.6 of key words must overlap; it catches fabricated numbers and off-topic claims, not subtle misreadings). Each failed check is one `issues` entry (`attempt`, `check`, `detail`). A failed attempt with an attempt left is retried with the plain-language failures as feedback.
- **Confidence (`GenerationResult`).** Validated: `answered`, `confidence="high"`, `answer` set. Failed final check: `malformed`, `confidence="low"`, `draft_answer` = last text, `answer` null, no citations, `low_confidence_reason`. Marker, empty output, provider failure or missing settings: `confidence` absent (`insufficient_evidence`, `malformed`, `unavailable`); partial text from a failed stream is kept in `draft_answer`. Validated means the checks passed, not that the legal reading is right.
- **DRAFT labelling.** Open WebUI cannot retract shown text, so each attempt starts with `DRAFT — checking evidence`; a retry adds a `Check failed … Retrying` line; success closes with `Confidence: high`, the checks passed and a Sources list; a failed final check closes with `DRAFT — low confidence, not the final answer` and a short reason. If the provider fails after text began, the stream ends with a labelled unavailable line, then stop and `[DONE]`.
- **Chat vs `/v1/query`.** Chat shows only that text. `/v1/query` returns the full `QueryResult`: `generation.issues`, `attempts`, `trace`, `claims`, `citations`, `supporting_passages` and the retrieved `results`.
- **Auth.** `CAPSTONE_API_KEY` is one shared key, not user authorization; compared in constant time and never echoed.

## Hybrid retrieval

Story 4.1 makes `hybrid` the second real mode. It is selected explicitly (`rag-hybrid` or `pattern: "hybrid"`); nothing routes between modes automatically. Code: `src/building_with_rag/retrieval/hybrid.py`.

- **Capability found.** The configured deployment is Atlas (M0) with Search index management. Keyword search is Atlas Search: a `search`-type index and `$search` with the `text` operator, created with `create_search_index(SearchIndexModel(type="search"))`. If it is unavailable (not Atlas, missing privilege, search-index cap), `load_mongodb` stops with the prerequisite, `semantic` is unchanged and `hybrid` is not usable; there is no regex, `$text` or in-memory fallback.
- **Field and index.** `chunks.text` (the clean chunk text; not `embedding_input_text`, not the section text). Index `chunks_text_search_index` on `chunks`, static mapping: `text` as `string` with `lucene.standard`; `act`, `chapter`, `status`, `access_level` as `token` for filters. Created or verified by `uv run python -m building_with_rag.ingestion.load_mongodb --skip-embed` (reuse a match; report a mismatch, never change it). Nothing was re-embedded, re-chunked or rewritten.
- **Flow.** Validate → readiness (vector index, embeddings and keyword index all queryable) → semantic route (existing embed + `$vectorSearch`, 20 candidates) → keyword route (one `$search`, `text` on `chunks.text`, the same server-built filter as `compound.filter`, 20 candidates) → fuse → resolve `chunk_id` to `chunks` and `sections` → `QueryResult.results`. Corpus, MongoDB, `RetrievedChunk`, context assembly, citations, confidence, `/v1/query`, chat streaming and `.env.example` are shared with semantic; no setting was added.
- **Fusion rule.** Reciprocal rank fusion in the application: `fused_score = Σ 1 / (60 + rank)` over the routes that returned the chunk, equal weights (`RRF_K`, `ROUTE_WEIGHTS`, `HYBRID_ROUTE_CANDIDATES` in `hybrid.py`). Rank-based because cosine similarity and the keyword score are on different scales. Ties: better single-route rank, then `chunk_id`. `$rankFusion` is not used, so the rule is visible and identical on every tier.
- **Result.** `RetrievedChunk` gains `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank` (null in semantic results); in hybrid `score` equals `fused_score`. `trace[0]` records the fusion rule, per-route hit counts, overlap, semantic-only and keyword-only counts, drops, and one `candidates` row per fused candidate (routes, ranks, scores, `returned`), including those cut by `limit`. No passage text or vectors.
- **Failures.** A missing or not-queryable vector or keyword index, a failed `$search`, or an embedding failure is HTTP 503, never `no_results` and never a silent semantic-only result.
- **Limitations.** The keyword route sees only `chunks.text`, so section numbers and headings are not keyword-searchable (a section's own number is usually absent from its body; cross-references are present). A natural-language question is matched term by term with no stop-word, phrase or synonym handling, so common words add noise. RRF discards score magnitude. The fused score is not confidence or correctness. `k` and pool size were chosen, not tuned. Hybrid costs one extra search call. No re-ranking (Story 4.2 adds it as a separate mode; `hybrid` itself is unchanged).
- **Diagnostic command:** `uv run python -m building_with_rag.retrieval "<question>" --mode hybrid` (`--check` also reports the keyword index).

## Re-ranking

Story 4.2 makes `hybrid-reranked` the third real mode. It is selected explicitly (`rag-hybrid-reranked` or `pattern: "hybrid-reranked"`); nothing routes between modes. Code: `src/building_with_rag/retrieval/rerank.py`; hybrid's candidate step is `hybrid_candidates` in `hybrid.py`, which `hybrid_search` also uses.

- **Flow.** Validate `RERANK_*` settings (before any retrieval) → hybrid candidates in fused order → top `RERANK_CANDIDATE_LIMIT` (20) → top `RERANK_SEND_LIMIT` (10) sent in one re-rank call → top `min(request.limit, RERANK_RETURN_LIMIT)` (5) in re-rank order → `QueryResult.results` → context assembly, answer, citations, confidence, `/v1/query` and chat unchanged. The generator receives only the final selected evidence.
- **Provider contract.** Voyage rerank: `POST {RERANK_API_BASE_URL}/rerank`, `Authorization: Bearer RERANK_API_KEY`, body `query`, `documents`, `model` (`RERANK_MODEL_NAME`), `top_k` (number sent); timeout `RERANK_REQUEST_TIMEOUT_SECONDS`; one attempt, no retry. The reply must have `data` with exactly one unique in-range integer `index` and a finite numeric `relevance_score` per document sent; anything else is a failure, never a partial result. `usage.total_tokens` is recorded in the trace when present. What leaves the app: the question and the chunk `text` of the sent candidates (public corpus). The key, headers and provider bodies are never logged or returned.
- **Result.** `RetrievedChunk` gains `rerank_score` (the provider's `relevance_score`) and `rerank_rank` (1-based among all sent); in this mode `score` equals `rerank_score` and the hybrid fields are kept. Ties: better `fused_rank`, then `chunk_id`. `QueryResult.omitted_candidates` holds every cut candidate with `rerank_candidate_limit`, `rerank_send_limit` or `rerank_return_limit`. `trace[0]` keeps the hybrid fields and adds `rerank` (model, limits, counts, usage), `candidates` in fused order (fused and re-rank rank/score, `sent`, `final_rank`, `outcome`) and `reranked_order`.
- **Outcomes.** Empty `RERANK_API_KEY` (the default) is HTTP 503 "RERANK_API_KEY is not set.", checked first. An empty base URL or model name, inconsistent limits (each at least 1; send at most candidate limit; return at most send limit), a timeout, connection error, non-2xx (status code only) or unusable reply are also 503 with short safe messages (chat: OpenAI error envelope before any stream). There is no silent fallback to hybrid or semantic order. No hybrid candidates is `no_results` without a re-rank call.
- **Limitations.** The re-rank score is relevance as judged by the provider, not correctness or confidence. Re-ranking only reorders or cuts what hybrid found, so a passage hybrid missed stays missed, and a good passage below the send limit is never seen. Only chunk text is sent (not heading or act label). One extra network call adds latency and provider cost and can fail or be rate limited. Limits were chosen, not tuned; the Voyage reply format is relied on. Explicit selection only; no evaluation.
- **Diagnostic command:** `uv run python -m building_with_rag.retrieval "<question>" --mode hybrid-reranked` (`--check` reports whether `RERANK_API_KEY` is set).

## Structured exact retrieval

Story 5.1 makes `structured` the fourth real mode: a read-only exact lookup of one named section. It is selected explicitly (`rag-structured` or `pattern: "structured"`); nothing routes to it. It is not LLM extraction and not a semantic-search replacement. Code: `src/building_with_rag/retrieval/structured.py`.

- **Exact-input contract.** A rule-based classifier (no LLM, no network) reads only `QueryRequest.question` and produces the existing `StructuredSignals`, with only the fixed intents `exact_lookup`, `filter`, `aggregation`. Phrasings recognised: an act (`BNS`, `IPC`, or the full act name) with a section number, such as "BNS section 103", "section 302 of the IPC", "IPC 304A", "Section 103, Bharatiya Nyaya Sanhita"; a number is 1 to 4 digits with at most one trailing letter. An optional chapter (a Roman numeral or number, normalised to the stored upper-case Roman form) comes from "chapter X" or `QueryRequest.chapter`; a conflict or invalid value is `clarification_needed`. Only a validated act and section number run a lookup.
- **Predicate.** One `find` on `sections` (cap 5), built by the server from validated signals and constants: `act` (`BNS`->`BNS_2023`, `IPC`->`IPC_1860`), integer `section_number`, `status` (`BNS`->`in_force`, `IPC`->`repealed`; this selects the act's stored status and is not a currency filter), `access_level` `public`, plus `chapter` if given. Request text never reaches MongoDB. `request.filters` can only narrow (outside scope is `not_found`). No writes, no other operations, no new index or setting.
- **Statuses** (HTTP 200): `ok` (one section), `not_found` (no stored match, or outside the filters), `clarification_needed` (number without an act, two acts, several numbers, chapter conflict, or a bare number shared by suffixed sections such as `ipc:304`, `ipc:304A`; the candidates are listed in the trace), `unsupported` (`filter` or `aggregation` intent, or no recognisable intent; the message recommends `rag-hybrid` or `rag-semantic`, which the user must choose). Classification failures make no MongoDB call. MongoDB errors or a missing `MONGODB_URI` on a lookup are 503.
- **Result.** `results` holds one `RetrievedChunk`: `chunk_id` `<section_id>#section`, the full stored section `text`, `score` null (no similarity), and the new optional `source_status_version`. `trace[0]` records intent, signals, reason, `mongodb_called`, the predicate and the match count.
- **Answer boundary.** The section is evidence in `QueryResult.results`, visible with no LLM call when `generate_answer` is false. Only when `generate_answer` is true and the status is `ok` does the existing context, grounded-answer, citation and streaming path run (one retrieval, at most one generation); other statuses never call the LLM and chat streams the message. Stored status and `source_status_version` are source metadata, not a claim that a provision is currently applicable.
- **Limitations.** Exact lookup only; a small fixed set of phrasings (others are `unsupported`); no routing, no natural-language search, no listing or counting; a section longer than the context character limit is dropped by the existing context rule.
- **Diagnostic command:** `uv run python -m building_with_rag.retrieval "BNS section 103" --mode structured`.
