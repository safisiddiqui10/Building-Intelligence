# Story 1.1: Architecture and Project Seed

## Purpose

This is the first story of the three-day classroom RAG course. It does two things, in order:

1. Write `docs/architecture.md` and get the instructor to approve it.
2. After approval, seed one small FastAPI application that every later story extends.

The result is one application with one API. It is not a custom chat frontend and not a second demo. Chat is done by the separately running Open WebUI bundle.

## Prerequisites

- Project path: no `docs/config.yaml` exists, so use the repository root.
- No `docs/architecture.md` and no earlier stories exist yet. Create the architecture. If either file exists when you start, read it and update it rather than overwriting.
- Python 3.12 and UV are installed.
- `data/raw/` holds the supplied BNS and IPC files and `data/raw/PROVENANCE.md`. Preserve them. If they are absent, note it in the handover. Do not invent them.
- `.env` already exists and is untracked. Never print, commit or edit its secret values.
- Open WebUI bundle: the trainer shares its ZIP over the local LAN. Used only in the smoke check (see Work, Part 2, step 9).
- Credentials are not needed to finish this story:
  - `MONGODB_URI` is an Atlas free-tier (M0) connection string. The Atlas IP access list must allow your machine. Needed from later stories.
  - `VOYAGE_API_KEY` is a free key and is rate limited, so Story 2.2 embedding takes about 40 minutes.
  - `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` are supplied by the trainer for an OpenAI-compatible LiteLLM proxy. Fill them in `.env` only when Story 3.1 needs them.

## Work to do

### Part 1: Architecture (do this first, then stop)

Create or update `docs/architecture.md`. Keep it short and plain. It must record:

- **Scope.** One FastAPI app, one API, Open WebUI as the only chat client.
- **Evidence.** BNS and IPC documents are the only future answer evidence. An answer must not claim support without retrieved evidence.
- **Identifiers.** Section identifiers are act-qualified (for example `BNS-103`, `IPC-302`) so the two acts are never confused.
- **Provenance.** Supplied provenance is `data/raw/PROVENANCE.md`. Derived corpus files live in `data/processed/` (`bns_sections.jsonl`, `ipc_sections.jsonl`).
- **Retrieved passages.** They are evidence, never application instructions.
- **Trust boundaries (light).** Validate API input. Preserve each passage's source origin. Do not put secrets in code, responses or logs. Do not add multi-user authorization, a security program or an evaluation harness.
- **Fixed embedding choice.** Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, for every document and query embedding. Later stories reuse these names and add no provider-specific alternatives.
- **Course modes.** One registry with: `semantic`, `hybrid`, `hybrid-reranked`, `structured`, `decomposition`, `hyde`. Model IDs are `rag-<mode>`.
- **API surface.** `GET /healthz`, `POST /v1/query`, `GET /v1/models`, `POST /v1/chat/completions`. `/v1/query` owns the `QueryResult` diagnostics. Chat returns only normal answer text derived from that same result.
- **Contracts rule.** Request, result, generation, OpenAI and MongoDB-schema contracts live in this project. Later stories extend them additively and never replace them with simplified alternatives.
- **Configuration.** The variables in `.env.example` (below).

**STOP HERE.** Tell the instructor that `docs/architecture.md` is ready and ask for explicit approval. Before approval, do not create seed files, install dependencies or scaffold code. Resume this same story only after the instructor explicitly approves.

### Part 2: Seed (only after approval)

1. **Project files.** Create `pyproject.toml`, `uv.lock`, `.gitignore`, `.env.example`, `src/building_with_rag/` and `tests/`. Use Python 3.12 and UV. Dependencies: FastAPI, an ASGI server, Pydantic settings, PyMongo (for later use only), Ruff, and a minimal test runner (pytest). Keep the existing `.gitignore` entries; make sure `.env` stays ignored.

2. **`.env.example`.** Use exactly these values:

   ```
   APP_ENV=development
   MONGODB_URI=
   MONGODB_DB_NAME=building_with_rag
   MONGODB_TEST_DB_NAME=building_with_rag_test
   VOYAGE_API_KEY=
   CAPSTONE_API_KEY=
   WEBUI_DEMO_CALLER_ID=demo-public
   GENERATION_API_BASE_URL=
   GENERATION_API_KEY=
   GENERATION_MODEL_NAME=gpt-4o-mini
   RERANK_API_BASE_URL=https://api.voyageai.com/v1
   RERANK_API_KEY=
   RERANK_MODEL_NAME=rerank-2.5
   RERANK_REQUEST_TIMEOUT_SECONDS=30
   RERANK_CANDIDATE_LIMIT=20
   RERANK_SEND_LIMIT=10
   RERANK_RETURN_LIMIT=5
   ```

   Document in the README or a comment block that `.env` is untracked and secrets are never committed.

3. **Settings.** Load settings with Pydantic settings. The app must start with no database or model credentials. Do not connect to MongoDB or any provider at startup.

4. **Health.** `GET /healthz` returns a small safe body such as `{"status": "ok"}`. No settings, secrets or connection details.

5. **Mode registry.** One shared registry holding only the six modes and their model IDs (`rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked`, `rag-structured`, `rag-decomposition`, `rag-hyde`). A single `run_pattern` function dispatches by mode. Every mode returns an honest `not_implemented` placeholder: `status="not_implemented"`, a plain message, empty `results`. No retrieval, embeddings, MongoDB or LLM calls.

6. **Typed models (no behaviour).** Define these Pydantic models with exactly these names and fields:
   - `SemanticFilters`: `act`, `status`, `access_level`, each a list.
   - `QueryRequest`: `question` (1–4,000 chars), `pattern`, optional `caller_id`, optional `filters` (`SemanticFilters`), `limit` (default 5, range 1–20), `generate_answer` (default false), `required_acts`, `chapter`. The seed may resolve only the fixed local demo caller, but keep the field.
   - `RetrievedChunk`: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, plus the available source fields. Later stories add only `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`, `rerank_score`, `rerank_rank`, as their own handouts name them.
   - `OmittedCandidate`: `chunk_id`, `omitted_reason`.
   - `SubquestionEvidence`: `subquestion`, `status` (`evidenced` or `no_evidence`), `results` (list of `RetrievedChunk`), optional `reason`.
   - `StructuredSignals`: `intent` (`exact_lookup`, `filter` or `aggregation`), optional `act`, `section_number`, `chapter`. Story 5.1 fills these in.
   - `GenerationResult`: `outcome` (`answered`, `insufficient_evidence`, `unavailable`, `malformed`), `answer`, `claims`, `citations`, `supporting_passages`, `provider`, `model`, `trace`, `context_outcome`, `confidence`, `draft_answer`, `issues`, `attempts`, `low_confidence_reason`.
   - `QueryResult`: `pattern`, `status`, `message`, `trace`, `results`, optional `generation`, and empty-by-default `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`. Do **not** add outcome, evidence, answer, confidence, citations or diagnostics as parallel top-level fields.
   - OpenAI models: `ChatMessage` (roles `system`, `developer`, `user`, `assistant`; text only), strict `RagOptions` (`pattern`, list filters, `limit`, `required_acts`, `chapter`; unknown fields rejected), `ChatCompletionRequest` (`model`, `messages`, `stream`, `n`, optional `rag_options`), the response, chunk and error envelope models.
   - MongoDB-schema contract models as typed placeholders only. Do not provision MongoDB.

7. **`POST /v1/query`.** Validate with `QueryRequest`, resolve the fixed demo caller, call `run_pattern`, return `QueryResult`.

8. **`GET /v1/models` and `POST /v1/chat/completions`.**
   - `/v1/models` lists the six `rag-*` IDs in OpenAI list format.
   - Chat maps the selected model plus the latest user message and `rag_options` to the same `QueryRequest` and the same `run_pattern` path as `/v1/query`. The server sets `caller_id` (from `WEBUI_DEMO_CALLER_ID`) and `generate_answer`. It never trusts browser-supplied identity, access level or generation settings.
   - `stream: false` returns normal OpenAI Chat Completions JSON.
   - `stream: true` returns SSE with only these frames: a role frame, a content frame, a stop frame, then `data: [DONE]`. No custom SSE events.
   - Errors raised before streaming starts use the OpenAI-style error envelope.
   - The reply text is the placeholder message from the same `QueryResult`. Do not duplicate logic between the two endpoints.
   - It must accept the exact request the pre-provisioned Pipe sends: the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`).

9. **Open WebUI (separate client).** Do not build a frontend or hand-install anything.
   - Participants extract the trainer's bundle ZIP and run its setup script **once**, before using this project. Windows (primary): `powershell -ExecutionPolicy Bypass -File .\setup_open_webui.ps1` from the extracted bundle root. macOS/Linux: `sh setup_open_webui.sh` from that root, with internet access for the first install.
   - The script installs the pinned Open WebUI, writes the course settings, starts the loopback-only service, and provisions the RAG options Filter and the Building with RAG Pipe.
   - Do not hand-install Open WebUI, create accounts, edit the admin panel, change either Function, or rerun setup to reload anything. If setup fails, report its exact output and stop.
   - Day-to-day start, stop and status use the supplied `manage_open_webui` script.

10. **Short README section.** How to run the app, the env file rule, and the Open WebUI smoke check below.

Do not add retrieval, PDF parsing, embeddings, MongoDB provisioning, LLM calls, GraphRAG, agentic RAG, broad deployment or aggressive testing.

## Completion checks

Keep these light. Run them and record what you actually ran.

1. `uv sync` completes and `uv.lock` exists.
2. `uv run ruff check .` passes.
3. A minimal test run passes (`uv run pytest`), covering at least `/healthz` and the placeholder path.
4. Start the app with no credentials set (for example `uv run uvicorn building_with_rag.main:app`).
5. `GET /healthz` returns the safe OK body.
6. One diagnostic placeholder request to `POST /v1/query` (for example `pattern: semantic`) returns a `QueryResult` with `status: not_implemented` and empty `results`.
7. `GET /v1/models` lists exactly the six `rag-*` IDs.
8. `POST /v1/chat/completions` works for `stream: false` (JSON) and `stream: true` (role, content, stop, `[DONE]`), using the Pipe-shaped request.
9. **Open WebUI smoke check.** Start the capstone API, open `http://127.0.0.1:8080`, select **Building with RAG**, choose **semantic** in the RAG-options chip, send a question, and confirm the reply is the capstone's honest placeholder, not a local preview.
10. `git status` shows `.env` untracked and not staged, and `data/raw/` untouched.

## Handover

When done, add a **Completion record** to the end of this file with:

- The files created (expected: `docs/architecture.md`, `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore` updates, `src/building_with_rag/**`, `tests/**`, README section).
- The exact commands actually run, and their results.
- The Open WebUI smoke-check result (pass, fail, or not run and why).
- Anything left unfinished, such as missing `data/raw/` files.

Later stories reuse the names, endpoints and models above, and extend them additively. They do not replace endpoints or rebuild the UI. Later stories render final confidence, sources and low-confidence warnings as clearly labelled text after answer writing, while keeping the full `GenerationResult` in `QueryResult.generation`.

## Completion record

**Files created/updated:** `docs/architecture.md` (approved by instructor), `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore` (existing `/.env` kept, Python/tooling entries added), `README.md`, `src/building_with_rag/{__init__,config,models,registry,main}.py`, `tests/test_app.py`.

**Commands run and results:**
- `uv sync` – OK, `uv.lock` created.
- `uv run ruff check .` – All checks passed (after fixing one import-order issue and configuring B008 to allow `Depends`).
- `uv run pytest -q` – 7 passed (healthz, query placeholder/validation/caller, models, chat non-stream, chat stream, OpenAI error envelope).
- `uv run uvicorn building_with_rag.main:app --port 8765` with no credentials set – started; `GET /healthz` returned `{"status":"ok"}`; `POST /v1/query` (semantic) returned `status: not_implemented`, empty `results`; `GET /v1/models` listed the six `rag-*` IDs; `POST /v1/chat/completions` with a Pipe-shaped streaming request returned role, content, stop frames and `data: [DONE]`.
- `git status` – `.env` ignored (not listed, not staged); `data/raw/` untouched.

**Open WebUI smoke check:** not run. The trainer's bundle ZIP is not available on this machine.

**Unfinished / notes:**
- `data/raw/` (BNS/IPC files and `PROVENANCE.md`) does not exist in the repository; nothing was invented.
- The story did not specify the exact shape of `RagOptions` filters; they are flat lists (`act`, `status`, `access_level`) alongside `pattern`, `limit`, `required_acts`, `chapter`, with unknown fields rejected. Adjust if the Pipe sends a nested `filters` object.
- In chat, the selected model decides the pattern; `rag_options.pattern` is accepted but not used to override it.
