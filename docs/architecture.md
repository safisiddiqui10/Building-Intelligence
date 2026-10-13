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
| `hybrid` | `rag-hybrid` |
| `hybrid-reranked` | `rag-hybrid-reranked` |
| `structured` | `rag-structured` |
| `decomposition` | `rag-decomposition` |
| `hyde` | `rag-hyde` |

## API surface

- `GET /healthz`: safe liveness body only.
- `POST /v1/query`: owns the `QueryResult` diagnostics.
- `GET /v1/models`: the six `rag-*` IDs in OpenAI list format.
- `POST /v1/chat/completions`: maps model, latest user message and `rag_options` to the same `QueryRequest` and `run_pattern` path as `/v1/query`. The server sets `caller_id` and `generate_answer`. Returns only normal answer text derived from that same `QueryResult`. Streaming emits role, content, stop frames, then `data: [DONE]`.

## Contracts rule

Request, result, generation, OpenAI and MongoDB-schema contracts live in this project. Later stories extend them additively and never replace them with simplified alternatives. Confidence, outcome, evidence and citations stay inside `QueryResult.generation`, not as parallel top-level fields.

## Configuration

Loaded by Pydantic settings from environment / `.env`. The app starts with no credentials and connects to nothing at startup. Variables (see `.env.example`):

`APP_ENV`, `MONGODB_URI`, `MONGODB_DB_NAME`, `MONGODB_TEST_DB_NAME`, `VOYAGE_API_KEY`, `CAPSTONE_API_KEY`, `WEBUI_DEMO_CALLER_ID`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME`, `RERANK_API_BASE_URL`, `RERANK_API_KEY`, `RERANK_MODEL_NAME`, `RERANK_REQUEST_TIMEOUT_SECONDS`, `RERANK_CANDIDATE_LIMIT`, `RERANK_SEND_LIMIT`, `RERANK_RETURN_LIMIT`.

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
