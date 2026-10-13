# Story 2.2: MongoDB Storage, Embeddings and Vector Index

## Purpose

Take the section JSONL from Story 2.1 and make it searchable evidence in one MongoDB Atlas database: four collections (`sources`, `sections`, `chunks`, `embeddings`) and one Atlas Vector Search index on `embeddings.vector`.

This story only prepares data. It adds no retrieval endpoint, no answer generation, no hybrid or keyword search, no re-ranking and no separate vector store. `POST /v1/query` and the other Story 1.1 routes and models stay exactly as they are; `semantic` keeps returning its `not_implemented` placeholder until the next story.

Later semantic retrieval will query `embeddings` with `$vectorSearch`, then resolve each returned `chunk_id` to its `chunks` and `sections` records to build a `RetrievedChunk`.

## Prerequisites

- Stories 1.1 and 2.1 are complete. `data/processed/bns_sections.jsonl` (358 records) and `data/processed/ipc_sections.jsonl` (533 records) exist. If not, stop and say so; do not re-parse the PDFs.
- No `docs/config.yaml` exists, so use the repository root.
- Read `docs/architecture.md`, Stories 1.1 and 2.1, the existing code (`src/building_with_rag/`) and the generated corpus before you start. Preserve all existing routes, models, settings and tests.
- `.env` is untracked and already filled in. Use only the existing `.env.example` settings, exactly as named: `MONGODB_URI`, `MONGODB_DB_NAME`, `MONGODB_TEST_DB_NAME`, `VOYAGE_API_KEY`. Do not rename, remove or add any variable and do not edit `.env.example`. Never print, log or commit secret values (including the URI's password).
- `MONGODB_URI` is a MongoDB Atlas free-tier (M0) connection string, and the Atlas IP access list allows your machine. `VOYAGE_API_KEY` is a free, rate-limited key.
- The embedding choice is fixed in `docs/architecture.md` ("Fixed embedding choice") and already in `Settings`: Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions. Reuse `embedding_model`, `embedding_model_version` and `embedding_dimensions` from settings. Do not guess or offer alternatives.
- If MongoDB is unreachable or `MONGODB_URI` is empty, stop with a clear prerequisite message (what is missing, which variable, no secret values). If `VOYAGE_API_KEY` is empty, you may still create collections, chunks and the index, but stop before embedding with a clear message and report verification as pending.

## Work to do

### 1. Add dependencies (only what is needed)

Add the Voyage client with `uv add voyageai`. `pymongo` is already a dependency. Add nothing else. This updates `pyproject.toml` and `uv.lock`.

### 2. Project layout

The Story 2.1 module `src/building_with_rag/ingestion.py` becomes a package `src/building_with_rag/ingestion/` so that the requested `ingestion/mongodb_schema.py` exists. Move the existing code into the package (for example `ingestion/parser.py`) and keep `uv run python -m building_with_rag.ingestion` working unchanged through `ingestion/__init__.py` and `ingestion/__main__.py`. Do not change its behaviour or output. Add the new code beside it:

- `ingestion/mongodb_schema.py`: schema, constants, IDs, validators, ordinary indexes (step 4).
- `ingestion/chunking.py`: chunker (step 5).
- `ingestion/embedding.py`: Voyage batching, delay and retry (step 7).
- `ingestion/load_mongodb.py`: the command (step 8).

Do not import any of these from the app's startup path. The app still starts with no credentials and connects to nothing at startup.

### 3. Identify the deployment first

Before creating anything, connect with `MONGODB_URI` and a short server-selection timeout, then record what you found:

- Reachable? If not, stop (see Prerequisites).
- Is it Atlas? Atlas Search and Vector Search index management (`createSearchIndexes`, `listSearchIndexes`) are Atlas features. Verify by calling `list_search_indexes()` on a collection. A non-Atlas deployment (or a missing privilege) fails with a command-not-found or authorization error. Report it plainly and stop. Do not fall back to a generic index or another vector store.
- Record in `docs/architecture.md` the exact capability found (Atlas M0, Atlas Search index management and `$vectorSearch` available) and use the documented PyMongo API: `collection.create_search_index(SearchIndexModel(..., type="vectorSearch"))`, `list_search_indexes()`.
- Use `MONGODB_DB_NAME` for real data. `MONGODB_TEST_DB_NAME` is for tests and dry runs only; never write corpus data to it unless a command flag explicitly selects it.

### 4. Schema module: `ingestion/mongodb_schema.py`

This one module owns everything below. Every other file imports from it; nothing hard-codes a collection name, ID format or field name elsewhere.

**Collection constants:** `SOURCES = "sources"`, `SECTIONS = "sections"`, `CHUNKS = "chunks"`, `EMBEDDINGS = "embeddings"`, plus `VECTOR_INDEX_NAME = "embeddings_vector_index"` and `VECTOR_FIELD = "vector"`. Four separate collections. Do not merge them and do not store vectors on chunks. (The Story 1.1 `ChunkDocument` placeholder that carries `embedding` is a typed placeholder only; leave it untouched and do not use it for storage. Note the difference in `docs/architecture.md`.)

**Document model** (field names are the contract; use them exactly):

| Collection | `_id` | Fields |
|---|---|---|
| `sources` | `source_id` | `source_id`, `source_pdf`, `source_sha256`, `act`, `act_label`, `status`, `source_status_version`, `parser`, `parser_version`, `created_at` |
| `sections` | `section_id` | `section_id` (`bns:103`), `source_id`, `act`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `section_label` (optional), `heading`, `text`, `text_sha256`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version`, `needs_review`, `page_start`, `page_end`, `access_level` |
| `chunks` | `chunk_id` | `chunk_id`, `section_id`, `chunk_index`, `chunking_version`, `splitter`, `text`, `embedding_input_text`, `act`, `chapter`, `status`, `access_level` |
| `embeddings` | `embedding_id` | `embedding_id`, `chunk_id`, `embedding_model`, `embedding_model_version`, `dimensions`, `vector`, `act`, `chapter`, `status`, `access_level` |

Rules:

- `access_level` is `"public"` for the whole classroom corpus, set on `sections`, `chunks` and `embeddings`. Keep the field so the shape stays compatible with later access rules. Add no access-control logic.
- `act`, `chapter`, `status` on `chunks` and `embeddings` are copied from the section, and `access_level` too. Section number and source identifiers (`source_pdf`, `source_sha256`) are **not** denormalized onto `embeddings`; they live on `sections`/`sources` and are applied after `chunk_id` resolution.
- `embedding_input_text` is what gets sent to Voyage: the act label, section heading and chunk text (for example `"{act_label} — {heading}\n\n{text}"`). `text` stays the clean chunk text for evidence display. Record the template in `docs/architecture.md`.
- Keep Story 2.1's `section_id` rule (`bns:<number>`, `ipc:<number>`, suffix kept: `ipc:304A`).

**Deterministic IDs** (pure functions, same input gives the same ID on every run):

- `source_id`: from `source_pdf` and `source_sha256` (for example `bns_2023:<first 12 hex of sha256>`).
- `chunk_id`: `f"{section_id}#{chunking_version}#{chunk_index}"` style, readable and act-qualified, so `bns:103` and `ipc:103` never collide.
- `embedding_id`: from `chunk_id`, model, model version, dimensions and a hash of `embedding_input_text` (for example `"{chunk_id}|{model}|{dims}|{input_hash12}"`). Changed text or model settings therefore produce a different ID and are never mixed silently.

**Validators:** one `$jsonSchema` validator per collection (required fields and types only, `validationLevel: "moderate"`, `validationAction: "error"`). The `embeddings.vector` validator checks "array of doubles". Do not build a broad validation framework.

**Ordinary indexes** (function `ensure_indexes(db)`, idempotent): `sections` on `act`+`section_number` and on `source_id`; `chunks` on `section_id`+`chunk_index` (unique); `embeddings` on `chunk_id`; `sources` on `source_sha256`. These are normal B-tree indexes, separate from the vector index.

**`ensure_collections(db)`:** create each collection with its validator if absent, otherwise apply the validator with `collMod`. Never drop a collection.

### 5. Chunking

- Chunk each section's `text` only (as Story 2.1 recorded). A chunk never crosses a section or an act.
- Target about 512 tokens, which this story fixes as about 2,048 characters, with about 256 characters of overlap. Use one simple character-based recursive splitter written in the project (split on paragraph breaks, then lines, then sentence ends, then spaces; hard-cut only if nothing else works). Do not add a splitter dependency.
- A section no longer than the target is exactly one chunk (`chunk_index` 0). Roughly 52 of the 891 sections are longer than 2,048 characters, so expect around 1,000 chunks in total; record the actual count.
- Set `chunking_version` (for example `"char-2048-256-v1"`) and `splitter` (for example `"recursive-char"`) on every chunk. Changing size, overlap or splitter means a new `chunking_version`.

### 6. Load sources, sections and chunks (idempotent)

For each act file present: build one `sources` record (from the first record's source fields), upsert each section by `section_id` with a `text_sha256`, and upsert chunks by `chunk_id`.

- Unchanged data is reused: a section or chunk whose content hash matches is left as it is (no needless writes).
- If a section's text changed, or the chunking version changed, replace that section's `chunks` and delete the now-stale `chunks` and `embeddings` for that `section_id`, and print a clear line saying what was replaced and why. Do this per section, in a way that cannot leave old and new chunks mixed. If the `source_sha256` of an act changed, report it as a changed source and refresh that act's `sources` and `sections`.
- Embeddings with a different model, version or dimensions than the current fixed choice are not deleted automatically: report their count and leave them. The vector index and later queries filter on the fixed model by construction; state this in the architecture note.
- Re-running must produce the same document counts.

### 7. Embeddings (small batches, resumable)

- Use the Voyage client with `VOYAGE_API_KEY` (document embeddings use `input_type="document"`, model `voyage-3.5`, `output_dimension=1024`). Do not add another provider or key.
- Query first for chunks that already have an `embeddings` record with the current `embedding_id`; embed only the rest. This makes an interrupted run resumable and a repeat run a no-op.
- Embed in small batches (about 8 chunks, well under the free-tier token budget) with a delay between batches (default about 20 seconds; make both numbers named constants). On a rate-limit error (HTTP 429 or the client's rate-limit exception), wait and retry with growing back-off and a bounded retry count; on repeated failure, stop cleanly, say how many remain and that re-running resumes.
- Insert each batch's embeddings as soon as they return, using `embedding_id` as `_id` so a repeated insert cannot duplicate. Check the returned vector length equals 1,024 before storing; if not, stop and report.
- Print progress after each batch (for example `embedded 64/994 chunks, 930 remaining`). Never print keys.
- **Expected duration:** about 1,000 chunks at roughly 8 per batch and a 20-second delay is about 125 batches, so **about 40 minutes** on the free key for the first run. A repeat run with nothing to do finishes in seconds. Say this at the start of the command's output.

### 8. Command

Provide `uv run python -m building_with_rag.ingestion.load_mongodb` (the command), which does in order: check settings and MongoDB (stop clearly if unavailable) → identify deployment capability → ensure collections, validators and ordinary indexes → load sources, sections and chunks → create or verify the vector index (step 9) → embed in batches (step 7) → wait for index readiness → run the sample query (step 10). Keep one optional flag, `--use-test-db`, to target `MONGODB_TEST_DB_NAME`. No other options are needed. A short `--skip-embed` is allowed if it keeps the code simpler for the instructor demo; otherwise leave it out.

### 9. Vector index (Atlas `vectorSearch` type)

Create the index on `embeddings` named `VECTOR_INDEX_NAME` with `SearchIndexModel(type="vectorSearch")` and a **fields-array** definition (not the legacy `knnVector`/`mappings` syntax, which belongs to `search`-type indexes). Field names come from the schema constants:

```python
{
  "fields": [
    {"type": "vector", "path": "vector", "numDimensions": 1024, "similarity": "cosine"},
    {"type": "filter", "path": "act"},
    {"type": "filter", "path": "chapter"},
    {"type": "filter", "path": "status"},
    {"type": "filter", "path": "access_level"},
  ]
}
```

- Safe to repeat: call `list_search_indexes(name)`. If an index with this name exists and its definition matches, reuse it. If it exists with a different definition (dimensions, similarity, path or filters), report exactly what differs and stop. Never delete or update it automatically.
- Wait a bounded time for readiness (for example poll every 5 seconds for up to 5 minutes until `queryable` is true). If it is not ready in time, report "index not ready yet; re-run to continue" rather than hanging or failing silently.
- If the deployment is not Atlas or the user lacks the privilege, stop with that prerequisite message (step 3).

### 10. One sample vector query

When MongoDB, the key and a ready index are all available, embed one fixed sample question (for example "What is the punishment for murder?") with `input_type="query"` using the same model and run one `$vectorSearch` on `embeddings` (`index` = `VECTOR_INDEX_NAME`, `path` = `vector`, a small `numCandidates`, `limit` 3, and a `filter` on `access_level: "public"`). Resolve the top hit's `chunk_id` to `chunks` and `sections` and print only: `chunk_id`, `section_id`, `heading`, `score`. This is a script check, not a retrieval endpoint: do not add it to the API or return a `RetrievedChunk`. If any service or credential is missing, print `verification pending` with the reason.

### 11. Update `docs/architecture.md`

Extend the existing document (do not rewrite it, and do not add or rename settings). Add a short "MongoDB storage and vector index" section recording:

- The four collections and their fields, and why vectors live in `embeddings`, not `chunks`.
- The deployment capability found (Atlas M0, `$vectorSearch`, `create_search_index`) and the stop-with-message rule otherwise.
- The chunking choice (about 2,048 characters, about 256 overlap, splitter name, `chunking_version`) and the `embedding_input_text` template.
- The embedding decision: Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, cosine, batch size and delay.
- The index name and definition (vector path plus the four filter fields), and that section number and source identifiers are applied after `chunk_id` resolution.
- Commands: `uv run python -m building_with_rag.ingestion` (unchanged) and `uv run python -m building_with_rag.ingestion.load_mongodb`.
- Limitations: only classroom `public` data; rate-limited embedding takes about 40 minutes; stale embeddings from other models are reported, not deleted; no retrieval yet; the M0 cluster allows only a few search indexes.

Add one line to `README.md` for the new command. Do not change `.env.example`.

## Completion checks

Keep these classroom-sized. Run them and record what you actually did.

1. `uv sync` completes, `uv run ruff check .` passes, and the existing `uv run pytest` still passes (the seven Story 1.1 tests). Add only a few small unit tests with no network: deterministic IDs repeat, the chunker never crosses sections and respects size and overlap, and the index definition matches the fields array above.
2. `uv run python -m building_with_rag.ingestion` still works and reports both files current.
3. Run the load command. Report the deployment finding, collection counts (`sources` 2, `sections` 891, `chunks` about 1,000 and `embeddings` equal to `chunks`).
4. **Section-to-chunk link:** pick one stored section (for example `bns:103`) and show that its chunk(s) carry its `section_id`, `chunk_index` starting at 0, `chunking_version` and the same `act`, `chapter`, `status`.
5. **Embedding dimension:** read one stored embedding and confirm `len(vector) == 1024` and model/version match the fixed choice.
6. **Repeat setup:** run the command again. Confirm unchanged counts, no new embeddings, "index reused" and no duplicate records.
7. **Index readiness:** show the index named `embeddings_vector_index` is `queryable` and is of type `vectorSearch` with the fields-array definition.
8. **Sample result:** report the sample query's top result `chunk_id`, `section_id`, `heading` and `score`. If MongoDB, the Voyage key or the index is unavailable, report this as pending with the reason; do not fake it.
9. `git status` shows `.env` untracked and unchanged, `.env.example` unchanged, `data/raw/` untouched, and no secret in any output or file.

## Handover

When done, add a **Completion record** to the end of this file with:

- Deployment capability found (Atlas M0, search-index API used) or the exact prerequisite that stopped the run.
- Files created or changed, and the exact commands run with their results (collection counts, embedding run time and any rate-limit retries).
- Chunking numbers: chunk count, how many sections produced more than one chunk, the largest chunk.
- The completion-check results above, including the sample top result, or "verification pending" with the reason.
- Known limitations or anything left unfinished.
- **For the next story (semantic retrieval):** query `embeddings` with `$vectorSearch` on `embeddings_vector_index`, filter only on `act`, `chapter`, `status`, `access_level`, resolve `chunk_id` to `chunks` and then `sections`, and apply section number and source fields from those records when building a `RetrievedChunk`. Reuse the constants, IDs and validators in `ingestion/mongodb_schema.py`.

Later stories extend these collections and routes additively. They do not merge the collections, move vectors onto chunks, or replace the Story 1.1 contracts.

## Completion record

**Status: code complete; live verification pending.** The load command stopped at its first prerequisite: `MONGODB_URI` is empty in `.env` (and `.env` has a duplicate, empty `VOYAGE_API_KEY=` line after a filled one, which the settings loader may read as empty). Nothing was written to MongoDB. Deployment capability (Atlas M0, search-index API) is therefore not yet confirmed.

**Files created or changed:** `src/building_with_rag/ingestion.py` moved to `ingestion/parser.py` (only `ROOT` path depth changed); new `ingestion/{__init__,__main__,mongodb_schema,chunking,embedding,load_mongodb}.py`; `tests/test_ingestion_mongo.py`; `pyproject.toml` and `uv.lock` (`voyageai`); `docs/architecture.md`; `README.md`. `.env.example` unchanged.

**Commands run:**
- `uv add voyageai`: OK.
- `uv run ruff check .`: passed. `uv run pytest -q`: 10 passed (7 existing and 3 new offline tests: IDs, chunker, index definition).
- `uv run python -m building_with_rag.ingestion`: both files "already current; left unchanged".
- `uv run python -m building_with_rag.ingestion.load_mongodb --skip-embed`: "Prerequisite missing: MONGODB_URI is empty."

**Pending (need MONGODB_URI and VOYAGE_API_KEY in `.env`):** load counts, section-to-chunk link, embedding dimension, repeat run, index readiness, sample query result (chunk_id, section_id, heading, score), rate-limit retries and chunk statistics. Run `uv run python -m building_with_rag.ingestion.load_mongodb`, then re-check items 3 to 8 above.

**For the next story (semantic retrieval):** query `embeddings` with `$vectorSearch` on `embeddings_vector_index`, filter only on `act`, `chapter`, `status`, `access_level`, resolve `chunk_id` to `chunks` then `sections`, and apply section number and source fields from those records when building a `RetrievedChunk`. Reuse `ingestion/mongodb_schema.py`.
