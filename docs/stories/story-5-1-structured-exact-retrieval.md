# Story 5.1: Structured Exact Retrieval

## Purpose

Make `structured` the fourth real course mode. Semantic, hybrid and re-ranked retrieval find passages by meaning or keywords. They are the wrong tool when a user names a section outright ("BNS section 103"): the answer is one specific record, and a similarity search may rank it anywhere. `structured` looks that record up directly in MongoDB by act and section number.

It is a **read-only exact-lookup path**. It is not an LLM extraction step, not a replacement for semantic search, and not an automatic router. It is selected explicitly (`rag-structured` or `pattern: "structured"`), and a question it cannot turn into a supported exact request gets a plain recommendation or clarification, never a guess and never a MongoDB call.

It uses the same `QueryRequest` / `QueryResult` contract, the same `RetrievedChunk` evidence shape, the existing MongoDB collections and the existing `.env.example` settings. No setting is added or renamed. `semantic`, `hybrid` and `hybrid-reranked` keep working exactly as they do today.

Out of scope: arbitrary or caller-supplied database queries, any write, automatic routing between modes, LLM-based extraction or classification, filter and aggregation *execution* (they are recognised and answered with a recommendation only), a permissions system, new endpoints, new UI screens, re-embedding, re-chunking, and any change to the Pipe or Filter.

A returned section is the stored text of that section in the supplied BNS or IPC document, with its stored status and source version. It is not a statement that the provision is currently applicable law.

## Prerequisites

- Stories 1.1, 2.1, 2.2, 2.3, 3.1, 3.2, 4.1 and 4.2 are complete. No `docs/config.yaml` exists, so use the repository root.
- Read before you start: `docs/architecture.md`; Stories 1.1 to 4.2 including their completion records; and the code: `src/building_with_rag/{config,models,registry,main}.py`, `ingestion/mongodb_schema.py`, `retrieval/{semantic,hybrid,rerank,__main__}.py`, `generation/{context,answer,render}.py`, `tests/`, `docs/manual-tests.md`. Preserve all existing routes, models, settings, tests, the shared registry and the uncommitted work in the tree.
- Use only the existing `.env.example` names. This mode needs `MONGODB_URI` and `MONGODB_DB_NAME`; an explanatory answer also needs `GENERATION_API_BASE_URL` and `GENERATION_API_KEY`. It needs no `VOYAGE_API_KEY` and no `RERANK_*`. Add, rename or remove no variable, edit no `.env.example`, add no dependency. Fixed values go in named code constants. Never print, log or return a secret.
- Do not change the corpus, `sections`, `chunks`, `embeddings`, any index, the embedding model or chunking. Do not run the load. The existing `sections` index on `(act, section_number)` is enough; create no index.
- Stored facts this story relies on (verify against `mongodb_schema.py` and the loaded data before coding): `sections.act` is `BNS_2023` or `IPC_1860`; `section_number` is an **integer**; `section_id` is `bns:<n>` / `ipc:<n>` with a printed suffix kept for IPC (`ipc:304A`, number 304); `status` is `in_force` (BNS) or `repealed` (IPC); `access_level` is `public`; `source_status_version` may be `null`.
- Live checks need loaded MongoDB data (Story 2.2). If `MONGODB_URI` is missing, stop, say so and mark live verification pending. Do not fake results.

## Work to do

### 1. Rule-based classifier (no LLM, no network)

New module `retrieval/structured.py`. A pure function turns `QueryRequest.question` into the existing `StructuredSignals` (Story 1.1: `intent`, `act`, `section_number`, `chapter`; add no field). It uses fixed patterns only: no model call, no database access, no state.

- **Intents are exactly** `exact_lookup`, `filter`, `aggregation`. Nothing else is invented.
  - `exact_lookup`: one act and one section number, from phrases such as "BNS section 103", "section 302 of the IPC", "IPC 302", "Section 103, Bharatiya Nyaya Sanhita". Full act names map to the same two acts.
  - `filter`: a request to list or select by attribute, such as "list the sections in chapter VI of the BNS".
  - `aggregation`: a request to count or total, such as "how many sections are in the IPC?".
  - Anything else has no intent (see below).
- **Extraction and validation.** `act` is `BNS` or `IPC` only. `section_number` is the digits (1 to 4) with at most one trailing letter, kept as a string in the signals (`"103"`, `"304A"`). `chapter` is optional and is accepted only as a Roman numeral or number matching the stored chapter format (check the stored values; normalise case). A value that fails validation is not extracted.
- **Not exact and not guessed.** When the question has no supported intent, or `exact_lookup` cannot be established, the classifier returns an outcome (a small internal dataclass, not a new API model) with a status and a plain reason; it never fills a missing act or number. Rules, each with a fixed reason code:
  - number but no act ("What does section 103 say?") → `clarification_needed`, ask which act (BNS or IPC);
  - both acts, or more than one section number → `clarification_needed`, ask for one act and one section;
  - `filter` or `aggregation` intent → `unsupported`, saying this mode only does exact lookups and recommending `rag-hybrid` or `rag-semantic` for that kind of question (a recommendation to the user, never an automatic switch);
  - no recognisable intent ("What is the punishment for murder?") → `unsupported` with the same recommendation.
- Question text is data. Classification never executes, expands or forwards it.

### 2. Exact lookup (read-only, server-built predicate)

New `structured_search(request, settings) -> QueryResult` in `retrieval/structured.py`, reusing `get_db`, `RetrievalUnavailable`, `to_retrieved`-style mapping and `ACCESS_LEVEL` from existing code. One `find` on `sections`, nothing else.

- Run the classifier first. If it does not yield a supported `exact_lookup`, return its status and reason immediately: **no MongoDB connection, no query**, even if `MONGODB_URI` is unset.
- **The predicate is built only from validated signals and fixed constants**, never from request text: `act` from a fixed map (`BNS` → `BNS_2023`, `IPC` → `IPC_1860`), `section_number` as the validated integer, `status` from a fixed map per act (`BNS` → `in_force`, `IPC` → `repealed`), `access_level` = `public`, plus `chapter` only if validated. `request.filters` can only narrow (an act, status or access level outside the predicate's scope yields `not_found`); it never widens. The IPC is repealed, so the status map is what lets IPC lookups succeed, not a currency filter. `request.chapter` and a classifier chapter that conflict is `clarification_needed`.
- Read only the needed fields; cap the fetch at a small constant (for example 5). No other operation, no pipeline, no caller-supplied field names or operators.
- **Matches.** Exactly one section → `ok`. A bare number that matches several stored sections (for example `304` also matches `304A`, `304B`) with no suffix given → `clarification_needed` listing the candidate `section_id`s and no results; if a suffix was given, select the section whose label matches. Zero → `not_found`. Do not widen, fuzzy-match or fall back to another act or to semantic search.
- MongoDB errors, a missing `MONGODB_URI` on a request that needs the database, or an unreachable server raise `RetrievalUnavailable` (HTTP 503, chat error envelope before any stream), as in the other modes. A missing record is `not_found` (HTTP 200), never a 503.

### 3. Result contract (additive only)

- `QueryResult.pattern` is `"structured"`. `status` is one of `ok`, `not_found`, `clarification_needed`, `unsupported`. `message` is one plain sentence: no answer text, no claim of legal currency. `results` holds the section as one `RetrievedChunk` for `ok`, and is empty otherwise.
- The `RetrievedChunk` for the section: `chunk_id` = `<section_id>#section` (named constant; clearly distinct from real chunk ids, which end in `#<chunking_version>#<index>`), `section_id`, `act` (stored value), `text` (the full stored section text, unaltered), `heading`, `source` (act label), `source_file`, `page` (`page_start`), `chapter`, `status`, `access_level`. `score` is `None` (there is no similarity; do not invent one).
- **Source status and version.** Add one optional field to `RetrievedChunk`, `source_status_version: str | None = None`, filled from the section record (null elsewhere). It is the only model change. Show it in the trace too. Nothing may describe a section as "current", "valid" or "applicable".
- `trace[0]` is a plain dict (no passage text, no secrets): `step`, `mode`, `intent`, the validated `signals`, `classifier_reason`, `mongodb_called` (bool), the predicate as applied (the server-built one), `matched`, `candidate_section_ids` when ambiguous, `returned`, and `ignored_fields` as the other modes do. From `/v1/query` alone a reader can see how the question was read and what was queried.

### 4. Registry, `/v1/query` and chat (no new path)

- `registry.start_pattern` dispatches `structured` to `structured_search` through the existing retrieval branch, and exports it from `retrieval/__init__.py`. `/v1/query`, `/v1/models` (still the same six IDs) and `/v1/chat/completions` are otherwise unchanged; `main.py` and `generation/` are not edited.
- **Answer boundary.** Direct inspection (`generate_answer` false, the default) returns the section in `results` and the trace with no LLM call. Only when `generate_answer` is true **and** `status` is `ok` does `start_pattern` return the existing unstarted `generate_events` over `result.results`, so one request is one retrieval and at most one generation operation, with the existing context assembly, citation and support checks, confidence, DRAFT rendering and Sources. For `not_found`, `clarification_needed` and `unsupported` there is no generation call: `run_pattern` returns the result, and chat (which always sets `generate_answer`) streams `result.message` through the same path the placeholder uses.
- Context assembly's passage and character limits still apply; a section longer than `MAX_CONTEXT_CHARS` is dropped by the existing rule and reported in the generation trace. Do not change them.
- `decomposition` and `hyde` stay placeholders. Tests and manual-test rows that use `structured` as the placeholder example (`tests/test_app.py`, `tests/test_chat_streaming.py`, `docs/manual-tests.md` Story 1.1 and 3.2 placeholders) switch to `decomposition`; say so in the completion record.

### 5. Diagnostic command

Extend `uv run python -m building_with_rag.retrieval` with `--mode structured` (default stays `semantic`). It prints the intent, signals, status, reason, `mongodb_called` and, for `ok`, one line with `section_id`, heading, status and `source_status_version`. Compact; no passage text beyond the existing `--inspect` excerpt.

### 6. Update `docs/architecture.md`

Extend; do not rewrite earlier sections. Update the **Course modes** table (`structured` real since Story 5.1) and the sentence in the Streamed-answer section that lists which modes are real. In **Configuration**, note no variable was added. Add a short "Structured exact retrieval" section recording:

- **Exact-input contract:** the classifier is rule-based and sees only `question`; only the fixed `exact_lookup`, `filter`, `aggregation` intents exist; the lookup runs only for a validated act (`BNS`/`IPC`) and section number, plus optional validated chapter; the MongoDB predicate is built by the server from those signals and constants, never from request text; read-only, one `find` on `sections`, public access only; the statuses `ok`, `not_found`, `clarification_needed`, `unsupported` and when each occurs; ambiguity is clarified, never guessed.
- **Answer boundary:** the section is evidence in `QueryResult.results`; an explanatory answer happens only through the existing grounded path, only for `ok`, only when `generate_answer` is true; other statuses never call the LLM. The source status and version are metadata, not a legal-currency claim.
- **Limitations:** exact lookup only; the rule-based classifier recognises a small set of phrasings and will answer `unsupported` for others; no routing; no natural-language search; chapter filter is a narrowing, not a listing.
- The command `uv run python -m building_with_rag.retrieval "BNS section 103" --mode structured`.

Add one line to `README.md` for the command. Add a short Story 5.1 section to `docs/manual-tests.md` in the style of Story 4.2: the exact lookup, ambiguous, missing and unsupported commands, an answer command, the chat command, and the expected results.

### 7. Offline tests (few, targeted)

No network, MongoDB or LLM. Fake the collection; assert on the predicate passed to it.

- Classifier: "BNS section 103", "section 302 of the IPC" and "IPC 304A" yield the right signals; "What does section 103 say?", a two-act question and a two-number question yield `clarification_needed`; "how many sections are in the IPC?" and "What is the punishment for murder?" yield `unsupported`; text that looks like a query operator (for example `{"$ne": null}` in the question) yields no signals.
- Lookup: a valid request builds exactly the expected predicate (mapped act, integer number, mapped status, `public`) with no raw text in it; one match returns one compatible `RetrievedChunk` with `source_status_version`, `score` null; zero matches is `not_found`; an ambiguous bare number is `clarification_needed` with candidate ids; a `clarification_needed`/`unsupported` request never touches the (fake) database, even with `MONGODB_URI` unset; a MongoDB error raises `RetrievalUnavailable`.
- Wiring: `/v1/query` with `pattern: "structured"` returns the result and trace; with `generate_answer`, the generator's provider call receives only the one section and cites it, and no provider call is made for `not_found`/`clarification_needed`/`unsupported`; chat for `rag-structured` streams the answer for `ok` and the message otherwise; chat for `decomposition` still streams the placeholder.

## Verification rules

Offline checks may be run by the assistant: `uv run ruff check src tests`, `uv run pytest -q`, `git status`. Live checks call MongoDB (and the generation provider for answers); run them only if those credentials are present, print compact output only and never show a secret. Otherwise record them as **pending participant**.

## Completion checks

1. `uv run ruff check src tests` and `uv run pytest -q` pass, including every existing test (updated only as step 4 requires). Record the pass count.
2. **Valid exact lookup (live):** `retrieval "BNS section 103" --mode structured` (and one IPC, for example `IPC section 302`) returns `ok`, one section with the right `section_id`, status and `source_status_version`, `mongodb_called: true`, and the predicate in the trace.
3. **Ambiguous (live or offline):** a section number with no act ("What does section 103 say?") returns `clarification_needed` with a reason, no results and `mongodb_called: false`. If the loaded data has suffixed sections, `IPC section 304` also returns `clarification_needed` listing the candidates.
4. **Missing record (live):** a well-formed request for a section not stored (for example `BNS section 999`) returns `not_found`, HTTP 200, no results, not a 503 and not a semantic fallback.
5. **Unsupported:** "What is the punishment for murder?" returns `unsupported` with a recommendation to choose `rag-hybrid` or `rag-semantic`, and makes no MongoDB call.
6. **Answer boundary (live, needs generation settings, or pending participant):** `generate_answer: true` on the valid lookup returns a grounded answer citing that section; the same flag on a missing record returns no generation call. Chat `rag-structured` streams as `hybrid` does for `ok`, and shows the message for the other statuses.
7. `GET /v1/models` still lists the six IDs. By code and test inspection: one retrieval and at most one generation per request through `start_pattern` / `run_pattern`; the predicate comes only from validated signals; no write operation; no new setting, index or dependency; no automatic routing; `decomposition` and `hyde` still placeholders; `generation/` and `main.py` unchanged.
8. `git status`: `.env` untracked and unchanged; `.env.example`, `data/raw/`, `data/processed/`, the Pipe and the Filter unchanged; no corpus, index or secret change.

## Handover

When done, add a **Completion record** to the end of this file with:

- Files created or changed, and the exact commands run with results (counts and statuses only).
- The classifier phrasings supported and the reason codes; the predicate as built; how suffixed sections are handled; the `chunk_id` convention for section records.
- Tests updated for the placeholder change, and how many tests were added.
- The live results for checks 2 to 6 that were run (section ids and statuses only); any not run listed as **pending participant** with no results written.
- Anything unfinished, with the reason.
- Known limitations: exact lookup only; small fixed phrase set; explicit selection only; stored status and version are source metadata, not legal currency; one demo caller, `public` only.

**For the next story:** `decomposition` and `hyde` remain placeholders. Any later mode reuses `start_pattern`, `RetrievedChunk` and the unchanged generation path.

## Completion record

**Files created:** `retrieval/structured.py`, `tests/test_structured.py`. **Changed:** `models.py` (`RetrievedChunk.source_status_version`), `registry.py` (`structured` in the shared retrieval branch; generation only for `ok`), `retrieval/{__init__,__main__}.py` (`structured_search` export, `--mode structured`), `tests/{test_app,test_chat_streaming}.py`, `docs/{architecture,manual-tests}.md`, `README.md`. `main.py`, `generation/`, `.env.example`, `pyproject.toml`, data, indexes, the Pipe and the Filter are unchanged.

**Classifier:** `BNS`, `IPC` or the full act name with `section N`, `N of the ACT`, `ACT N` or `ACT-N` (1 to 4 digits, optional letter suffix); chapter from "chapter X". Reason codes: `exact_lookup`, `missing_act`, `multiple_acts`, `multiple_sections`, `unsupported_filter`, `unsupported_aggregation`, `no_supported_intent`, `invalid_chapter`, `chapter_conflict`, `ambiguous_section_number`, `excluded_by_filters`. **Predicate:** `{act: BNS_2023|IPC_1860, section_number: int, status: in_force|repealed, access_level: public[, chapter]}`. **Suffixes:** a bare number matching several stored sections is `clarification_needed` with the candidate ids; an explicit suffix (`304A`) selects the section whose `section_id` matches. **Section `chunk_id`:** `<section_id>#section`.

**Tests updated:** placeholder tests and manual-test rows that used `structured` now use `decomposition`. Added 19 offline tests in `tests/test_structured.py` (parametrised).

**Commands and results:** `uv run ruff check src tests`: passed. `uv run pytest -q`: 68 passed (49 before plus 19 new). Live (2026-10-14, MongoDB and generation configured): `BNS section 103` ok `bns:103`; `IPC section 302` ok `ipc:302` (status `repealed`); `IPC section 304` clarification_needed (`ipc:304`, `ipc:304A`, `ipc:304B`); "What does section 103 say?" clarification_needed, no MongoDB call; `BNS section 999` not_found; "What is the punishment for murder?" unsupported, no MongoDB call; `generate_answer` on `BNS section 103` answered, high confidence, cited `bns:103`; on `BNS section 999` no generation. **Pending participant:** Open WebUI chat (covered offline only).

**Unfinished:** none. **Known limitations:** exact lookup only; small fixed phrase set; explicit selection only; stored status and version are source metadata, not legal currency; one demo caller, `public` only.
