# Story 3.2: Streamed Answer, Confidence and Open WebUI

## Purpose

Connect real RAG generation to the Open WebUI streaming path that Story 1.1 built. Today `POST /v1/chat/completions` returns a placeholder and never retrieves or generates. After this story, a question typed in Open WebUI runs the selected pattern (`semantic` is the only real one), builds the bounded context, and streams the answer text from the configured LLM. It then closes with the validated confidence and the sources and checks behind it.

`/v1/query` and `/v1/chat/completions` are one path, not two. Both call the same `run_pattern` for the selected pattern, share the same `QueryResult` with its nested `GenerationResult`, and share retrieval, context assembly, answer validation, citations and the final outcome. Chat renders that result as text; `/v1/query` stays the richer JSON view for learning and inspection. Do not build a second chat path.

Only answer writing streams. Retrieval and context assembly stay non-streamed. For one request there is exactly one generation/validation operation. Its text events feed the stream, and its final result is the `GenerationResult` that the same request returns. The stream and the result describe the same operation, never two independent LLM calls.

Open WebUI cannot retract text it has already shown. So the answer is streamed as a visibly labelled **draft**, and only validation can promote it. Nothing unvalidated is ever shown as the answer.

Out of scope: agentic behaviour, new retrieval modes (the other five modes stay placeholders), a Pipe or Filter rewrite, any Open WebUI UI plugin, a broad test suite, an evaluation harness, follow-up memory, multi-user authorization and production observability.

## Prerequisites

- Stories 1.1, 2.1, 2.2, 2.3 and 3.1 are complete. No `docs/config.yaml` exists, so use the repository root.
- Read before you start: `docs/architecture.md`, Story 1.1, Stories 2.1–3.1 (including the 3.1 completion record), and the code: `src/building_with_rag/{config,models,registry,main}.py`, `generation/{context,answer,__main__}.py`, `retrieval/semantic.py`, `tests/test_app.py`, `tests/test_generation.py`. Also read the supplied Open WebUI bundle notes: `open_webui_functions/{open-webui-operations.md,webui-instructions.txt,building_with_rag_ui_preview.py}`. Preserve all existing routes, models, settings, tests, the shared registry and uncommitted work (`open-webui-bundle/` and `open_webui_functions/` are untracked; do not move or delete them).
- Open WebUI facts you must design around, from the bundle:
  - The **Building with RAG** Pipe posts `model=rag-<pattern>`, `stream: true`, the latest user message and `rag_options` to `{capstone_base_url}/chat/completions`. It sends `Authorization: Bearer <valve key>` when its admin-only `capstone_api_key` valve is set.
  - The Pipe relays only `choices[].delta.content` and stops at `[DONE]`. A non-2xx reply is shown as `Capstone API error (HTTP n): <error.message>`, so errors before streaming must use the OpenAI error envelope. Because the Pipe forwards content only, all confidence, source and check information must travel **as streamed text**.
  - The Pipe's read timeout is 120 s per read, so keep text or a status line flowing during a retry. Open WebUI keeps its own copy of the Pipe and Filter, so this story does **not** edit them.
  - The bundle turns title, tag, follow-up and search-query generation off, so one chat message is one capstone request.
- `.env` is untracked. Use **exactly** the existing names in `.env.example`: `CAPSTONE_API_KEY`, `WEBUI_DEMO_CALLER_ID`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME`. Do not add, rename, replace or remove a variable, and do not edit `.env.example`. No new setting. Bounds go in named code constants. Never print, log or return a key, header or provider response body.
- Add no dependency. `httpx` is already a runtime dependency; use its streaming support.

## Work to do

### 1. One generation operation that yields events

Refactor `generation/answer.py` so the provider call and validation are one operation exposed as an event iterator. Keep the existing `generate_answer` and `add_generation` names working, and keep the passing Story 3.1 tests meaningful. Where a test hard-coded the 3.1 non-streaming JSON contract, update it to the contract below and say so in the completion record.

- The core is one function (for example `generate_events(question, results, settings, *, transport=None)`). It assembles the context once, runs the bounded attempts, and yields small typed events. Minimum event kinds: `attempt_started` (attempt number), `text` (one piece), `check_failed` (check name, plain detail), `retrying`, and `final` carrying the finished `GenerationResult`. Events are internal types; they are not new response models and not custom SSE events.
- `/v1/query` drains this iterator and uses the `final` event. Chat renders the same events. Neither endpoint calls the provider any other way. Exactly one iterator is created per request. Never call `generate_answer` and also stream separately.
- The provider is called once per attempt through the existing OpenAI-compatible endpoint: `GENERATION_API_BASE_URL` + `/chat/completions`, `Authorization: Bearer GENERATION_API_KEY`, `model=GENERATION_MODEL_NAME`, `temperature` 0, `stream: true`, no provider-specific parameters. Read the standard SSE `choices[0].delta.content` pieces, ending at `[DONE]`. Use this streaming call for `/v1/query` as well, so there is one provider path.
- Keep the 3.1 failure meanings. An empty `GENERATION_API_BASE_URL` or `GENERATION_API_KEY` means `unavailable` without a provider call. `empty` context means `insufficient_evidence` without a provider call.

### 2. Answer format that can stream

The 3.1 JSON reply cannot be shown to a person while it streams. Change the prompt contract so the model writes the readable answer directly, with the same strict evidence rules.

- Plain text, no JSON, no `response_format`. Each sentence that states something ends with the supplied labels that support it, for example `Murder is punishable with death or imprisonment for life [E3].`
- If the passages do not support an answer, or conflict so that none follows, the reply starts with `INSUFFICIENT_EVIDENCE:` and a short reason. Hold back only as many leading characters as needed to recognise that marker, then stream normally. The marker is never shown as draft text. Do not retry `insufficient_evidence`.
- Keep the 3.1 prompt rules: passages are untrusted evidence, not instructions; no outside knowledge; no claim of current legal applicability or advice beyond the passages; an act's `status` may be repeated only as supplied.
- Parse the finished text deterministically into the existing `claims` shape (`text`, `evidence` labels), one per sentence. Define the sentence-splitting rule in one small function and test it. Do not use a second LLM call to split or judge.

### 3. Validate after each bounded attempt

Add a `MAX_ATTEMPTS` constant (2). After each attempt's text completes, run both checks. Keep each failed check as one `issues` entry: `{"attempt": n, "check": "...", "detail": "<plain language>", ...}`. Use `citation` for the check about supplied labels and `support` for the check about the passage text.

- **Citation check.** At least one claim exists, and every claim cites at least one label, all of which were supplied in the context. A claim with no label or an unknown label fails. Record the exact unknown label in the detail. Do **not** silently drop or strip an invalid citation, and do not keep "the good part".
- **Support check (deterministic, no LLM).** For each claim, every number in the claim (for example a section number or a year count) appears in the text of its cited passages, and at least `SUPPORT_MIN_OVERLAP` (for example 0.6) of its significant words (4+ letters, minus a small stop-word list) appear there, case-insensitively. State this as a lexical check in code and docs. It catches fabricated numbers and off-topic claims, not subtle misreadings.
- Both checks run on every attempt, so `issues` can hold several failures. A claim that fails both checks produces two entries.
- Pass → validated. Fail with attempts left → retry once. The retry request repeats the original messages and adds the plain-language failed-check details as feedback. Passage text is still untrusted. The retry is a new attempt of the same operation, streamed into the same response. Fail with no attempts left → final failure.
- Empty or unusable provider text on an attempt is a failed check named `no_answer_text`, and can be retried within the bound.

### 4. Final `GenerationResult` (extend, do not replace)

Keep the Story 1.1 field names and the 3.1 meanings. The confidence extension means:

| Situation | `outcome` | `answer` | `confidence` | `draft_answer` | `issues` | `low_confidence_reason` |
|---|---|---|---|---|---|---|
| Final attempt passes both checks | `answered` | validated text | `"high"` | `null` | failed checks of earlier attempts, if any | `null` |
| Final check fails, text exists | `malformed` | `null` | `"low"` | the last attempt's text, unedited | every failed check, all attempts | short reason, for example "citation and support checks failed after 2 attempts" |
| `INSUFFICIENT_EVIDENCE:` | `insufficient_evidence` | `null` | absent (`null`) | `null` | `[]` | `null` (the model's reason goes in `trace`, as in 3.1) |
| No answer text could be judged (empty output) | `malformed` | `null` | absent (`null`) | `null` | the `no_answer_text` entries | `null` |
| Provider fails (timeout, connection, non-2xx, stream cut) | `unavailable` | `null` | absent (`null`) | any partial text received, else `null` | the failure with a safe reason | `null` |
| Missing `GENERATION_*` settings | `unavailable` | `null` | absent (`null`) | `null` | `[]` (reason in `trace`) | `null` |

- `confidence` is only `"high"` or `"low"`, or absent when nothing could be judged. No numeric score. Do not add outcome values.
- `claims`, `citations` and `supporting_passages` are filled (resolved from the retained context, as in 3.1) only for `answered`. A failed draft is never promoted, and its labels are never silently corrected.
- `attempts` is the number of provider calls actually started. `provider`, `model`, `context_outcome` and `trace` keep their 3.1 meanings. `trace` also records, per attempt, whether the text streamed fully and which checks ran.
- `QueryResult.message` stays one plain sentence that never contains answer text. Add one sentence each for a low-confidence draft ("A draft was produced but failed evidence checks; no validated answer is returned.") and a stream failure. Reuse the 3.1 sentences otherwise.
- Never put confidence, outcome or citations in new top-level `QueryResult` fields.

### 5. One chat path through `run_pattern`

- In `registry.py`, extend the shared path so chat and `/v1/query` both go through `run_pattern` for the selected pattern and the shared generation helper, with one retrieval call per request. The preferred shape is that `run_pattern` runs the pattern, then either drains the event iterator (`/v1/query`) or hands the iterator back to the caller to stream (chat). It must not retrieve twice and must not build a second `QueryResult` shape. The chat handler keeps its existing model lookup, `n`, empty-user-message and `rag_options` mapping, and keeps `caller_id` from `WEBUI_DEMO_CALLER_ID`. The server forces `generate_answer=true` for chat. A browser never supplies identity, access level or generation settings.
- Retrieval, and any retrieval failure, run **before** the response starts, so a failure is an HTTP error and never a half-open stream. A `RetrievalUnavailable` on a chat request returns the OpenAI error envelope (status 503, short safe message), so the Pipe shows `Capstone API error (HTTP 503): ...`. `/v1/query` keeps its existing 503 body.
- `rag-semantic` is real. The other five model IDs still use the placeholder and stream it as role, content, stop, `[DONE]` exactly as before. `GET /v1/models` and the Pipe's mode selection are unchanged. Chat still ignores `rag_options.pattern` and uses the selected model, as before.
- `stream: true`: role frame, then content frames as the generation events arrive, then a stop frame, then `data: [DONE]`, using the existing `ChatCompletionChunk` models. Stream the LLM's text pieces as they arrive, not buffered into one frame. No custom SSE events. Response `media_type` stays `text/event-stream`.
- `stream: false`: normal OpenAI JSON. The content is the same rendered text the stream would have produced for that operation.
- If the provider fails after text began, emit one final clearly labelled line, for example `⚠ Answer generation unavailable: provider connection failed. The text above is an incomplete DRAFT, not a final answer.` Then send stop and `[DONE]`. The stream never hangs open and never ends without `[DONE]`.
- Use the existing `_stream` framing helper. Do not duplicate it.

### 6. Streamed display (one small renderer)

Add one pure render function (for example `generation/render.py`) that turns the events into text pieces. Chat uses it for both stream modes. Keep the text plain Markdown that Open WebUI renders well.

For each attempt, in order:

1. Before the attempt's first text piece, emit `DRAFT — checking evidence` on its own line, then a blank line. Every unvalidated attempt carries this prefix, including the first.
2. Stream the text pieces unchanged.
3. On a failed check with an attempt left, emit one line such as `Check failed: <plain detail>. Retrying (attempt 2 of 2)…`, then start the next attempt at step 1. Earlier text stays visible, because it cannot be retracted. The retry line makes that clear.
4. **Validated:** emit a closing block with `Confidence: high` and `Checks passed: citations, support (attempt n of 2)`, followed by a `Sources` list with one line per citation: label, act-qualified section (for example `bns:103`), heading, source and page. The last attempt's text is thereby labelled as the validated answer.
5. **Failed final check:** emit `DRAFT — low confidence, not the final answer` plus the short `low_confidence_reason` and the failed check details. Do not show a Sources list as if validated.
6. **`INSUFFICIENT_EVIDENCE`:** emit a plain line that the retrieved passages do not support an answer, with the model's short reason. Do not use the draft prefix, and give no confidence.
7. **Provider failure or unavailable:** emit the unavailable line from section 5.

Keep the closing block short. Full detail (`issues`, `attempts`, `trace`, `claims`, `supporting_passages`) stays in `/v1/query`, which remains the richer learning view.

### 7. Auth with `CAPSTONE_API_KEY`

The setting exists, but the app does not enforce it yet. When `CAPSTONE_API_KEY` is non-empty, `POST /v1/query` and `POST /v1/chat/completions` require `Authorization: Bearer <CAPSTONE_API_KEY>`. A missing or wrong key returns 401, as the OpenAI error envelope for chat and the normal `{"detail": ...}` for `/v1/query`. Compare in constant time. Never echo the key, and never log or return the supplied header. When it is empty, nothing is enforced, which is local-development behaviour you must document. `/healthz` and `/v1/models` stay open. Use a single dependency for both routes. The Pipe already sends the key from its admin-only valve, so the Pipe needs no change.

### 8. Offline tests (small, targeted)

Extend `tests/test_generation.py` and `tests/test_app.py`. Fake the provider with `httpx.MockTransport` returning SSE bodies, and fake retrieval as the existing tests do. No network, MongoDB, Voyage or live server. Add a few focused tests only, covering at least:

- Streamed text pieces arrive in order, with a valid OpenAI SSE shape: role, content frames, stop, `[DONE]`.
- A validated first attempt: `outcome="answered"`, `confidence="high"`, citations resolved to act-qualified sections, and the closing block lists the same citations.
- A failed first attempt and passing retry: both attempts' text appears in the stream with the DRAFT prefix and a retry line, `attempts == 2`, and the first failure is kept in `issues`.
- Two failed attempts: `outcome="malformed"`, `confidence="low"`, `draft_answer` set, `answer` null, `citations` empty, the stream ends with `DRAFT — low confidence, not the final answer`, and `issues` holds every failed check including the unknown label un-stripped.
- Citation failure and support failure are each detected (unknown label; fabricated number or off-topic claim).
- `INSUFFICIENT_EVIDENCE:` → `insufficient_evidence`, no draft prefix, no retry, confidence absent.
- Provider cut mid-stream, and provider HTTP 502 before any text → `unavailable`; stream still ends with the unavailable line, stop and `[DONE]`.
- **Single operation:** the fake transport counts provider calls. One chat request with one passing attempt makes exactly one provider call. The streamed closing block and the returned `GenerationResult` agree on confidence, outcome, attempts and citation count. The same fake provider through `/v1/query` yields an equal `GenerationResult`.
- Chat with `stream: false` returns the same rendered text as the stream concatenated.
- Chat for a non-semantic model still streams the placeholder and performs no retrieval or provider call. A retrieval failure on chat returns the 503 OpenAI error envelope before any stream.
- Auth: no key configured → open; key configured → 401 for no or wrong header on both routes, success with the right header, `/healthz` and `/v1/models` open.

Keep every existing test passing (update only what the contract change above forces).

### 9. Update the docs

- `docs/architecture.md`: extend it, do not rewrite earlier sections. Update the **API surface** chat line and the **Grounded answer generation** section (which currently says chat is a placeholder and confidence is deferred). Add a short "Streamed answer and confidence" section recording: one path through `run_pattern`; only answer writing streams; one operation per request; the text-answer contract and `INSUFFICIENT_EVIDENCE:`; `MAX_ATTEMPTS`; the citation and support checks (and that support is lexical); the confidence table; the DRAFT labelling rule and why (Open WebUI cannot retract); stream failure behaviour; `CAPSTONE_API_KEY` auth. State what chat shows against what `/v1/query` returns.
- `open_webui_functions/open-webui-operations.md`: add a short "What chat shows now" section covering the draft, retry and check lines, the closing confidence block, sources and the unavailable line. Add a note that the capstone API must be started with `CAPSTONE_API_KEY` matching the Pipe's `capstone_api_key` valve when it is set, and that the Pipe and Filter are unchanged so nothing needs re-pasting.
- `README.md`: update the Open WebUI smoke check so the reply is now a real answer, not the placeholder. Add a line for `CAPSTONE_API_KEY` auth.
- `docs/manual-tests.md`: add a "Story 3.2" entry that holds the participant's Open WebUI steps (section "Participant live checks" below). Do not add curl or Postman steps for this story.
- Do not edit `.env.example`, the Pipe or the Filter.

## Verification rules for the coding assistant (Claude / Codex)

**Do not test with curl or any live HTTP or server call.** Do not start the app or `uvicorn`, do not call `/healthz`, `/v1/query`, `/v1/models` or `/v1/chat/completions` over the network, and do not call MongoDB, Voyage, the generation provider or Open WebUI. Do not run the diagnostic commands `python -m building_with_rag.generation` or `retrieval`, since they make live calls. Verify only with:

1. `uv run ruff check .`
2. `uv run pytest -q` (offline, in-process tests with faked provider and retrieval; FastAPI's in-process test client is fine because it opens no socket)
3. `git status`, showing `.env` untracked and unchanged, `.env.example` unchanged, `data/raw/` and `data/processed/` untouched, and no secret in any file or output.

Report only these results. The live checks belong to the participant and are recorded as **pending participant**.

## Participant live checks (run by the participant, not the assistant)

These are run by the participant directly in the **Open WebUI interface** at `http://127.0.0.1:8080`, not in Postman or curl. Postman would only repeat Story 3.1's checks and cannot show the stream as a person sees it. Prerequisites: the Story 2.3 data and index, `VOYAGE_API_KEY`, working `GENERATION_*` values in `.env`, the capstone API running (`uv run uvicorn building_with_rag.main:app --port 8000`), and Open WebUI started with `manage_open_webui`. Select **Building with RAG** with **semantic** in the RAG-options chip. Ask one question at a time, because the free Voyage key is rate limited. For `/v1/query` counts, the participant can compare with the same question in `/v1/query` with `generate_answer: true`, which shows the same fields.

| # | Check | What to do | Compact expected result |
|---|---|---|---|
| 1 | Supported answer | Ask "What is the punishment for murder?" | Text streams in pieces. It opens with `DRAFT — checking evidence`. It closes with `Confidence: high`, `Checks passed`, and a Sources list with BNS 103 (and possibly IPC 302). Equivalent result: `outcome=answered`, `confidence=high`, `attempts` 1 (or 2 with a retry line), `citations` ≥ 1, `issues` empty or only earlier-attempt failures. |
| 2 | Unsupported question | Ask "What is the penalty for jaywalking in Delhi under the Motor Vehicles Act?" | One plain line that the retrieved passages do not support an answer, with a short reason. No draft prefix, no confidence, no Sources list. Equivalent result: `outcome=insufficient_evidence`, `answer` null, `confidence` absent, `citations` 0, `results` still 5. |
| 3 | Invalid or unvalidated output | Deliberately force a failure. Stop the app, set a temporary invalid `GENERATION_MODEL_NAME` (for example `not-a-model`) in the shell for that run only (never edit `.env`), restart, and ask the murder question. Or use a low-quality model if the trainer provides one. Restore the setting afterwards. | The chat ends with a clearly labelled unavailable/error line (stop and end, no hang). If a draft was produced but failed checks, it closes with `DRAFT — low confidence, not the final answer` and a short reason instead. Equivalent result: `unavailable`, or `malformed` + `confidence=low` + `draft_answer` set + `answer` null + `citations` 0 + `issues` ≥ 1. |
| 4 | Stream shape | Watch check 1 as it renders. | Text appears incrementally (not one block at the end), the message finishes cleanly, and the Pipe's status line shows the RAG options. No raw JSON, no `[DONE]` text and no error toast in the chat. |
| 5 | Auth | If `CAPSTONE_API_KEY` is set in `.env`, set the Pipe valve `capstone_api_key` to the same value and repeat check 1 (works). Then clear or change the valve and send a question. | With the matching key, check 1 works. With a wrong or missing key, the chat shows `Capstone API error (HTTP 401): ...` and no answer. Restore the valve afterwards. |
| 6 | Mode selection and other modes | Choose `hybrid` in the chip and send a question. | The same "not implemented" placeholder as before. The model list still shows the same modes. |

If `GENERATION_*`, `MONGODB_URI` or `VOYAGE_API_KEY` is missing, the participant stops and notes it, and does not fake a result.

## Completion checks

1. `uv run ruff check .` passes.
2. `uv run pytest -q` passes, including every pre-existing test and the new offline tests. Record the pass count.
3. By code and test inspection, confirm: `/v1/query` and chat both go through `run_pattern` and the shared generation iterator; one retrieval call and one generation iterator per request; no second chat path; only the five existing `.env.example` names are used; no new setting; the other five modes still return placeholders.
4. `git status` shows `.env` untracked and unchanged, `.env.example`, `data/raw/` and `data/processed/` unchanged, the Pipe and Filter files unchanged, and no secret in any file or output.
5. The live checks in the table above are **not run by the assistant**.

## Handover

When done, add a **Completion record** to the end of this file with:

- Files created or changed, and the exact offline commands run (`ruff`, `pytest`, `git status`) with their results (counts and statuses only).
- Which 3.1 tests, if any, were updated for the new streaming/text contract, and why.
- The design as implemented: the event types, `MAX_ATTEMPTS`, the support-check threshold, and where `run_pattern` hands the iterator to chat.
- **Live verification: pending participant.** List the six participant checks above as `pending participant`. Do not write results, outcomes, counts or status codes for them, and do not mark them passed. State plainly that the assistant did not run curl, a server, or any live HTTP call.
- Anything unfinished, with the reason.
- Known limitations: validated means the checks passed, not that the legal reading is right; the support check is lexical; a draft stays visible in Open WebUI after a failed check because it cannot be retracted; `semantic` is the only real mode; no agentic behaviour, memory, evaluation harness or multi-user controls; `CAPSTONE_API_KEY` is a single shared key, not user authorization; an empty `CAPSTONE_API_KEY` means no auth (local development only).

**For the next story:** reuse the same `run_pattern` and event iterator for the next real mode, and keep rendering from the shared `QueryResult.generation`. Extend `GenerationResult` additively. Do not add a second chat path.

## Completion record

**Files created:** `src/building_with_rag/generation/{events,checks,render}.py`, `tests/test_chat_streaming.py`. **Changed:** `generation/{answer,__init__,__main__}.py`, `registry.py`, `main.py`, `tests/{test_app,test_generation}.py`, `docs/architecture.md`, `docs/manual-tests.md`, `README.md`, `open_webui_functions/open-webui-operations.md`. `.env.example`, the Pipe and the Filter are unchanged. `.gitignore` shows a `/open-webui-bundle` line that I did not add.

**Offline commands run:** `uv run ruff check .`: all checks passed. `uv run pytest -q`: 30 passed. `git status`: `.env`, `.env.example` and `data/` unchanged.

**3.1 tests updated for the new contract:** the JSON-reply tests in `tests/test_generation.py` (answered, code fence, five malformed cases, request shape, failures) were rewritten for the plain-text-with-labels contract and a streamed (`stream: true`) provider body. Unconfigured, timeout/502, empty-retrieval and `generate_answer` flag tests keep their meaning. In `tests/test_app.py` the placeholder chat tests now use `rag-hybrid`, since `rag-semantic` is real, and settings are overridden so a local `.env` cannot affect them.

**Design as implemented:** events `AttemptStarted`, `Text`, `CheckFailed`, `Retrying`, `Final` (`generation/events.py`); `generate_events` is the one operation. `MAX_ATTEMPTS` = 2, `SUPPORT_MIN_OVERLAP` = 0.6 (numbers must also appear in the cited passage, heading, source or section id). `registry.start_pattern` runs retrieval once and returns an unstarted iterator: `run_pattern` drains it for `/v1/query`, and the chat handler passes it through `render` into the existing `_stream` framing. A claim with an unknown or missing label fails both checks, so it produces two `issues` entries. `DEFAULT_TRANSPORT` in `answer.py` is a test seam for faking the provider at app level. `CAPSTONE_API_KEY` is enforced on `/v1/query` and chat only when it is non-empty.

**Live verification: pending participant.** The assistant did not run curl, start a server, or make any live HTTP call.
1. Supported answer: pending participant
2. Unsupported question: pending participant
3. Invalid or unvalidated output: pending participant
4. Stream shape: pending participant
5. Auth: pending participant
6. Mode selection and other modes: pending participant

**Unfinished:** none in code. The Pipe's `START_HINT` still names a stale module path; it was left alone because the Pipe is out of scope.

**Known limitations:** validated means the checks passed, not that the legal reading is right; the support check is lexical; a draft stays visible in Open WebUI after a failed check because it cannot be retracted; `semantic` is the only real mode; no agentic behaviour, memory, evaluation harness or multi-user controls; `CAPSTONE_API_KEY` is one shared key, not user authorization, and an empty value means no auth (local development only).
