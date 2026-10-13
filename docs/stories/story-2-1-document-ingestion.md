# Story 2.1: Document Ingestion

## Purpose

Turn the two supplied PDFs, the Bharatiya Nyaya Sanhita 2023 (BNS) and the Indian Penal Code 1860 (IPC), into a small corpus that a person can open and read: one JSON record per legal section, one file per act.

This story only parses and saves text. MongoDB, chunking, embeddings, vector indexes, retrieval and answer generation belong to later stories. Do not start any of them here.

This story follows Story 1.1. Reuse its application (`src/building_with_rag/`), its documents (`docs/architecture.md`, `README.md`) and its existing `.env.example`. Do not add, rename or remove any setting. Extraction needs no credentials and no `.env` values.

## Prerequisites

- Story 1.1 is complete. Python 3.12 and UV are installed.
- No `docs/config.yaml` exists, so use the repository root.
- Inputs, which must already be in `data/raw/`:
  - `BNS_2023_bare_act.pdf`
  - `IPC_1860_bare_act.pdf`
  - `PROVENANCE.md`
- Read `docs/architecture.md`, Story 1.1, the existing code and `data/raw/PROVENANCE.md` before you start.
- Preserve the original PDFs and `PROVENANCE.md` exactly. Never edit, move, rename or regenerate them. If any input is missing, stop and report which one in the handover. Do not invent or download a replacement.
- Preserve all existing work, including uncommitted files, `.gitignore`, `.env` and `.env.example`.
- Note that `PROVENANCE.md` mentions an older `distillation/` parser that is not part of this repository. Do not look for it or depend on it. This story writes its own parser.

## Work to do

### 1. Check the inputs

Compute the SHA-256 of each PDF. Compare it with the value in `PROVENANCE.md`. If a hash differs, report it and treat it as a changed source (see step 5). Do not change `PROVENANCE.md` to match.

### 2. Choose a PDF parser

Pick one suitable PDF text-extraction library and add it with UV (`uv add <package>`). This updates `pyproject.toml` and `uv.lock`. Choose a library that gives text in reading order with page numbers, and note why you chose it. Keep it to one parser dependency. Do not add OCR, machine learning or layout-analysis tools.

If a page has no extractable text (a scanned page), do not try to repair it. Flag it as described in step 4.

### 3. Write the extractor

Add a small module inside the existing package, for example `src/building_with_rag/ingestion.py`, with a command that is run through UV, for example `uv run python -m building_with_rag.ingestion`. Keep it plain and readable. Do not connect to MongoDB or any provider. Do not import it from the app's startup path.

For each act, extract the sections of the enacted body text:

- Find each section by its number and heading. Take the section text as written, up to the start of the next section.
- Keep the wording as it appears in the PDF. Do not summarise, rewrite, translate, correct or reorder it. Allowed clean-up is limited to repeated page headers and footers, page numbers, line-break hyphenation, and extra spaces or blank lines. Join text that continues across a page break into one section.
- Read each section's chapter number and chapter title from the chapter heading above it.
- Skip material that is not a section: the cover, the arrangement of sections or index, and the BNS-to-IPC correspondence table. `PROVENANCE.md` says the BNS PDF contains these. If the table or index is skipped, mention it in the limitations.
- Be conservative with obvious duplicate extraction (for example a section found twice because of an index entry or a repeated page). Keep the one that comes from the body text and set `needs_review` on it if you are not sure.
- Keep BNS and IPC records strictly separate, even when a section number appears in both acts.

### 4. Record format

Write one JSON object per line (JSONL, UTF-8, no blank lines, no wrapping array) to:

- `data/processed/bns_sections.jsonl`
- `data/processed/ipc_sections.jsonl`

Create `data/processed/` if it does not exist. Every record must have exactly these required fields, and these values:

| Field | Value |
|---|---|
| `section_id` | `bns:<number>` for BNS, `ipc:<number>` for IPC (for example `bns:103`) |
| `act` | `BNS_2023` or `IPC_1860` |
| `act_label` | The act's plain name, taken from the PDF title (for example "Bharatiya Nyaya Sanhita, 2023") |
| `status` | `in_force` for BNS, `repealed` for IPC |
| `chapter` | String, for example `"I"` |
| `chapter_title` | The chapter title as printed |
| `section_number` | Integer |
| `heading` | The section heading as printed |
| `text` | The section text, not empty |
| `source_pdf` | The PDF file name, for example `BNS_2023_bare_act.pdf` |
| `source_sha256` | The SHA-256 of that PDF, computed at run time |
| `parser` | The parser library name |
| `parser_version` | The installed version of that library |
| `source_status_version` | Taken from `PROVENANCE.md` only (for example the retrieval date and the in-force or repealed note). If `PROVENANCE.md` does not give a value, use `null`. Do not invent it. |
| `needs_review` | `true` or `false` |

Rules:

- Do not invent provenance. If a value is unknown, use `null` and set `needs_review` to `true`.
- Section numbers can carry a letter suffix in the IPC (for example 304A, 376DA). `section_number` must still be an integer, so keep the number and put the printed form, such as `304A`, in the extra field `section_label` as well. Make `section_id` unique for each record. State in `docs/architecture.md` how suffixed sections are written in `section_id`. Choose one rule and use it for every such section, for example `ipc:304A`. If you choose a different rule, explain why.
- Extra extraction data is allowed in addition to the required fields, for example `page_start`, `page_end` and `section_label`. Extra fields never replace a required field.
- Set `needs_review` to `true` when the extractor is unsure: a missing or odd heading, a chapter that could not be read, a very short or empty-looking section, a suspected duplicate, or a scanned page with no text. Flag these records. Do not silently repair them.
- Never write a record with empty `text`. If a section would be empty, leave it out and list it in the run summary, or keep it with `needs_review: true` and the reason in an extra field. Choose one approach and state it in `docs/architecture.md`. The completion check "no empty section text" applies to the files as written, so the first approach is simpler.

### 5. Safe, repeatable writing

- Write each output to a temporary file in `data/processed/`, then replace the real file in one step. Never append to an existing output.
- If a valid corpus file already exists and its records show the same `source_sha256` and `parser_version` as the current run, leave it as it is and say so. Do not rewrite it unnecessarily.
- If a supplied corpus file exists but is not valid (not parseable, or built from a different `source_sha256`), regenerate that act's file in full, as one replacement. Never mix records from different source versions inside one file.
- Running the command twice must give the same line count and the same content. No duplicates.
- If a PDF is missing, do not write or change that act's output. Report it and carry on with the other act only if it is present.

### 6. Update `docs/architecture.md`

Add a short "Document ingestion" section, extending the existing document rather than rewriting it. Record:

- The parser chosen and why.
- The output format: JSONL, one record per line, the required fields and fixed values above, and where the files live.
- The command to run the extraction.
- The `section_id` form used in this corpus (`bns:<number>`, `ipc:<number>`) and how it relates to the `BNS-103` style shown in the Identifiers section. Say plainly which one the stored records use and which one is only display wording. Do not leave two competing identifier rules.
- The `section_id` rule for suffixed sections, and the empty-section approach.
- Known limitations: for example the skipped index and correspondence table, scanned or odd pages that are flagged, and any sections that need review.

Add one line to `README.md` on how to run the extraction. Do not change `.env.example`.

Do not add anything beyond this: no comprehensive validation framework, no reference-difference report, no broad security tests and no synthetic permission or access-level records.

## Completion checks

Keep these small. Run them and record what you actually did.

1. `uv sync` completes and `uv run ruff check .` passes. The existing `uv run pytest` still passes.
2. Run the extraction command. Both output files exist at the paths above.
3. Required fields: check that every line in both files parses as JSON, has all the required fields, uses only the allowed values, and has an integer `section_number`, a string `chapter`, and non-empty `text`. A short script or one-off command is enough.
4. Inspect representative records against the PDF for each act: at least three for BNS and three for IPC, including one section that runs across pages if the PDFs contain one. Include one suffixed IPC section and one section from a different chapter. Compare the wording to the PDF page.
5. Confirm that a section number found in both acts, for example 1 or 103, exists as two separate records with different `act` and `section_id`.
6. Run the extraction a second time. Confirm that the line counts and the contents are unchanged and that nothing was appended.
7. `git status` shows `data/raw/` untouched, `.env` untracked and unchanged, and `.env.example` unchanged.

## Handover

When done, add a **Completion record** to the end of this file with:

- **Inputs:** `data/raw/BNS_2023_bare_act.pdf`, `data/raw/IPC_1860_bare_act.pdf`, `data/raw/PROVENANCE.md`, and whether each hash matched `PROVENANCE.md`.
- **Output:** `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl`, with the record count and the count of `needs_review: true` records for each.
- **Parser:** the library, its version, and why it was chosen.
- **Files created or changed**, and the exact commands run with their results.
- **Records inspected:** which sections you compared against which PDF pages, and what you found.
- **Known limitations and anything left unfinished**, including any missing input.
- **How later chunking will keep the parent section and source information.** Later chunking must treat each record as the parent section. Every chunk must carry the parent's `section_id`, `act`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `heading`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version` and `needs_review`, plus the position of the chunk within the section. Chunks are cut from the section `text` only, never from the PDFs again, so a retrieved chunk can always be traced back to its section and its source file. Write this in your own words, specific to what you built.

Later stories (the MongoDB import, then embeddings) read these two files as they are. They do not re-parse the PDFs. If you change a field name or value here, the later importer breaks, so keep to the table above.

## Completion record

**Inputs:** `data/raw/BNS_2023_bare_act.pdf` (SHA-256 matches `PROVENANCE.md`), `data/raw/IPC_1860_bare_act.pdf` (matches), `data/raw/PROVENANCE.md`. All untouched.

**Output:** `data/processed/bns_sections.jsonl` (358 records, 2 `needs_review`) and `data/processed/ipc_sections.jsonl` (533 records, 2 `needs_review`).

**Parser:** `pypdf` 6.20.1, added with `uv add pypdf`. It is pure Python, gives text per page in reading order, and needs no OCR or layout tools.

**Files created or changed:** `src/building_with_rag/ingestion.py` (new), `pyproject.toml` and `uv.lock` (pypdf), `docs/architecture.md` ("Document ingestion" section), `README.md` (one line), `data/processed/*.jsonl`. `.env.example` unchanged.

**Commands run:**
- `uv add pypdf`: OK.
- `uv run python -m building_with_rag.ingestion`: wrote both files. BNS: all 358 sections. IPC: 533 records, including suffixed sections; 19 numbers listed as repealed or omitted in the PDF.
- Field check (short script): every line parses as JSON, has all 15 required fields, uses the fixed `act` and `status` values, has an integer `section_number`, a string `chapter` and non-empty `text`, and has a unique `section_id`. Passed for both files.
- Second run: both files reported "already current; left unchanged". The line counts (358 / 533) and the file contents were byte-identical after the first run, with no appended records. Deleting the outputs and re-running regenerated identical files.
- `uv run ruff check .`: passed. `uv run pytest -q`: 7 passed.
- `git status`: `data/raw/` and `.env.example` unchanged; `.env` untracked.

**Records inspected against the PDFs:**
- BNS `bns:1` (p. 74), `bns:103` (p. 125), `bns:2` (pp. 75 to 81, spans pages; text from "In this Sanhita" to the last clause matches), `bns:358` (pp. 236 to 237, spans pages), and `bns:7`/`bns:269` (flagged; headings and neighbouring section tails now correct).
- IPC `ipc:302` (p. 136), `ipc:304A` (p. 137, suffixed), `ipc:420` (p. 187), `ipc:127` (pp. 56 to 57, heading wraps lines), `ipc:330` and `ipc:511` (pp. 226 to 227, spans pages).
- BNS and IPC both have `103`-style numbers as separate records (`bns:103`, `ipc:103`) with different `act` and `section_id`.
- Found and fixed while inspecting: heading and group-heading lines leaking into the previous section, IPC separators written as `- -` or `.-`, and a stray IPC margin line mistaken for a section start.

**Known limitations and unfinished:** see "Document ingestion" in `docs/architecture.md`. In short: IPC footnote markers and amendment footnotes stay in the text; stray spaces inside words from the PDF text layer are not repaired; IPC 330 is printed as "320." in the body and is flagged; 376DA is not in the supplied PDF. No input was missing and no page lacked extractable text.

**How later chunking keeps the parent section and source information:** each JSONL line is the parent section. Chunking reads only `text` from these files, never the PDFs. Every chunk copies the parent's `section_id`, `act`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `heading`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version` and `needs_review`, plus its position in the section. Section text starts after the printed heading, so chunks should also carry `heading` as context. `bns:103` and `ipc:103` stay separate because `section_id` is act-qualified.
