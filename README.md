# Building with RAG

One FastAPI app, one API. Chat is done by the separately running Open WebUI bundle.
See `docs/architecture.md`.

## Run

```
uv sync
uv run uvicorn building_with_rag.main:app
uv run pytest
uv run ruff check .
```

The app starts with no credentials and connects to nothing at startup.

## Environment file

Copy `.env.example` to `.env`. `.env` is untracked and secrets are never committed.

## Open WebUI smoke check

1. Once, from the extracted trainer bundle root:
   `powershell -ExecutionPolicy Bypass -File .\setup_open_webui.ps1`
   (macOS/Linux: `sh setup_open_webui.sh`). Day-to-day use `manage_open_webui`.
2. Start this API, open `http://127.0.0.1:8080`, select **Building with RAG**, choose **semantic**
   in the RAG-options chip and send a question.
3. With `semantic`, the reply is a real streamed answer: a `DRAFT — checking evidence` start, then a closing `Confidence` block with sources (or a labelled low-confidence/unavailable line). All seven modes are real (Story 5.2 added `decomposition` and `hyde`; Story 6.1 added `agentic`).
4. If `CAPSTONE_API_KEY` is set, `/v1/query` and chat require `Authorization: Bearer <key>`; set the same key in the Pipe's `capstone_api_key` valve. Empty means no auth (local development only).

Extract the BNS/IPC sections into `data/processed/` (no credentials needed): `uv run python -m building_with_rag.ingestion`. Details in `docs/architecture.md`.
Load MongoDB and build the vector index (needs `MONGODB_URI` and `VOYAGE_API_KEY`; first run about 40 minutes): `uv run python -m building_with_rag.ingestion.load_mongodb`.

Semantic retrieval diagnostic (compact output): `uv run python -m building_with_rag.retrieval "What is the punishment for murder?" --limit 5` (readiness: `--check`).

Grounded answer diagnostic (needs the `GENERATION_*` settings; compact JSON): `uv run python -m building_with_rag.generation "What is the punishment for murder?" --limit 5`. `POST /v1/query` with `"generate_answer": true` returns the same in `generation`.

Hybrid retrieval diagnostic (semantic + keyword, fused by reciprocal rank; needs the keyword index from `load_mongodb --skip-embed`): `uv run python -m building_with_rag.retrieval "culpable homicide not amounting to murder" --mode hybrid`.

Structured exact lookup (no LLM, no embeddings; needs only MongoDB): `uv run python -m building_with_rag.retrieval "BNS section 103" --mode structured`.

Hybrid-reranked diagnostic (hybrid candidates, one re-rank call; needs `RERANK_API_KEY`, otherwise a clear "not set" message): `uv run python -m building_with_rag.retrieval "What happens to someone who kills another person on purpose?" --mode hybrid-reranked`.

Decomposition diagnostic (one model split, then a required search per subquestion; needs `MONGODB_URI`, `VOYAGE_API_KEY` and `GENERATION_*`): `uv run python -m building_with_rag.retrieval "Compare the punishment for murder under the BNS and under the IPC." --mode decomposition`.

HyDE diagnostic (a hypothetical passage is embedded only to find real passages; same settings): `uv run python -m building_with_rag.retrieval "What happens to a person who knowingly buys a phone that a thief stole?" --mode hyde`.

Agentic diagnostic (a fixed controller over the structured, semantic and HyDE adapters, at most two calls, bounded by `AGENTIC_DEADLINE_SECONDS`): `uv run python -m building_with_rag.retrieval "What happens to a person who knowingly buys a phone that a thief stole?" --mode agentic`.
