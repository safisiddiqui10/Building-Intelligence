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
3. With `semantic`, the reply is a real streamed answer: a `DRAFT — checking evidence` start, then a closing `Confidence` block with sources (or a labelled low-confidence/unavailable line). Other modes still reply "not implemented".
4. If `CAPSTONE_API_KEY` is set, `/v1/query` and chat require `Authorization: Bearer <key>`; set the same key in the Pipe's `capstone_api_key` valve. Empty means no auth (local development only).

Extract the BNS/IPC sections into `data/processed/` (no credentials needed): `uv run python -m building_with_rag.ingestion`. Details in `docs/architecture.md`.
Load MongoDB and build the vector index (needs `MONGODB_URI` and `VOYAGE_API_KEY`; first run about 40 minutes): `uv run python -m building_with_rag.ingestion.load_mongodb`.

Semantic retrieval diagnostic (compact output): `uv run python -m building_with_rag.retrieval "What is the punishment for murder?" --limit 5` (readiness: `--check`).

Grounded answer diagnostic (needs the `GENERATION_*` settings; compact JSON): `uv run python -m building_with_rag.generation "What is the punishment for murder?" --limit 5`. `POST /v1/query` with `"generate_answer": true` returns the same in `generation`.
