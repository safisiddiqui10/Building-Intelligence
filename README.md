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
3. The reply must be the honest "not implemented" placeholder, not a local preview.

Extract the BNS/IPC sections into `data/processed/` (no credentials needed): `uv run python -m building_with_rag.ingestion`. Details in `docs/architecture.md`.
