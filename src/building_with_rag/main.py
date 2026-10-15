import hmac
import time
import uuid
from collections.abc import Iterable, Iterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from building_with_rag.config import Settings, get_settings
from building_with_rag.models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorBody,
    ErrorEnvelope,
    ModelList,
    ModelObject,
    QueryRequest,
    QueryResult,
    SemanticFilters,
)
from building_with_rag.registry import MODE_BY_MODEL_ID, MODEL_IDS, run_pattern, start_pattern
from building_with_rag.retrieval import RetrievalUnavailable

app = FastAPI(title="Building with RAG")


def _openai_error(status: int, message: str, type_: str, code: str | None = None) -> JSONResponse:
    body = ErrorEnvelope(error=ErrorBody(message=message, type=type_, code=code))
    return JSONResponse(status_code=status, content=body.model_dump())


class OpenAIError(Exception):
    def __init__(self, status: int, message: str, type_: str, code: str | None = None):
        self.status, self.message, self.type, self.code = status, message, type_, code


@app.exception_handler(OpenAIError)
async def _openai_error_handler(_: Request, exc: OpenAIError) -> JSONResponse:
    return _openai_error(exc.status, exc.message, exc.type, exc.code)


@app.exception_handler(RetrievalUnavailable)
async def _retrieval_unavailable_handler(request: Request, exc: RetrievalUnavailable) -> JSONResponse:
    if request.url.path.startswith("/v1/chat/"):
        return _openai_error(503, f"Retrieval unavailable: {exc}", "server_error", "retrieval_unavailable")
    return JSONResponse(status_code=503, content={"detail": f"Retrieval unavailable: {exc}"})


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    if request.url.path.startswith("/v1/chat/"):
        first = errors[0] if errors else {}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        msg = f"{loc}: {first.get('msg', 'invalid request')}" if loc else "invalid request"
        return _openai_error(400, msg, "invalid_request_error", "invalid_request")
    detail = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in errors]
    return JSONResponse(status_code=422, content={"detail": detail})


def resolve_caller(caller_id: str | None, settings: Settings) -> str:
    """The seed resolves only the fixed local demo caller."""
    demo = settings.webui_demo_caller_id
    if caller_id is None or caller_id == demo:
        return demo
    raise HTTPException(status_code=403, detail="Unknown caller.")


def require_capstone_key(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """Shared-key check for the answer routes. An empty CAPSTONE_API_KEY enforces nothing."""
    key = settings.capstone_api_key
    if not key:
        return
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), key.encode()):
        return
    if request.url.path.startswith("/v1/chat/"):
        raise OpenAIError(401, "Invalid or missing API key.", "invalid_request_error", "invalid_api_key")
    raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResult, dependencies=[Depends(require_capstone_key)])
def query(request: QueryRequest, settings: Settings = Depends(get_settings)) -> QueryResult:
    request.caller_id = resolve_caller(request.caller_id, settings)
    return run_pattern(request, settings)


@app.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    return ModelList(data=[ModelObject(id=mid) for mid in MODEL_IDS.values()])


def _stream(completion_id: str, created: int, model: str, pieces: Iterable[str]) -> Iterator[str]:
    def frame(delta: ChatCompletionDelta, finish: str | None = None) -> str:
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[ChatCompletionChunkChoice(delta=delta, finish_reason=finish)],
        )
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    yield frame(ChatCompletionDelta(role="assistant"))
    try:
        for piece in pieces:
            if piece:
                yield frame(ChatCompletionDelta(content=piece))
    except Exception:  # noqa: BLE001 - never leave the stream open or leak details
        yield frame(ChatCompletionDelta(content="\n\n⚠ Answer generation failed unexpectedly."))
    yield frame(ChatCompletionDelta(), "stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", response_model=None, dependencies=[Depends(require_capstone_key)])
def chat_completions(
    body: ChatCompletionRequest, settings: Settings = Depends(get_settings)
) -> ChatCompletionResponse | StreamingResponse:
    pattern = MODE_BY_MODEL_ID.get(body.model)
    if pattern is None:
        raise OpenAIError(
            404, f"The model '{body.model}' does not exist.", "invalid_request_error",
            "model_not_found",
        )
    if body.n != 1:
        raise OpenAIError(400, "Only n=1 is supported.", "invalid_request_error", "invalid_n")
    question = next((m.content for m in reversed(body.messages) if m.role == "user"), None)
    if not question or not question.strip():
        raise OpenAIError(
            400, "A non-empty user message is required.", "invalid_request_error",
            "missing_user_message",
        )

    fields: dict = {"question": question, "pattern": pattern}
    opts = body.rag_options
    if opts is not None:
        nested = opts.filters or SemanticFilters()
        merged = {k: list(dict.fromkeys([*getattr(opts, k), *getattr(nested, k)]))
                  for k in ("act", "status", "access_level")}
        if any(merged.values()):
            fields["filters"] = SemanticFilters(**merged)
        if opts.limit is not None:
            fields["limit"] = opts.limit
        fields["required_acts"] = opts.required_acts
        fields["chapter"] = opts.chapter
    # Server-controlled: never trust browser-supplied identity or generation settings.
    fields["caller_id"] = settings.webui_demo_caller_id
    fields["generate_answer"] = True
    try:
        request = QueryRequest(**fields)
    except ValidationError as exc:
        raise OpenAIError(
            400, "Invalid request.", "invalid_request_error", "invalid_request"
        ) from exc

    # Same path as /v1/query: one retrieval, and one generation iterator that is either
    # drained there or rendered here. Retrieval errors surface before any stream starts.
    run = start_pattern(request, settings)
    if run.events is None:
        pieces: Iterable[str] = [run.result.message]
    else:
        from building_with_rag.generation import render

        pieces = render(run.events)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    if body.stream:
        return StreamingResponse(
            _stream(completion_id, created, body.model, pieces),
            media_type="text/event-stream",
        )
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=body.model,
        choices=[ChatCompletionChoice(message=ChatCompletionMessage(content="".join(pieces)))],
    )
