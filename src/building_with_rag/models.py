"""Typed contracts. No behaviour. Later stories extend these additively."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Mode = Literal["semantic", "hybrid", "hybrid-reranked", "structured", "decomposition", "hyde", "agentic"]


FilterValue = Annotated[str, Field(min_length=1, max_length=64)]


class SemanticFilters(BaseModel):
    """Caller filters only narrow the scope; unknown fields are refused."""

    model_config = ConfigDict(extra="forbid")

    act: list[FilterValue] = Field(default_factory=list, max_length=10)
    status: list[FilterValue] = Field(default_factory=list, max_length=10)
    access_level: list[FilterValue] = Field(default_factory=list, max_length=10)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    pattern: Mode
    caller_id: str | None = None
    filters: SemanticFilters | None = None
    limit: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = False
    required_acts: list[str] = Field(default_factory=list)
    chapter: str | None = None

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must contain non-whitespace text")
        return v


class RetrievedChunk(BaseModel):
    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str | None = None
    score: float | None = None
    source: str | None = None
    source_url: str | None = None
    source_file: str | None = None
    page: int | None = None
    chapter: str | None = None
    status: str | None = None
    access_level: str | None = None
    # Hybrid only (Story 4.1); null in semantic results. `score` is the fused score there.
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None
    fused_rank: int | None = None
    # Hybrid-reranked only (Story 4.2); null elsewhere. `score` is the re-rank score there.
    rerank_score: float | None = None
    rerank_rank: int | None = None
    # Structured only (Story 5.1); null elsewhere. Source metadata, not legal currency.
    source_status_version: str | None = None


class OmittedCandidate(BaseModel):
    chunk_id: str
    omitted_reason: str


class SubquestionEvidence(BaseModel):
    subquestion: str
    status: Literal["evidenced", "no_evidence"]
    results: list[RetrievedChunk] = Field(default_factory=list)
    reason: str | None = None


class StructuredSignals(BaseModel):
    intent: Literal["exact_lookup", "filter", "aggregation"]
    act: str | None = None
    section_number: str | None = None
    chapter: str | None = None


class GenerationResult(BaseModel):
    outcome: Literal["answered", "insufficient_evidence", "unavailable", "malformed"]
    answer: str | None = None
    claims: list[Any] = Field(default_factory=list)
    citations: list[Any] = Field(default_factory=list)
    supporting_passages: list[RetrievedChunk] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    trace: list[Any] = Field(default_factory=list)
    context_outcome: str | None = None
    confidence: str | None = None
    draft_answer: str | None = None
    issues: list[Any] = Field(default_factory=list)
    attempts: int = 0
    low_confidence_reason: str | None = None


class QueryResult(BaseModel):
    pattern: Mode
    status: str
    message: str
    trace: list[Any] = Field(default_factory=list)
    results: list[RetrievedChunk] = Field(default_factory=list)
    generation: GenerationResult | None = None
    omitted_candidates: list[OmittedCandidate] = Field(default_factory=list)
    subquestions: list[SubquestionEvidence] = Field(default_factory=list)
    hyde_direct_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_query_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_hypothetical_text_debug: str | None = None


# --- OpenAI-compatible contracts -------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class RagOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: Mode | None = None
    act: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)
    access_level: list[str] = Field(default_factory=list)
    # The Open WebUI Pipe nests act/status here; merged with the flat lists in chat.
    filters: SemanticFilters | None = None
    limit: int | None = Field(default=None, ge=1, le=20)
    required_acts: list[str] = Field(default_factory=list)
    chapter: str | None = None

    @field_validator("act", "status", "access_level", "required_acts", mode="before")
    @classmethod
    def none_means_empty_list(cls, value: Any) -> Any:
        # The Open WebUI Filter/Pipe emits ``null`` for an unset multi-select.
        return [] if value is None else value


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    n: int = 1
    rag_options: RagOptions | None = None


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class ChatCompletionDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "building-with-rag"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


class ErrorBody(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


# --- MongoDB schema contracts (typed placeholders; nothing is provisioned) --------


class ChunkDocument(BaseModel):
    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str | None = None
    status: str | None = None
    access_level: str | None = None
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_model_version: str | None = None


class CallerDocument(BaseModel):
    caller_id: str
    access_levels: list[str] = Field(default_factory=list)
