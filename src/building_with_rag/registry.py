"""Single mode registry and dispatcher. Only `semantic` is implemented so far."""

from typing import get_args

from building_with_rag.config import Settings, get_settings
from building_with_rag.models import Mode, QueryRequest, QueryResult

MODES: tuple[str, ...] = get_args(Mode)
MODEL_IDS: dict[str, str] = {mode: f"rag-{mode}" for mode in MODES}
MODE_BY_MODEL_ID: dict[str, str] = {v: k for k, v in MODEL_IDS.items()}


def placeholder(request: QueryRequest) -> QueryResult:
    return QueryResult(
        pattern=request.pattern,
        status="not_implemented",
        message=f"The '{request.pattern}' pattern is not implemented yet.",
    )


def run_pattern(request: QueryRequest, settings: Settings | None = None) -> QueryResult:
    if request.pattern == "semantic":
        from building_with_rag.retrieval import semantic_search  # lazy: no connections at startup

        settings = settings or get_settings()
        result = semantic_search(request, settings)
        if request.generate_answer:
            from building_with_rag.generation import add_generation

            result = add_generation(request.question, result, settings)
        return result
    return placeholder(request)
