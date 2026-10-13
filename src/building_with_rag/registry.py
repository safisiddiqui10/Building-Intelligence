"""Single mode registry and dispatcher. Every mode is a placeholder in the seed."""

from typing import get_args

from building_with_rag.models import Mode, QueryRequest, QueryResult

MODES: tuple[str, ...] = get_args(Mode)
MODEL_IDS: dict[str, str] = {mode: f"rag-{mode}" for mode in MODES}
MODE_BY_MODEL_ID: dict[str, str] = {v: k for k, v in MODEL_IDS.items()}


def run_pattern(request: QueryRequest) -> QueryResult:
    return QueryResult(
        pattern=request.pattern,
        status="not_implemented",
        message=f"The '{request.pattern}' pattern is not implemented yet.",
    )
