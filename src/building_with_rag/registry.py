"""Single mode registry and dispatcher. `semantic`, `hybrid`, `hybrid-reranked` and `structured` are implemented so far."""

from collections.abc import Iterator
from dataclasses import dataclass
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


@dataclass
class PatternRun:
    """A pattern's retrieval result plus, when generation was requested, its one event iterator."""

    result: QueryResult
    events: Iterator | None = None


def start_pattern(request: QueryRequest, settings: Settings | None = None) -> PatternRun:
    """Run the selected pattern once. Generation, if requested, is returned unstarted so the
    caller either drains it (`run_pattern`) or streams it (chat): one retrieval, one operation."""
    if request.pattern in ("semantic", "hybrid", "hybrid-reranked", "structured"):
        from building_with_rag import retrieval  # lazy: no connections at startup

        settings = settings or get_settings()
        search = {"semantic": retrieval.semantic_search, "hybrid": retrieval.hybrid_search,
                  "hybrid-reranked": retrieval.rerank_search,
                  "structured": retrieval.structured_search}[request.pattern]
        result = search(request, settings)
        # structured: only an exact `ok` section is evidence; other statuses never call the LLM.
        if request.generate_answer and (request.pattern != "structured" or result.status == "ok"):
            from building_with_rag.generation import generate_events

            return PatternRun(result, generate_events(request.question, result.results, settings))
        return PatternRun(result)
    return PatternRun(placeholder(request))


def run_pattern(request: QueryRequest, settings: Settings | None = None) -> QueryResult:
    run = start_pattern(request, settings)
    if run.events is None:
        return run.result
    from building_with_rag.generation.answer import final_of, with_generation

    return with_generation(run.result, final_of(run.events))
