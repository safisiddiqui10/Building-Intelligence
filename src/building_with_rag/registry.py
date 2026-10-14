"""Single mode registry and dispatcher. Only `semantic` is implemented so far."""

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
    if request.pattern == "semantic":
        from building_with_rag.retrieval import semantic_search  # lazy: no connections at startup

        settings = settings or get_settings()
        result = semantic_search(request, settings)
        if request.generate_answer:
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
