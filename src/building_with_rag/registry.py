"""Single mode registry and dispatcher. All six modes are implemented."""

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


def _generation_plan(request: QueryRequest, result: QueryResult) -> tuple[str, str | None] | None:
    """(question for the writer, leading notice) when the result is answerable, else None.

    structured and hyde generate only for `ok` (hyde passes real passages and the original
    question, never the hypothetical text). decomposition also generates for `partial_answer`, but
    only about the evidenced subquestions, with a notice naming the missing ones.
    """
    if request.pattern in ("semantic", "hybrid", "hybrid-reranked"):
        return request.question, None
    if result.status == "ok":
        return request.question, None
    if request.pattern == "decomposition" and result.status == "partial_answer":
        have = [s.subquestion for s in result.subquestions if s.status == "evidenced"]
        lack = [s.subquestion for s in result.subquestions if s.status == "no_evidence"]
        return " ".join(have), ("Partial answer — no corpus evidence found for: "
                                + "; ".join(lack) + "\n\n")
    return None


def start_pattern(request: QueryRequest, settings: Settings | None = None) -> PatternRun:
    """Run the selected pattern once. Generation, if requested, is returned unstarted so the
    caller either drains it (`run_pattern`) or streams it (chat): one retrieval, one operation."""
    if request.pattern in MODES:
        from building_with_rag import retrieval  # lazy: no connections at startup

        settings = settings or get_settings()
        search = {"semantic": retrieval.semantic_search, "hybrid": retrieval.hybrid_search,
                  "hybrid-reranked": retrieval.rerank_search,
                  "structured": retrieval.structured_search,
                  "decomposition": retrieval.decomposition_search,
                  "hyde": retrieval.hyde_search}[request.pattern]
        result = search(request, settings)
        plan = _generation_plan(request, result)
        if request.generate_answer and plan is not None:
            from building_with_rag.generation import generate_events

            question, notice = plan
            events = generate_events(question, result.results, settings)
            if notice:
                from itertools import chain

                from building_with_rag.generation.events import Text

                events = chain([Text(notice)], events)
            return PatternRun(result, events)
        return PatternRun(result)
    return PatternRun(placeholder(request))


def run_pattern(request: QueryRequest, settings: Settings | None = None) -> QueryResult:
    run = start_pattern(request, settings)
    if run.events is None:
        return run.result
    from building_with_rag.generation.answer import final_of, with_generation

    return with_generation(run.result, final_of(run.events))
