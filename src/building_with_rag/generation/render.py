"""Turn one generation operation's events into the plain-Markdown text Open WebUI shows.

Open WebUI cannot retract text, so every unvalidated attempt is labelled as a DRAFT and only the
final event can promote the text (confidence high) or demote it (low confidence).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..models import GenerationResult
from .answer import MAX_ATTEMPTS
from .events import AttemptStarted, Event, Final, Retrying, Text

DRAFT_PREFIX = "DRAFT — checking evidence\n\n"
LOW_CONFIDENCE = "DRAFT — low confidence, not the final answer"


def _unavailable_reason(gen: GenerationResult) -> str:
    if gen.issues:
        return str(gen.issues[-1].get("detail", "provider failed"))
    return str((gen.trace[0] if gen.trace else {}).get("reason") or "provider failed")


def _closing(gen: GenerationResult, streamed: bool) -> Iterator[str]:
    sep = "\n\n---\n" if streamed else ""
    if gen.outcome == "answered":
        yield (f"{sep}**Confidence: high**\n\nChecks passed: citations, support "
               f"(attempt {gen.attempts} of {MAX_ATTEMPTS})\n\n**Sources**\n")
        for c in gen.citations:
            parts = [c["label"], c["section_id"], c.get("heading"), c.get("source"),
                     f"p.{c['page']}" if c.get("page") is not None else None]
            yield "\n- " + " · ".join(str(p) for p in parts if p)
    elif gen.outcome == "insufficient_evidence":
        reason = (gen.trace[0] if gen.trace else {}).get("reason")
        yield (f"{sep}The retrieved passages do not support an answer."
               + (f" Reason: {reason}" if reason else ""))
    elif gen.outcome == "unavailable":
        tail = " The text above is an incomplete DRAFT, not a final answer." if streamed else ""
        yield f"{sep}⚠ Answer generation unavailable: {_unavailable_reason(gen)}.{tail}"
    elif gen.confidence == "low":
        yield f"{sep}**{LOW_CONFIDENCE}**\n\n{gen.low_confidence_reason or 'Evidence checks failed.'}"
        for issue in [i for i in gen.issues if i.get("attempt") == gen.attempts][:5]:
            yield f"\n- {issue['detail']}"
    else:
        yield f"{sep}No usable answer text was produced; no answer is returned."


def render(events: Iterable[Event]) -> Iterator[str]:
    need_prefix, streamed = False, False
    for event in events:
        if isinstance(event, AttemptStarted):
            need_prefix = True
        elif isinstance(event, Text):
            if need_prefix:
                yield DRAFT_PREFIX
                need_prefix = False
            streamed = True
            yield event.text
        elif isinstance(event, Retrying):
            shown = "; ".join(d[:160] for d in event.details[:2])
            yield (f"\n\n*Check failed: {shown} Retrying (attempt {event.attempt} of "
                   f"{event.max_attempts})…*\n\n")
        elif isinstance(event, Final):
            yield from _closing(event.result, streamed)
