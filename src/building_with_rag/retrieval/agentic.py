"""Bounded agentic RAG: a deterministic, read-only controller over three existing adapters.

Runs only when `agentic` is selected explicitly. No LLM planner, no routing from other modes, no
retry, loop, thread or write. The allowlist is closed: structuredLookup, semanticRetrieve and
hydeRetrieve. At most two calls and one HyDE call per request, inside a cooperative deadline.
Real retrieved passages are the only evidence; a HyDE hypothetical text is dropped on return and
never reaches the trace, result, context or answer.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

from ..config import Settings
from ..models import QueryRequest, QueryResult, RetrievedChunk
from .decomposition import MIN_EVIDENCE_SCORE
from .hyde import hyde_search
from .semantic import RetrievalUnavailable, semantic_search
from .structured import classify, structured_search

ACCESS_LEVEL = "public"  # the fixed demo/public scope every adapter already enforces
MAX_CALLS = 2
MAX_HYDE_CALLS = 1
SCORE_FLOOR = MIN_EVIDENCE_SCORE  # reused from Story 5.2; a chosen guard, not correctness

# Closed allowlist: tool name -> existing adapter. Nothing else can be called.
ADAPTERS: dict[str, Callable[[QueryRequest, Settings], QueryResult]] = {
    "structuredLookup": structured_search,
    "semanticRetrieve": semantic_search,
    "hydeRetrieve": hyde_search,
}
NATIVE_MODE = {"structuredLookup": "structured", "semanticRetrieve": "semantic",
               "hydeRetrieve": "hyde"}
DOCUMENTED_STATUS = {"structuredLookup": {"ok", "not_found", "clarification_needed", "unsupported"},
                     "semanticRetrieve": {"ok", "no_results"},
                     "hydeRetrieve": {"ok", "no_results", "hyde_unavailable"}}

# Prepared vocabulary-mismatch rule: everyday words the statute does not use. Hand-chosen from the
# Story 5.2 HyDE question and similar cases; deliberately narrow, not a classifier.
LAY_TERMS = ("thief", "thieves", "stole", "stolen", "steal", "phone", "mobile", "scam",
             "scammer", "cheated", "conned", "robbed")
_LAY_RX = re.compile(r"\b(" + "|".join(LAY_TERMS) + r")\b", re.IGNORECASE)

MESSAGES = {
    "validation_rejected": "The request is unclear. Name an act and a section (for example "
                           "'IPC section 302') or ask a complete question.",
    "record_not_found": "No stored section matched that exact request. Check the act and "
                        "section number.",
    "no_usable_evidence": "No usable corpus passages were found. Try rephrasing the question or "
                          "name a section.",
    "weak_only": "Only weakly related passages were found, so none is used as evidence. Try "
                 "rephrasing, or choose rag-semantic to view the nearest passages.",
    "stopped": "The controller stopped before finding usable evidence (see the trace stop reason).",
    "ready_for_synthesis": "Usable corpus passages were retained for the grounded answer.",
}


class _Stop(Exception):
    def __init__(self, reason: str):
        self.reason = reason


def _in_scope(request: QueryRequest, p: RetrievedChunk) -> bool:
    f = request.filters
    if p.access_level != ACCESS_LEVEL:
        return False
    return not (f and ((f.act and p.act not in f.act) or (f.status and p.status not in f.status)
                       or (f.access_level and p.access_level not in f.access_level)))


def _valid(request: QueryRequest, tool: str, res: QueryResult) -> bool:
    if res.pattern != NATIVE_MODE[tool] or res.status not in DOCUMENTED_STATUS[tool]:
        return False
    if tool == "structuredLookup" and res.status == "ok" and len(res.results) != 1:
        return False
    return all(p.chunk_id and p.section_id and p.text and _in_scope(request, p)
               for p in res.results)


def _validate_request(question: str) -> tuple[str, str | None]:
    """('ok'|'exact'|'rejected', reason). Pure; no network."""
    stripped = question.strip()
    if not re.search(r"[A-Za-z]", stripped):
        return "rejected", "bare_section_number" if stripped.isdigit() else "malformed"
    cls = classify(stripped)
    reasons = {"missing_act": "bare_section_number", "multiple_acts": "ambiguous_request",
               "multiple_sections": "ambiguous_request",
               "unsupported_aggregation": "unsupported_intent",
               "unsupported_filter": "unsupported_intent"}
    if cls.reason in reasons:
        return "rejected", reasons[cls.reason]
    return ("exact" if cls.status == "ok" else "ok"), None


def agentic_search(request: QueryRequest, settings: Settings, *,
                   clock: Callable[[], float] = time.monotonic) -> QueryResult:
    deadline = settings.agentic_deadline_seconds
    if not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or not deadline > 0:
        raise RetrievalUnavailable("AGENTIC_DEADLINE_SECONDS must be greater than 0.")
    start = clock()
    trace: dict = {
        "step": "agentic_controller", "mode": "agentic",
        "validation": {"outcome": "pending", "reason": None}, "path": None,
        "mismatch_rule": {"applied": False, "matches": 0}, "calls": [], "call_count": 0,
        "hyde_calls": 0, "adequacy": [], "stop_reason": None, "deadline_seconds": deadline,
        "elapsed_ms": 0, "returned": 0, "ignored_fields": [],
    }
    retained: list[RetrievedChunk] = []
    ignored: set[str] = set()

    def finish(stop: str, status: str, message: str) -> QueryResult:
        trace["stop_reason"] = stop
        trace["elapsed_ms"] = int((clock() - start) * 1000)
        trace["returned"] = len(retained)
        trace["ignored_fields"] = sorted(ignored)
        return QueryResult(pattern="agentic", status=status, message=message, trace=[trace],
                           results=list(retained))

    def call(tool: str) -> QueryResult:
        """Guard, call once, inspect. Raises _Stop; a returned result is already validated."""
        if tool not in ADAPTERS:
            raise _Stop("tool_invalid")
        if trace["call_count"] >= MAX_CALLS or (tool == "hydeRetrieve"
                                                and trace["hyde_calls"] >= MAX_HYDE_CALLS):
            raise _Stop("call_budget_exhausted")
        if clock() - start >= deadline:
            raise _Stop("deadline_expired")
        trace["call_count"] += 1
        trace["hyde_calls"] += tool == "hydeRetrieve"
        row = {"order": trace["call_count"], "adapter": tool, "outcome": "pending",
               "status": None, "passages": 0, "usable": 0, "retained": [], "weak": []}
        trace["calls"].append(row)
        sub = request.model_copy(update={"generate_answer": False})
        try:
            res = ADAPTERS[tool](sub, settings)
        except Exception:  # noqa: BLE001 - adapter failure is a stop reason, never a body
            row["outcome"] = "unavailable"
            raise _Stop("tool_unavailable") from None
        # Drop what the controller must never keep: the hypothetical text. The adapter's own
        # trace is read only for its ignored-field names.
        res = res.model_copy(update={"hyde_hypothetical_text_debug": None})
        if res.trace and isinstance(res.trace[0], dict):
            ignored.update(res.trace[0].get("ignored_fields") or [])
        if clock() - start >= deadline:
            row["outcome"] = "discarded_after_deadline"
            raise _Stop("deadline_expired")
        if not _valid(request, tool, res):
            row["outcome"] = "invalid"
            raise _Stop("tool_invalid")
        row["outcome"], row["status"], row["passages"] = "ok", res.status, len(res.results)
        return res

    def decide(rule: str, adequate: bool) -> None:
        trace["adequacy"].append({"after_call": trace["call_count"], "rule": rule,
                                  "adequate": adequate})

    try:
        outcome, reason = _validate_request(request.question)
        trace["validation"] = {"outcome": "rejected" if outcome == "rejected" else "accepted",
                               "reason": reason}
        if outcome == "rejected":
            trace["path"] = "validation_rejected"
            return finish("validation_rejected", "clarify", MESSAGES["validation_rejected"])

        if outcome == "exact":
            trace["path"] = "exact"
            res = call("structuredLookup")
            found = res.status == "ok"
            decide("exact_record_found", found)
            if not found:
                return finish("record_not_found", "clarify", MESSAGES["record_not_found"])
            retained.extend(res.results)
            trace["calls"][-1]["usable"] = 1
            trace["calls"][-1]["retained"] = [p.chunk_id for p in res.results]
            return finish("ready_for_synthesis", "ok", MESSAGES["ready_for_synthesis"])

        matches = {m.lower() for m in _LAY_RX.findall(request.question)}
        trace["path"] = "vocabulary_mismatch" if matches else "ordinary"
        trace["mismatch_rule"] = {"applied": bool(matches), "matches": len(matches)}

        res = call("semanticRetrieve")
        usable = [p for p in res.results if p.score is not None and p.score >= SCORE_FLOOR]
        weak = [p for p in res.results if p not in usable]
        row = trace["calls"][-1]
        row["usable"] = len(usable)
        row["retained"] = [p.chunk_id for p in usable]
        row["weak"] = [{"chunk_id": p.chunk_id, "score": p.score} for p in weak]
        decide(f"semantic_score_at_least_{SCORE_FLOOR}", bool(usable))
        if usable:
            retained.extend(usable)
            return finish("ready_for_synthesis", "ok", MESSAGES["ready_for_synthesis"])

        if matches:
            res = call("hydeRetrieve")
            found = res.status == "ok" and bool(res.results)
            row = trace["calls"][-1]
            decide("hyde_ok_with_real_in_scope_passage", found)
            if found:
                retained.extend(res.results)
                row["usable"] = len(retained)
                row["retained"] = [p.chunk_id for p in retained]
                return finish("ready_for_synthesis", "ok", MESSAGES["ready_for_synthesis"])

        if weak:  # real passages came back, but none usable: nearest-only, kept out of results
            return finish("no_usable_evidence", "partial_result", MESSAGES["weak_only"])
        return finish("no_usable_evidence", "clarify", MESSAGES["no_usable_evidence"])
    except _Stop as stop:
        status = "partial_result" if retained else "unavailable"
        return finish(stop.reason, status, MESSAGES["stopped"])
