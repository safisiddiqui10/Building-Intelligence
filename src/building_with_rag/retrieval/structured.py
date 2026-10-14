"""Structured exact retrieval: a rule-based classifier and one read-only lookup on `sections`.

No LLM, no network. The MongoDB predicate is built only from validated signals and fixed
constants, never from question text. A stored section is source text, not a statement of
current legal applicability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pymongo.errors import PyMongoError

from ..config import Settings
from ..ingestion.mongodb_schema import ACCESS_LEVEL, SECTIONS
from ..models import QueryRequest, QueryResult, RetrievedChunk, StructuredSignals
from .semantic import RetrievalUnavailable, get_db

ACT_STORED = {"BNS": "BNS_2023", "IPC": "IPC_1860"}
ACT_STATUS = {"BNS": "in_force", "IPC": "repealed"}  # selects the act's stored status; not a currency claim
SECTION_CHUNK_SUFFIX = "#section"
MAX_FETCH = 5
PROJECTION = {"section_id": 1, "act": 1, "act_label": 1, "status": 1, "chapter": 1,
              "section_number": 1, "heading": 1, "text": 1, "source_pdf": 1,
              "source_status_version": 1, "page_start": 1, "access_level": 1}
RECOMMENDATION = ("This mode only looks up one named section, for example 'BNS section 103'. "
                  "For other questions choose rag-hybrid or rag-semantic.")

_ACTS = {"BNS": re.compile(r"\b(?:bns|bharatiya\s+nyaya\s+sanhita)\b", re.IGNORECASE),
         "IPC": re.compile(r"\b(?:ipc|indian\s+penal\s+code)\b", re.IGNORECASE)}
_NUM = r"(\d{1,4})([A-Za-z]?)\b"
_SECTION_NUM = re.compile(r"(?:\bsections?|\bsec\.?|§)\s*" + _NUM, re.IGNORECASE)
_ACT_NUM = re.compile(r"\b(?:bns|ipc)[\s\-:]*" + _NUM, re.IGNORECASE)
_MORE_NUM = re.compile(r"(?:\band\b|\bor\b|\bto\b|[,&])\s*" + _NUM, re.IGNORECASE)
_CHAPTER = re.compile(r"\bchapter\s+([IVXLC]+A?|\d{1,2})\b", re.IGNORECASE)
_ROMAN = re.compile(r"^[IVXLC]+A?$")
_AGGREGATION = re.compile(r"\b(?:how\s+many|count|total\s+number|number\s+of\s+sections)\b", re.IGNORECASE)
_FILTER = re.compile(r"\b(?:list|enumerate|which\s+sections|all\s+sections|every\s+section)\b", re.IGNORECASE)


@dataclass
class Classification:
    status: str  # "ok" | "clarification_needed" | "unsupported"
    reason: str  # fixed reason code
    message: str
    signals: StructuredSignals | None = None
    suffix: str = ""
    ignored: list[str] = field(default_factory=list)


def _roman(n: int) -> str:
    out = ""
    for value, sym in ((90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"),
                       (4, "IV"), (1, "I")):
        while n >= value:
            out, n = out + sym, n - value
    return out


def normalise_chapter(raw: str | None) -> str | None:
    """Stored chapters are upper-case Roman numerals (`IX`, `VA`). Anything else is rejected."""
    if raw is None:
        return None
    raw = raw.strip().upper()
    if raw.isdigit() and 1 <= int(raw) <= 99:
        raw = _roman(int(raw))
    return raw if _ROMAN.match(raw) else None


def classify(question: str) -> Classification:
    """Turn the question into StructuredSignals, or a clarification / recommendation."""
    acts = [a for a, rx in _ACTS.items() if rx.search(question)]
    numbers = {(m.group(1), m.group(2).upper())
               for rx in (_SECTION_NUM, _ACT_NUM) for m in rx.finditer(question)}
    if numbers:  # "sections 103 and 104": further bare numbers make the request ambiguous
        numbers |= {(m.group(1), m.group(2).upper()) for m in _MORE_NUM.finditer(question)}
    chapters = {c for c in (normalise_chapter(m.group(1)) for m in _CHAPTER.finditer(question)) if c}
    chapter = next(iter(chapters)) if len(chapters) == 1 else None
    act = acts[0] if len(acts) == 1 else None

    if _AGGREGATION.search(question):
        return Classification("unsupported", "unsupported_aggregation", RECOMMENDATION,
                              StructuredSignals(intent="aggregation", act=act, chapter=chapter))
    if _FILTER.search(question):
        return Classification("unsupported", "unsupported_filter", RECOMMENDATION,
                              StructuredSignals(intent="filter", act=act, chapter=chapter))
    if not numbers:
        return Classification("unsupported", "no_supported_intent", RECOMMENDATION)
    if len(acts) > 1:
        return Classification("clarification_needed", "multiple_acts",
                              "Name one act (BNS or IPC) and one section per request.")
    if len(numbers) > 1:
        return Classification("clarification_needed", "multiple_sections",
                              "Name one section per request.",
                              StructuredSignals(intent="exact_lookup", act=act))
    if act is None:
        return Classification("clarification_needed", "missing_act",
                              "Which act do you mean, BNS or IPC?")
    digits, suffix = next(iter(numbers))
    return Classification(
        "ok", "exact_lookup", "Exact section lookup.",
        StructuredSignals(intent="exact_lookup", act=act, section_number=digits + suffix,
                          chapter=chapter),
        suffix=suffix)


def _predicate(signals: StructuredSignals, chapter: str | None) -> dict:
    pred = {"act": ACT_STORED[signals.act],
            "section_number": int(signals.section_number.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
            "status": ACT_STATUS[signals.act], "access_level": ACCESS_LEVEL}
    if chapter:
        pred["chapter"] = chapter
    return pred


def _to_chunk(section: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{section['section_id']}{SECTION_CHUNK_SUFFIX}", section_id=section["section_id"],
        act=section["act"], text=section["text"], heading=section.get("heading"), score=None,
        source=section.get("act_label"), source_file=section.get("source_pdf"),
        page=section.get("page_start"), chapter=section.get("chapter"),
        status=section.get("status"), access_level=section.get("access_level"),
        source_status_version=section.get("source_status_version"))


def _result(status: str, message: str, trace: dict,
            results: list[RetrievedChunk] | None = None) -> QueryResult:
    trace["returned"] = len(results or [])
    return QueryResult(pattern="structured", status=status, message=message, trace=[trace],
                       results=results or [])


def structured_search(request: QueryRequest, settings: Settings) -> QueryResult:
    cls = classify(request.question)
    trace: dict = {
        "step": "structured_lookup", "mode": "structured",
        "intent": cls.signals.intent if cls.signals else None,
        "signals": cls.signals.model_dump() if cls.signals else None,
        "classifier_reason": cls.reason, "mongodb_called": False, "predicate": None,
        "matched": 0, "candidate_section_ids": [], "returned": 0,
        "ignored_fields": [n for n, used in (("required_acts", request.required_acts),) if used],
    }
    if cls.status != "ok":
        return _result(cls.status, cls.message, trace)

    signals = cls.signals
    req_chapter = normalise_chapter(request.chapter)
    if request.chapter and req_chapter is None:
        trace["classifier_reason"] = "invalid_chapter"
        return _result("clarification_needed", "The chapter is not a valid chapter numeral.", trace)
    if req_chapter and signals.chapter and req_chapter != signals.chapter:
        trace["classifier_reason"] = "chapter_conflict"
        return _result("clarification_needed",
                       "The question and the request name different chapters.", trace)
    chapter = signals.chapter or req_chapter
    if chapter and not signals.chapter:
        signals = signals.model_copy(update={"chapter": chapter})
        trace["signals"] = signals.model_dump()

    pred = _predicate(signals, chapter)
    trace["predicate"] = pred
    f = request.filters
    if f and ((f.act and pred["act"] not in f.act) or (f.status and pred["status"] not in f.status)
              or (f.access_level and ACCESS_LEVEL not in f.access_level)):
        trace["classifier_reason"] = "excluded_by_filters"
        return _result("not_found", "The requested section is outside the requested filters.",
                       trace)

    db = get_db(settings)
    try:
        found = list(db[SECTIONS].find(pred, PROJECTION).limit(MAX_FETCH))
    except PyMongoError as exc:
        raise RetrievalUnavailable(f"MongoDB lookup failed ({type(exc).__name__}).") from None
    trace["mongodb_called"] = True
    label = f"{signals.act.lower()}:{signals.section_number.upper()}"
    if cls.suffix:
        found = [s for s in found if s["section_id"].upper() == label.upper()]
    trace["matched"] = len(found)
    if not found:
        return _result("not_found", f"No stored section matches {signals.act} "
                       f"section {signals.section_number}.", trace)
    if len(found) > 1:
        trace["classifier_reason"] = "ambiguous_section_number"
        trace["candidate_section_ids"] = sorted(s["section_id"] for s in found)
        return _result("clarification_needed",
                       f"Several sections share that number: {', '.join(trace['candidate_section_ids'])}. "
                       "Name one.", trace)
    section = found[0]
    trace["source_status_version"] = section.get("source_status_version")
    return _result("ok", f"Section {section['section_id']} retrieved by exact lookup; stored "
                   "status and source version are source metadata, not current legal applicability.",
                   trace, [_to_chunk(section)])
