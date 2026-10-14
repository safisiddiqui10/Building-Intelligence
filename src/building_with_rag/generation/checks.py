"""Deterministic answer checks: claim parsing, citation check, lexical support check. No LLM."""

from __future__ import annotations

import re

from ..models import RetrievedChunk

SUPPORT_MIN_OVERLAP = 0.6
MIN_WORD_LENGTH = 4
STEM_LENGTH = 5

_LABEL_GROUP = re.compile(r"\[\s*(E\d+(?:\s*,\s*E\d+)*)\s*\]")
# A sentence ends at . ! or ? (plus any trailing [E..] labels) when followed by a capital letter,
# a quote, a bracket or the end of the line. "Sec. 103" is therefore not split.
_SENTENCE = re.compile(r".+?[.!?](?:\s*\[[^\]]*\])*(?=\s+[A-Z\"“(]|\s*$)|.+$")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_STOP = frozenset(
    ["that", "this", "with", "from", "shall", "also", "have", "been", "were", "will", "would", "which", "their", "there", "them", "they", "then", "than", "into", "upon", "under", "over", "such", "some", "what", "when", "where", "while", "whoever", "section", "sections", "passage", "passages", "provided", "states", "state", "stated", "says", "according", "answer", "only", "other", "more", "most", "any", "each", "both", "does", "done", "said"]
)


def split_claims(text: str) -> list[dict]:
    """One claim per sentence: {"text", "evidence": [labels cited in that sentence]}."""
    claims = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        for match in _SENTENCE.finditer(line):
            sentence = match.group().strip()
            if not sentence:
                continue
            groups = _LABEL_GROUP.findall(sentence)
            labels = list(dict.fromkeys(re.findall(r"E\d+", " ".join(groups))))
            claims.append({"text": sentence, "evidence": labels})
    return claims


def _clip(text: str, n: int = 80) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def check_citations(claims: list[dict], labels: set[str], attempt: int) -> list[dict]:
    if not claims:
        return [{"attempt": attempt, "check": "citation", "detail": "The answer has no statements to check."}]
    issues = []
    for i, claim in enumerate(claims, 1):
        unknown = [e for e in claim["evidence"] if e not in labels]
        if not claim["evidence"]:
            detail = f"Sentence {i} cites no passage: \"{_clip(claim['text'])}\""
        elif unknown:
            detail = f"Sentence {i} cites {', '.join(unknown)}, which was not supplied."
        else:
            continue
        issues.append({"attempt": attempt, "check": "citation", "detail": detail,
                       "claim": claim["text"], "labels": list(claim["evidence"])})
    return issues


def _stems(text: str) -> set[str]:
    return {w[:STEM_LENGTH] for w in re.findall(rf"[a-z]{{{MIN_WORD_LENGTH},}}", text.lower())
            if w not in _STOP}


def check_support(claims: list[dict], passages: dict[str, RetrievedChunk], attempt: int) -> list[dict]:
    """Lexical check: numbers must appear in the cited passages and enough key words must overlap.

    It catches fabricated numbers and off-topic statements, not subtle misreadings.
    """
    issues = []
    for i, claim in enumerate(claims, 1):
        cited = [e for e in claim["evidence"] if e in passages]
        base = {"attempt": attempt, "check": "support", "claim": claim["text"],
                "labels": list(claim["evidence"])}
        if not cited:
            issues.append({**base, "detail": f"Sentence {i} has no supplied passage to support it."})
            continue
        evidence = " ".join(
            f"{passages[e].text} {passages[e].heading or ''} {passages[e].source or ''} "
            f"{passages[e].section_id}" for e in cited)
        plain = _LABEL_GROUP.sub(" ", claim["text"])
        missing = [n for n in dict.fromkeys(re.findall(r"\d+", plain))
                   if n not in set(re.findall(r"\d+", evidence))]
        if missing:
            issues.append({**base, "detail": f"Sentence {i} mentions {', '.join(missing)}, which does "
                           f"not appear in {', '.join(cited)}."})
            continue
        words = _stems(plain)
        if words:
            overlap = len(words & _stems(evidence)) / len(words)
            if overlap < SUPPORT_MIN_OVERLAP:
                issues.append({**base, "detail": f"Sentence {i} shares only {overlap:.0%} of its key "
                               f"words with {', '.join(cited)} (needs {SUPPORT_MIN_OVERLAP:.0%})."})
    return issues
