"""Context assembly: choose the bounded set of retrieved passages the model may see.

Pure function over `QueryResult.results`. No network; passage text is never altered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import RetrievedChunk

MAX_CONTEXT_PASSAGES = 5
MAX_CONTEXT_CHARS = 8000


@dataclass
class Context:
    outcome: str  # "assembled" | "empty"
    passages: dict[str, RetrievedChunk] = field(default_factory=dict)  # label -> passage
    omitted: list[dict] = field(default_factory=list)

    def trace(self) -> dict:
        return {
            "labels": {label: p.chunk_id for label, p in self.passages.items()},
            "selected": len(self.passages),
            "omitted": self.omitted,
        }


def assemble_context(results: list[RetrievedChunk]) -> Context:
    if not results:
        return Context(outcome="empty")
    ctx = Context(outcome="assembled")
    used = 0
    for passage in results:
        if len(ctx.passages) >= MAX_CONTEXT_PASSAGES:
            ctx.omitted.append({"chunk_id": passage.chunk_id, "reason": "passage_limit"})
        elif used + len(passage.text) > MAX_CONTEXT_CHARS:
            ctx.omitted.append({"chunk_id": passage.chunk_id, "reason": "char_limit"})
        else:
            ctx.passages[f"E{len(ctx.passages) + 1}"] = passage
            used += len(passage.text)
    if not ctx.passages:
        ctx.outcome = "empty"
    return ctx
