"""Diagnostic command: runs the real /v1/query path with generation and prints compact JSON.

  uv run python -m building_with_rag.generation "question" [--limit N]
Never prints the key, headers, the full prompt or full passage text.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..config import get_settings
from ..models import QueryRequest
from ..registry import run_pattern
from ..retrieval import RetrievalUnavailable
from .context import assemble_context


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--limit", type=int, default=5)
    a = ap.parse_args()
    s = get_settings()
    req = QueryRequest(question=a.question, pattern="semantic", limit=a.limit,
                       caller_id=s.webui_demo_caller_id, generate_answer=True)
    try:
        res = run_pattern(req, s)
    except RetrievalUnavailable as exc:
        print(f"Retrieval unavailable: {exc}")
        return 2
    gen = res.generation
    ctx = assemble_context(res.results)  # same pure function the generator used
    out = {
        "retrieval_status": res.status, "message": res.message, "retrieved": len(res.results),
        "outcome": gen.outcome, "context_outcome": gen.context_outcome, "answer": gen.answer,
        "claims": gen.claims, "citations": gen.citations,
        "supporting_chunk_ids": [p.chunk_id for p in gen.supporting_passages],
        "context": [{"label": lb, "chunk_id": p.chunk_id, "section_id": p.section_id,
                     "page": p.page, "text_len": len(p.text), "excerpt": p.text[:160]}
                    for lb, p in ctx.passages.items()],
        "trace": gen.trace, "attempts": gen.attempts, "model": gen.model,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
