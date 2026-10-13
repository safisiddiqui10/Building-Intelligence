"""Diagnostic command. Prints counts, status and short fields only; never vectors or full text.

  uv run python -m building_with_rag.retrieval --check
  uv run python -m building_with_rag.retrieval "question" [--limit N] [--act A] [--status S]
  uv run python -m building_with_rag.retrieval "question" --inspect
"""

from __future__ import annotations

import argparse
import json
import sys

from pymongo.errors import PyMongoError

from ..config import get_settings
from ..ingestion.mongodb_schema import (
    CHUNKS,
    EMBEDDINGS,
    SECTIONS,
    SOURCES,
    VECTOR_FIELD,
    embedding_id,
    text_sha256,
)
from ..ingestion.parser import ACTS, OUT_DIR
from ..models import QueryRequest, SemanticFilters
from .semantic import RetrievalUnavailable, get_db, index_status, semantic_search


def check() -> int:
    s = get_settings()
    print(f"MONGODB_URI: {'set' if s.mongodb_uri else 'missing'}")
    print(f"VOYAGE_API_KEY: {'set' if s.voyage_api_key else 'missing'}")
    db = get_db(s)
    counts = {n: db[n].count_documents({}) for n in (SOURCES, SECTIONS, CHUNKS, EMBEDDINGS)}
    print("counts:", " ".join(f"{k}={v}" for k, v in counts.items()))
    have = set(db[EMBEDDINGS].distinct("embedding_id"))
    missing = sum(
        1 for c in db[CHUNKS].find({}, {"chunk_id": 1, "embedding_input_text": 1})
        if embedding_id(c["chunk_id"], s.embedding_model, s.embedding_model_version,
                        s.embedding_dimensions, c["embedding_input_text"]) not in have)
    print(f"chunks without current embedding: {missing}")
    other = db[EMBEDDINGS].count_documents({"$or": [
        {"embedding_model": {"$ne": s.embedding_model}},
        {"embedding_model_version": {"$ne": s.embedding_model_version}},
        {"dimensions": {"$ne": s.embedding_dimensions}}]})
    print(f"embeddings from other model settings: {other}")
    sample = db[EMBEDDINGS].find_one({})
    if sample:
        print(f"sample vector length={len(sample[VECTOR_FIELD])} "
              f"model={sample['embedding_model']} version={sample['embedding_model_version']} "
              f"dimensions={sample['dimensions']}")
    queryable, note = index_status(db)
    print(f"vector index: {note}")
    ready = (counts[EMBEDDINGS] == counts[CHUNKS] > 0 and missing == 0 and other == 0
             and queryable and sample is not None
             and len(sample[VECTOR_FIELD]) == s.embedding_dimensions)
    print("readiness:", "ready" if ready else "NOT ready")
    return 0 if ready else 1


def inspect(db, top) -> None:
    sec = db[SECTIONS].find_one({"_id": top.section_id})
    chunk = db[CHUNKS].find_one({"_id": top.chunk_id})
    jsonl_sha = None
    for cfg in ACTS.values():
        path = OUT_DIR / cfg["out"]
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec["section_id"] == top.section_id:
                        jsonl_sha = text_sha256(rec["text"])
    print(f"inspect: section_id={sec['section_id']} text_sha256={sec['text_sha256'][:12]} "
          f"matches_jsonl={jsonl_sha == sec['text_sha256']}")
    print(f"chunk_in_section_text={chunk['text'] in sec['text']} "
          f"page_start={sec.get('page_start')} page_end={sec.get('page_end')} "
          f"source_file={sec.get('source_pdf')}")
    print(f"excerpt: {chunk['text'][:160]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--act", action="append", default=[])
    ap.add_argument("--status", action="append", default=[])
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    try:
        if a.check:
            return check()
        if not a.question:
            ap.error("a question or --check is required")
        s = get_settings()
        req = QueryRequest(question=a.question, pattern="semantic", limit=a.limit,
                           caller_id=s.webui_demo_caller_id,
                           filters=SemanticFilters(act=a.act, status=a.status))
        res = semantic_search(req, s)
        t = res.trace[0]
        print(f"mode={res.pattern} query={t['query']!r} status={res.status}")
        print("trace:", {k: t[k] for k in ("limit", "num_candidates", "filters_applied", "hits",
                                           "dropped_model_mismatch", "dropped_unresolved",
                                           "returned")})
        for i, r in enumerate(res.results, 1):
            print(f"{i}. score={r.score:.4f} chunk_id={r.chunk_id} section_id={r.section_id} "
                  f"heading={r.heading!r} source_file={r.source_file} page={r.page} "
                  f"text_len={len(r.text)}")
        if a.inspect and res.results:
            inspect(get_db(s), res.results[0])
        return 0
    except RetrievalUnavailable as exc:
        print(f"Service/configuration problem: {exc}")
        return 2
    except PyMongoError as exc:
        print(f"MongoDB error ({type(exc).__name__}).")
        return 2


sys.exit(main())
