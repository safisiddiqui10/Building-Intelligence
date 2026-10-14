"""Hybrid retrieval: semantic ($vectorSearch) and keyword ($search) routes, fused by RRF.

Both routes search the same stored chunks with the same server-built filter. Fusion is
reciprocal rank fusion computed here, so the rule is visible and the same on every Atlas tier.
The fused score is a rank-based combination, not proof that a passage is correct.
"""

from __future__ import annotations

from pymongo.errors import PyMongoError

from ..config import Settings
from ..ingestion.mongodb_schema import CHUNKS, KEYWORD_FIELD, KEYWORD_INDEX_NAME
from ..models import QueryRequest, QueryResult, RetrievedChunk
from .semantic import (
    QUERY_INPUT_TYPE,
    RetrievalUnavailable,
    build_filter,
    check_quick,
    embed_query,
    get_db,
    load_evidence,
    num_candidates,
    to_retrieved,
    vector_search,
)

HYBRID_ROUTE_CANDIDATES = 20
RRF_K = 60
ROUTE_WEIGHTS = {"semantic": 1.0, "keyword": 1.0}
FILTER_PATHS = ("access_level", "act", "status")


def keyword_index_status(db) -> tuple[bool, str]:
    found = list(db[CHUNKS].list_search_indexes(KEYWORD_INDEX_NAME))
    if not found:
        return False, f"index {KEYWORD_INDEX_NAME} not found"
    if found[0].get("type") != "search":
        return False, f"index {KEYWORD_INDEX_NAME} is not a search index"
    if not found[0].get("queryable"):
        return False, f"index {KEYWORD_INDEX_NAME} is not queryable yet"
    return True, "queryable"


def check_keyword(db) -> None:
    try:
        ok, note = keyword_index_status(db)
    except PyMongoError as exc:
        raise RetrievalUnavailable(f"MongoDB error ({type(exc).__name__}).") from None
    if not ok:
        raise RetrievalUnavailable(
            f"Keyword {note}; run the Story 4.1 index step (load_mongodb --skip-embed).")


def keyword_search(db, question: str, clauses: dict, limit: int) -> list[dict]:
    """One $search on chunks.text with the same filter. Returns [{chunk_id, score}] best first."""
    filters = [{"in": {"path": p, "value": clauses[p]["$in"]}}
               for p in FILTER_PATHS if p in clauses]
    try:
        hits = list(db[CHUNKS].aggregate([
            {"$search": {"index": KEYWORD_INDEX_NAME, "compound": {
                "must": [{"text": {"query": question, "path": KEYWORD_FIELD}}],
                "filter": filters}}},
            {"$limit": limit},
            {"$project": {"chunk_id": 1, "score": {"$meta": "searchScore"}}},
        ]))
    except PyMongoError as exc:
        raise RetrievalUnavailable(f"Keyword search failed ({type(exc).__name__}).") from None
    return sorted(hits, key=lambda h: h["score"], reverse=True)


def fuse(semantic: list[dict], keyword: list[dict]) -> list[dict]:
    """Reciprocal rank fusion. Inputs are best-first lists of {chunk_id, score}.

    fused_score = sum over routes of weight / (RRF_K + rank). Ties: better single-route rank,
    then chunk_id. Returns rows in fused order with 1-based `fused_rank`.
    """
    rows: dict[str, dict] = {}
    for route, hits in (("semantic", semantic), ("keyword", keyword)):
        for rank, h in enumerate(hits, 1):
            row = rows.setdefault(h["chunk_id"], {
                "chunk_id": h["chunk_id"], "fused_score": 0.0,
                "semantic_rank": None, "semantic_score": None,
                "keyword_rank": None, "keyword_score": None})
            row[f"{route}_rank"], row[f"{route}_score"] = rank, h["score"]
            row["fused_score"] += ROUTE_WEIGHTS[route] / (RRF_K + rank)

    def best_rank(r: dict) -> int:
        return min(x for x in (r["semantic_rank"], r["keyword_rank"]) if x is not None)

    ordered = sorted(rows.values(),
                     key=lambda r: (-r["fused_score"], best_rank(r), r["chunk_id"]))
    for i, r in enumerate(ordered, 1):
        r["fused_rank"] = i
    return ordered


def _routes(row: dict) -> list[str]:
    return [r for r in ("semantic", "keyword") if row[f"{r}_rank"] is not None]


def hybrid_candidates(request: QueryRequest, settings: Settings,
                      keep: int) -> tuple[list[RetrievedChunk], dict]:
    """Both routes, fused and resolved. Returns every resolved candidate in fused order plus
    the trace; trace rows mark the first `keep` as `returned`. Callers decide what to cut."""
    db = get_db(settings)
    check_quick(db)
    check_keyword(db)
    clauses, applied = build_filter(request)
    pool = max(HYBRID_ROUTE_CANDIDATES, request.limit)
    trace = {
        "step": "hybrid_search", "mode": "hybrid", "query": request.question,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "input_type": QUERY_INPUT_TYPE, "limit": request.limit, "filters_applied": applied,
        "keyword_index": KEYWORD_INDEX_NAME, "keyword_field": KEYWORD_FIELD,
        "fusion": {"rule": "reciprocal_rank_fusion", "rrf_k": RRF_K,
                   "weights": ROUTE_WEIGHTS},
        "route_candidates": {"semantic": pool, "keyword": pool,
                             "num_candidates": num_candidates(pool)},
        "route_meaning": "routes lists which retrieval route returned a candidate; a null rank "
                         "means that route did not return it.",
        "semantic_hits": 0, "keyword_hits": 0, "in_both_routes": 0, "semantic_only": 0,
        "keyword_only": 0, "dropped_model_mismatch": 0, "dropped_unresolved": 0,
        "fused_total": 0, "returned": 0, "candidates": [],
        "ignored_fields": [n for n, used in (
            ("required_acts", request.required_acts), ("chapter", request.chapter),
            ("generate_answer", request.generate_answer)) if used],
    }
    if not applied["access_level"]:
        return [], trace

    vector = embed_query(request.question, settings)
    try:
        good, total = vector_search(db, settings, vector, clauses, pool, num_candidates(pool))
    except PyMongoError as exc:
        raise RetrievalUnavailable(f"MongoDB search failed ({type(exc).__name__}).") from None
    trace["dropped_model_mismatch"] = total - len(good)
    semantic = sorted(good, key=lambda h: h["score"], reverse=True)
    keyword = keyword_search(db, request.question, clauses, pool)
    trace["semantic_hits"], trace["keyword_hits"] = len(semantic), len(keyword)

    fused = fuse(semantic, keyword)
    try:
        chunks, sections = load_evidence(db, [r["chunk_id"] for r in fused])
    except PyMongoError as exc:
        raise RetrievalUnavailable(f"MongoDB search failed ({type(exc).__name__}).") from None

    resolved = []
    for r in fused:
        chunk = chunks.get(r["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if chunk and section:
            resolved.append((r, chunk, section))
        else:
            trace["dropped_unresolved"] += 1
    results: list[RetrievedChunk] = []
    for i, (r, chunk, section) in enumerate(resolved, 1):
        r["fused_rank"] = i
        trace["candidates"].append({
            "chunk_id": chunk["chunk_id"], "section_id": section["section_id"],
            "routes": _routes(r), "semantic_rank": r["semantic_rank"],
            "semantic_score": r["semantic_score"], "keyword_rank": r["keyword_rank"],
            "keyword_score": r["keyword_score"], "fused_rank": i,
            "fused_score": r["fused_score"], "returned": i <= keep})
        results.append(to_retrieved(
            chunk, section, r["fused_score"], semantic_score=r["semantic_score"],
            semantic_rank=r["semantic_rank"], keyword_score=r["keyword_score"],
            keyword_rank=r["keyword_rank"], fused_score=r["fused_score"], fused_rank=i))
    both = sum(1 for c in trace["candidates"] if len(c["routes"]) == 2)
    trace["in_both_routes"] = both
    trace["semantic_only"] = sum(1 for c in trace["candidates"] if c["routes"] == ["semantic"])
    trace["keyword_only"] = sum(1 for c in trace["candidates"] if c["routes"] == ["keyword"])
    trace["fused_total"] = len(resolved)
    return results, trace


def hybrid_search(request: QueryRequest, settings: Settings) -> QueryResult:
    candidates, trace = hybrid_candidates(request, settings, request.limit)
    return _result(candidates[:request.limit], trace)


def _result(results: list[RetrievedChunk], trace: dict) -> QueryResult:
    trace["returned"] = len(results)
    if results:
        status = "ok"
        message = (f"{len(results)} passages retrieved by semantic similarity and keyword match, "
                   "fused by reciprocal rank; scores are not proof of correctness.")
    else:
        status = "no_results"
        message = "Both searches ran and found no passages within the requested filters."
    return QueryResult(pattern="hybrid", status=status, message=message, trace=[trace],
                       results=results)
