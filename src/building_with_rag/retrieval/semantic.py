"""Semantic retrieval: embed the question, run $vectorSearch, resolve chunks and sections.

Scores are vector similarity, not proof that a passage is correct or answers the question.
"""

from __future__ import annotations

from functools import lru_cache

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from ..config import Settings
from ..ingestion.mongodb_schema import (
    ACCESS_LEVEL,
    CHUNKS,
    EMBEDDINGS,
    SECTIONS,
    VECTOR_FIELD,
    VECTOR_INDEX_NAME,
)
from ..models import QueryRequest, QueryResult, RetrievedChunk

QUERY_INPUT_TYPE = "query"
CANDIDATES_PER_RESULT = 10
MIN_CANDIDATES = 100
MAX_CANDIDATES = 200
VOYAGE_TIMEOUT_SECONDS = 20
MONGO_TIMEOUT_MS = 8000


class RetrievalUnavailable(Exception):
    """Service or configuration problem. The message is safe to return (no secrets)."""


def num_candidates(limit: int) -> int:
    return min(max(limit * CANDIDATES_PER_RESULT, MIN_CANDIDATES), MAX_CANDIDATES)


@lru_cache
def _client(uri: str) -> MongoClient:
    return MongoClient(uri, serverSelectionTimeoutMS=MONGO_TIMEOUT_MS)


def get_db(settings: Settings):
    if not settings.mongodb_uri:
        raise RetrievalUnavailable("MONGODB_URI is not set.")
    return _client(settings.mongodb_uri)[settings.mongodb_db_name]


def index_status(db) -> tuple[bool, str]:
    """Return (queryable, note) for the vector index."""
    found = list(db[EMBEDDINGS].list_search_indexes(VECTOR_INDEX_NAME))
    if not found:
        return False, f"index {VECTOR_INDEX_NAME} not found"
    if found[0].get("type") != "vectorSearch":
        return False, f"index {VECTOR_INDEX_NAME} is not a vectorSearch index"
    if not found[0].get("queryable"):
        return False, f"index {VECTOR_INDEX_NAME} is not queryable yet"
    return True, "queryable"


def check_quick(db) -> None:
    try:
        ok, note = index_status(db)
        if not ok:
            raise RetrievalUnavailable(f"Vector {note}.")
        if db[EMBEDDINGS].estimated_document_count() == 0:
            raise RetrievalUnavailable("No embeddings are stored; run the Story 2.2 load.")
    except PyMongoError as exc:
        raise RetrievalUnavailable(f"MongoDB error ({type(exc).__name__}).") from None


def embed_query(question: str, settings: Settings) -> list[float]:
    if not settings.voyage_api_key:
        raise RetrievalUnavailable("VOYAGE_API_KEY is not set.")
    import voyageai

    try:
        client = voyageai.Client(api_key=settings.voyage_api_key, max_retries=0,
                                 timeout=VOYAGE_TIMEOUT_SECONDS)
        res = client.embed([question], model=settings.embedding_model,
                           input_type=QUERY_INPUT_TYPE,
                           output_dimension=settings.embedding_dimensions)
    except Exception as exc:  # noqa: BLE001 - provider errors vary; never leak details
        raise RetrievalUnavailable(f"Query embedding failed ({type(exc).__name__}).") from None
    vector = [float(x) for x in res.embeddings[0]]
    if len(vector) != settings.embedding_dimensions:
        raise RetrievalUnavailable(
            f"Query embedding has {len(vector)} dimensions, expected "
            f"{settings.embedding_dimensions}.")
    return vector


def build_filter(request: QueryRequest) -> tuple[dict, dict]:
    """Server-built filter. Caller lists only narrow the fixed public scope."""
    f = request.filters
    access = [ACCESS_LEVEL]
    if f and f.access_level:
        access = [a for a in access if a in f.access_level]
    applied = {"access_level": access, "act": list(f.act) if f else [],
               "status": list(f.status) if f else []}
    clauses: dict = {"access_level": {"$in": access}}
    for key in ("act", "status"):
        if applied[key]:
            clauses[key] = {"$in": applied[key]}
    return clauses, applied


def semantic_search(request: QueryRequest, settings: Settings) -> QueryResult:
    db = get_db(settings)
    check_quick(db)
    clauses, applied = build_filter(request)
    trace = {
        "step": "semantic_search", "mode": "semantic", "query": request.question,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "input_type": QUERY_INPUT_TYPE, "limit": request.limit,
        "num_candidates": num_candidates(request.limit), "filters_applied": applied,
        "hits": 0, "dropped_model_mismatch": 0, "dropped_unresolved": 0, "returned": 0,
        "ignored_fields": [n for n, used in (
            ("required_acts", request.required_acts), ("chapter", request.chapter),
            ("generate_answer", request.generate_answer)) if used],
    }
    if not applied["access_level"]:
        return _result([], trace)

    vector = embed_query(request.question, settings)
    try:
        good, trace["hits"] = vector_search(db, settings, vector, clauses, request.limit,
                                            trace["num_candidates"])
        trace["dropped_model_mismatch"] = trace["hits"] - len(good)
        chunks, sections = load_evidence(db, [h["chunk_id"] for h in good])
    except PyMongoError as exc:
        raise RetrievalUnavailable(f"MongoDB search failed ({type(exc).__name__}).") from None

    results: list[RetrievedChunk] = []
    for h in sorted(good, key=lambda h: h["score"], reverse=True):
        chunk = chunks.get(h["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if not chunk or not section:
            trace["dropped_unresolved"] += 1
            continue
        results.append(to_retrieved(chunk, section, h["score"]))
    return _result(results[:request.limit], trace)


def vector_search(db, settings: Settings, vector: list[float], clauses: dict, limit: int,
                  n_candidates: int) -> tuple[list[dict], int]:
    """One $vectorSearch. Returns (hits from the fixed embedding model, total hits)."""
    hits = list(db[EMBEDDINGS].aggregate([
        {"$vectorSearch": {"index": VECTOR_INDEX_NAME, "path": VECTOR_FIELD,
                           "queryVector": vector, "numCandidates": n_candidates,
                           "limit": limit, "filter": clauses}},
        {"$project": {"chunk_id": 1, "embedding_model": 1, "embedding_model_version": 1,
                      "dimensions": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]))
    good = [h for h in hits
            if h.get("embedding_model") == settings.embedding_model
            and h.get("embedding_model_version") == settings.embedding_model_version
            and h.get("dimensions") == settings.embedding_dimensions]
    return good, len(hits)


def load_evidence(db, chunk_ids: list[str]) -> tuple[dict, dict]:
    """Batch-resolve chunk ids to `chunks` and their parent `sections`."""
    chunks = {c["_id"]: c for c in db[CHUNKS].find({"_id": {"$in": chunk_ids}})}
    sections = {s["_id"]: s for s in db[SECTIONS].find(
        {"_id": {"$in": list({c["section_id"] for c in chunks.values()})}})}
    return chunks, sections


def to_retrieved(chunk: dict, section: dict, score: float | None, **extra) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk["chunk_id"], section_id=section["section_id"], act=chunk["act"],
        text=chunk["text"], heading=section.get("heading"), score=score,
        source=section.get("act_label"), source_file=section.get("source_pdf"),
        page=section.get("page_start"), chapter=chunk.get("chapter"),
        status=chunk.get("status"), access_level=chunk.get("access_level"), **extra)


def _result(results: list[RetrievedChunk], trace: dict) -> QueryResult:
    trace["returned"] = len(results)
    if results:
        status = "ok"
        message = (f"{len(results)} passages retrieved by vector similarity; scores are not "
                   "proof of correctness.")
    else:
        status = "no_results"
        message = "The search ran and found no passages within the requested filters."
    return QueryResult(pattern="semantic", status=status, message=message, trace=[trace],
                       results=results)
