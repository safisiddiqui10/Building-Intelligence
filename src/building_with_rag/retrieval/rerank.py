"""Hybrid-reranked retrieval: hybrid candidates, one re-rank call, a bounded final selection.

The re-rank score is relevance as judged by the provider, not proof that a passage is correct.
Missing configuration or any provider failure is `RetrievalUnavailable`: the result is never
hybrid order presented as re-ranked.
"""

from __future__ import annotations

import math

import httpx

from ..config import Settings
from ..models import OmittedCandidate, QueryRequest, QueryResult, RetrievedChunk
from .hybrid import hybrid_candidates
from .semantic import RetrievalUnavailable

# Test seam: a transport used when none is passed (None means the real network).
DEFAULT_TRANSPORT: httpx.BaseTransport | None = None


def check_config(settings: Settings) -> None:
    """Fail clearly before any retrieval work. Messages name variables, never values."""
    for name, value in (("RERANK_API_KEY", settings.rerank_api_key),
                        ("RERANK_API_BASE_URL", settings.rerank_api_base_url),
                        ("RERANK_MODEL_NAME", settings.rerank_model_name)):
        if not value.strip():
            raise RetrievalUnavailable(f"{name} is not set.")
    cand, send, ret = (settings.rerank_candidate_limit, settings.rerank_send_limit,
                       settings.rerank_return_limit)
    for name, value in (("RERANK_CANDIDATE_LIMIT", cand), ("RERANK_SEND_LIMIT", send),
                        ("RERANK_RETURN_LIMIT", ret)):
        if value < 1:
            raise RetrievalUnavailable(f"{name} must be at least 1.")
    if send > cand:
        raise RetrievalUnavailable("RERANK_SEND_LIMIT must not exceed RERANK_CANDIDATE_LIMIT.")
    if ret > send:
        raise RetrievalUnavailable("RERANK_RETURN_LIMIT must not exceed RERANK_SEND_LIMIT.")
    if not settings.rerank_request_timeout_seconds > 0:
        raise RetrievalUnavailable("RERANK_REQUEST_TIMEOUT_SECONDS must be positive.")


def _fail(reason: str) -> RetrievalUnavailable:
    return RetrievalUnavailable(f"Re-ranking failed ({reason}).")


def call_reranker(question: str, documents: list[str], settings: Settings,
                  transport: httpx.BaseTransport | None = None) -> tuple[list[float], int | None]:
    """One Voyage-style POST {base}/rerank. Returns (score per input index, usage tokens)."""
    url = settings.rerank_api_base_url.rstrip("/") + "/rerank"
    body = {"query": question, "documents": documents, "model": settings.rerank_model_name,
            "top_k": len(documents)}
    headers = {"Authorization": f"Bearer {settings.rerank_api_key}"}
    try:
        with httpx.Client(timeout=settings.rerank_request_timeout_seconds,
                          transport=transport or DEFAULT_TRANSPORT) as client:
            res = client.post(url, json=body, headers=headers)
    except httpx.TimeoutException:
        raise _fail("timed out") from None
    except (httpx.HTTPError, httpx.InvalidURL):
        raise _fail("connection error") from None
    if not res.is_success:
        raise _fail(f"HTTP {res.status_code}")
    try:
        payload = res.json()
        data = payload["data"]
        if not isinstance(data, list) or len(data) != len(documents):
            raise ValueError
        scores: dict[int, float] = {}
        for entry in data:
            index, score = entry["index"], entry["relevance_score"]
            if (isinstance(index, bool) or not isinstance(index, int)
                    or not 0 <= index < len(documents) or index in scores):
                raise ValueError
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score)):
                raise ValueError
            scores[index] = float(score)
        usage = payload.get("usage")
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        if isinstance(tokens, bool) or not isinstance(tokens, int):
            tokens = None
    except (ValueError, KeyError, TypeError, AttributeError):
        raise _fail("unusable reply") from None
    return [scores[i] for i in range(len(documents))], tokens


def rerank_search(request: QueryRequest, settings: Settings) -> QueryResult:
    check_config(settings)
    cand_limit, send_limit = settings.rerank_candidate_limit, settings.rerank_send_limit
    final_limit = min(request.limit, settings.rerank_return_limit)
    candidates, trace = hybrid_candidates(request, settings, cand_limit)
    rows = {c["chunk_id"]: c for c in trace["candidates"]}
    trace.update({"step": "hybrid_rerank", "mode": "hybrid-reranked"})
    rerank = {"model": settings.rerank_model_name, "candidate_limit": cand_limit,
              "send_limit": send_limit, "return_limit": settings.rerank_return_limit,
              "effective_candidates": min(cand_limit, len(candidates)), "sent": 0,
              "returned": 0, "request_limit": request.limit}
    trace["rerank"] = rerank
    trace["reranked_order"] = []
    if not candidates:
        trace["candidates"] = []
        return _result([], [], trace)

    in_set, cut_candidate = candidates[:cand_limit], candidates[cand_limit:]
    sent, cut_send = in_set[:send_limit], in_set[send_limit:]
    scores, tokens = call_reranker(request.question, [c.text for c in sent], settings,
                                   DEFAULT_TRANSPORT)
    if tokens is not None:
        rerank["usage"] = {"total_tokens": tokens}
    order = sorted(range(len(sent)),
                   key=lambda i: (-scores[i], sent[i].fused_rank, sent[i].chunk_id))
    ranked = [(rank, sent[i], scores[i]) for rank, i in enumerate(order, 1)]
    final = [c.model_copy(update={"score": s, "rerank_score": s, "rerank_rank": r})
             for r, c, s in ranked[:final_limit]]
    cut_return = ranked[final_limit:]

    omitted: list[OmittedCandidate] = []
    outcome: dict[str, str] = {}
    for reason, group in (("rerank_candidate_limit", cut_candidate),
                          ("rerank_send_limit", cut_send)):
        for c in group:
            omitted.append(OmittedCandidate(chunk_id=c.chunk_id, omitted_reason=reason))
            outcome[c.chunk_id] = reason
    for _, c, _ in cut_return:
        omitted.append(OmittedCandidate(chunk_id=c.chunk_id,
                                        omitted_reason="rerank_return_limit"))
        outcome[c.chunk_id] = "rerank_return_limit"
    by_id = {c.chunk_id: (r, s) for r, c, s in ranked}
    final_rank = {c.chunk_id: i for i, c in enumerate(final, 1)}
    for c in final:
        outcome[c.chunk_id] = "returned"
    trace["candidates"] = [{
        **{k: v for k, v in rows[c.chunk_id].items() if k != "returned"},
        "sent": c.chunk_id in by_id,
        "rerank_rank": by_id[c.chunk_id][0] if c.chunk_id in by_id else None,
        "rerank_score": by_id[c.chunk_id][1] if c.chunk_id in by_id else None,
        "final_rank": final_rank.get(c.chunk_id),
        "outcome": outcome[c.chunk_id]} for c in candidates]
    trace["reranked_order"] = [c.chunk_id for _, c, _ in ranked]
    rerank["sent"] = len(sent)
    return _result(final, omitted, trace)


def _result(results: list[RetrievedChunk], omitted: list[OmittedCandidate],
            trace: dict) -> QueryResult:
    trace["returned"] = trace["rerank"]["returned"] = len(results)
    if results:
        status = "ok"
        message = (f"{len(results)} passages selected from {trace['rerank']['sent']} re-ranked "
                   "hybrid candidates; re-rank scores are not proof of correctness.")
    else:
        status = "no_results"
        message = "Both searches ran and found no passages within the requested filters."
    return QueryResult(pattern="hybrid-reranked", status=status, message=message,
                       trace=[trace], results=results, omitted_candidates=omitted)
