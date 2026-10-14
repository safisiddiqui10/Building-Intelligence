"""Semantic, hybrid and hybrid-reranked retrieval. Imported lazily; nothing connects at app startup."""

from .hybrid import hybrid_search
from .rerank import rerank_search
from .semantic import RetrievalUnavailable, semantic_search
from .structured import structured_search

__all__ = ["RetrievalUnavailable", "hybrid_search", "rerank_search", "semantic_search",
           "structured_search"]
