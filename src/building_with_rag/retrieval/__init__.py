"""Semantic, hybrid, hybrid-reranked, structured, decomposition and HyDE retrieval. Imported lazily; nothing connects at app startup."""

from .agentic import agentic_search
from .decomposition import decomposition_search
from .hybrid import hybrid_search
from .hyde import hyde_search
from .rerank import rerank_search
from .semantic import RetrievalUnavailable, semantic_search
from .structured import structured_search

__all__ = ["RetrievalUnavailable", "agentic_search", "decomposition_search", "hybrid_search", "hyde_search", "rerank_search", "semantic_search",
           "structured_search"]
