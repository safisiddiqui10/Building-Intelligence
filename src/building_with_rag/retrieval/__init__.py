"""Semantic and hybrid retrieval. Imported lazily; nothing connects at app startup."""

from .hybrid import hybrid_search
from .semantic import RetrievalUnavailable, semantic_search

__all__ = ["RetrievalUnavailable", "hybrid_search", "semantic_search"]
