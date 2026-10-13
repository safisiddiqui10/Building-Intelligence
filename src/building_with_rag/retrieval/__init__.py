"""Semantic retrieval (Story 2.3). Imported lazily; nothing connects at app startup."""

from .semantic import RetrievalUnavailable, semantic_search

__all__ = ["RetrievalUnavailable", "semantic_search"]
