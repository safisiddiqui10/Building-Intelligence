"""Grounded answer generation (Story 3.1). Imported lazily; nothing connects at app startup."""

from .answer import add_generation, generate_answer
from .context import assemble_context

__all__ = ["add_generation", "assemble_context", "generate_answer"]
