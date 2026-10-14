"""Grounded answer generation (Stories 3.1-3.2). Imported lazily; nothing connects at app startup."""

from .answer import add_generation, generate_answer, generate_events, with_generation
from .context import assemble_context
from .render import render

__all__ = ["add_generation", "assemble_context", "generate_answer", "generate_events", "render",
           "with_generation"]
