"""Character-based recursive splitter. Chunks never cross a section."""

from __future__ import annotations

from .mongodb_schema import CHUNK_OVERLAP, CHUNK_SIZE

SEPARATORS = ("\n\n", "\n", ". ", " ")


def _pieces(text: str, size: int, seps: tuple[str, ...]) -> list[str]:
    """Split into pieces no longer than ``size``, keeping separators attached."""
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    out: list[str] = []
    for i, p in enumerate(parts):
        p = p + sep if i < len(parts) - 1 else p
        out.extend(_pieces(p, size, rest) if len(p) > size else [p])
    return out


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Pack pieces up to ``size`` characters; each next chunk starts with ~overlap chars."""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    cur = ""
    for piece in _pieces(text, size - overlap, SEPARATORS):
        if cur and len(cur) + len(piece) > size:
            chunks.append(cur)
            tail = cur[-overlap:]
            cur = tail + piece
        else:
            cur += piece
    if cur and (not chunks or len(cur) > overlap):
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]
