from building_with_rag.ingestion import mongodb_schema as m
from building_with_rag.ingestion.chunking import split_text


def test_ids_are_deterministic_and_act_qualified():
    assert m.chunk_id("bns:103", m.CHUNKING_VERSION, 0) == m.chunk_id("bns:103", m.CHUNKING_VERSION, 0)
    assert m.chunk_id("bns:103", "v", 0) != m.chunk_id("ipc:103", "v", 0)
    a = m.embedding_id("c", "voyage-3.5", "voyage-3.5", 1024, "x")
    assert a == m.embedding_id("c", "voyage-3.5", "voyage-3.5", 1024, "x")
    assert a != m.embedding_id("c", "voyage-3.5", "voyage-3.5", 1024, "y")


def test_chunker_size_overlap_and_short_section():
    assert split_text("short text") == ["short text"]
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(12))
    chunks = split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= m.CHUNK_SIZE for c in chunks)
    assert chunks[0][-100:].strip()[-40:] in chunks[1]


def test_vector_index_definition():
    d = m.vector_index_definition(1024)
    assert d["fields"][0] == {"type": "vector", "path": "vector", "numDimensions": 1024,
                              "similarity": "cosine"}
    assert [f["path"] for f in d["fields"][1:]] == ["act", "chapter", "status", "access_level"]
