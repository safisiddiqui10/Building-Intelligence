"""MongoDB collections, document model, deterministic IDs, validators and indexes.

Every other ingestion module imports names from here; nothing else hard-codes them.
Vectors live only in ``embeddings``, never on ``chunks``.
"""

from __future__ import annotations

import hashlib

from pymongo.database import Database
from pymongo.operations import SearchIndexModel

SOURCES = "sources"
SECTIONS = "sections"
CHUNKS = "chunks"
EMBEDDINGS = "embeddings"
COLLECTIONS = (SOURCES, SECTIONS, CHUNKS, EMBEDDINGS)

VECTOR_INDEX_NAME = "embeddings_vector_index"
VECTOR_FIELD = "vector"
FILTER_FIELDS = ("act", "chapter", "status", "access_level")
ACCESS_LEVEL = "public"

CHUNK_SIZE = 2048
CHUNK_OVERLAP = 256
CHUNKING_VERSION = f"char-{CHUNK_SIZE}-{CHUNK_OVERLAP}-v1"
SPLITTER = "recursive-char"


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_id(source_pdf: str, source_sha256: str) -> str:
    return f"{source_pdf.rsplit('.', 1)[0].lower()}:{source_sha256[:12]}"


def chunk_id(section_id: str, chunking_version: str, chunk_index: int) -> str:
    return f"{section_id}#{chunking_version}#{chunk_index}"


def embedding_id(chunk_id_: str, model: str, model_version: str, dimensions: int,
                 embedding_input_text: str) -> str:
    return f"{chunk_id_}|{model}|{model_version}|{dimensions}|{sha12(embedding_input_text)}"


def embedding_input(act_label: str, heading: str, text: str) -> str:
    return f"{act_label} — {heading}\n\n{text}"


def _obj(required: list[str], props: dict) -> dict:
    return {"$jsonSchema": {"bsonType": "object", "required": required, "properties": props}}


_S, _I, _B = {"bsonType": "string"}, {"bsonType": ["int", "long"]}, {"bsonType": "bool"}

VALIDATORS: dict[str, dict] = {
    SOURCES: _obj(
        ["_id", "source_id", "source_pdf", "source_sha256", "act", "act_label", "status",
         "parser", "parser_version"],
        {"source_id": _S, "source_pdf": _S, "source_sha256": _S, "act": _S, "act_label": _S,
         "status": _S, "parser": _S, "parser_version": _S},
    ),
    SECTIONS: _obj(
        ["_id", "section_id", "source_id", "act", "status", "chapter", "section_number",
         "heading", "text", "text_sha256", "needs_review", "access_level"],
        {"section_id": _S, "source_id": _S, "act": _S, "status": _S, "chapter": _S,
         "section_number": _I, "heading": _S, "text": _S, "text_sha256": _S,
         "needs_review": _B, "access_level": _S},
    ),
    CHUNKS: _obj(
        ["_id", "chunk_id", "section_id", "chunk_index", "chunking_version", "splitter", "text",
         "embedding_input_text", "act", "chapter", "status", "access_level"],
        {"chunk_id": _S, "section_id": _S, "chunk_index": _I, "chunking_version": _S,
         "splitter": _S, "text": _S, "embedding_input_text": _S, "act": _S, "chapter": _S,
         "status": _S, "access_level": _S},
    ),
    EMBEDDINGS: _obj(
        ["_id", "embedding_id", "chunk_id", "embedding_model", "embedding_model_version",
         "dimensions", "vector", "act", "chapter", "status", "access_level"],
        {"embedding_id": _S, "chunk_id": _S, "embedding_model": _S,
         "embedding_model_version": _S, "dimensions": _I,
         "vector": {"bsonType": "array", "items": {"bsonType": "double"}},
         "act": _S, "chapter": _S, "status": _S, "access_level": _S},
    ),
}


def ensure_collections(db: Database) -> None:
    existing = set(db.list_collection_names())
    for name in COLLECTIONS:
        if name in existing:
            db.command("collMod", name, validator=VALIDATORS[name],
                       validationLevel="moderate", validationAction="error")
        else:
            db.create_collection(name, validator=VALIDATORS[name],
                                 validationLevel="moderate", validationAction="error")


def ensure_indexes(db: Database) -> None:
    db[SECTIONS].create_index([("act", 1), ("section_number", 1)])
    db[SECTIONS].create_index([("source_id", 1)])
    db[CHUNKS].create_index([("section_id", 1), ("chunk_index", 1)], unique=True)
    db[EMBEDDINGS].create_index([("chunk_id", 1)])
    db[SOURCES].create_index([("source_sha256", 1)])


def vector_index_definition(dimensions: int) -> dict:
    return {
        "fields": [
            {"type": "vector", "path": VECTOR_FIELD, "numDimensions": dimensions,
             "similarity": "cosine"},
            *({"type": "filter", "path": f} for f in FILTER_FIELDS),
        ]
    }


def vector_index_model(dimensions: int) -> SearchIndexModel:
    return SearchIndexModel(definition=vector_index_definition(dimensions),
                            name=VECTOR_INDEX_NAME, type="vectorSearch")
