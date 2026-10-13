"""Load sections, chunks and embeddings into MongoDB Atlas and create the vector index.

Run: ``uv run python -m building_with_rag.ingestion.load_mongodb [--use-test-db]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime

import voyageai
from pymongo import MongoClient, ReplaceOne
from pymongo.errors import OperationFailure, PyMongoError

from ..config import get_settings
from . import embedding
from .chunking import split_text
from .mongodb_schema import (
    ACCESS_LEVEL,
    CHUNKING_VERSION,
    CHUNKS,
    EMBEDDINGS,
    SECTIONS,
    SOURCES,
    SPLITTER,
    VECTOR_FIELD,
    VECTOR_INDEX_NAME,
    chunk_id,
    embedding_id,
    embedding_input,
    ensure_collections,
    ensure_indexes,
    source_id,
    text_sha256,
    vector_index_definition,
    vector_index_model,
)
from .parser import ACTS, OUT_DIR

INDEX_POLL_SECONDS = 5
INDEX_WAIT_SECONDS = 300
SAMPLE_QUESTION = "What is the punishment for murder?"


class Stop(Exception):
    """A clear prerequisite problem; the message is safe to print."""


def connect(uri: str, db_name: str):
    if not uri:
        raise Stop("Prerequisite missing: MONGODB_URI is empty. Set it in .env (Atlas M0 string).")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    try:
        client.admin.command("ping")
    except PyMongoError as exc:
        raise Stop(f"Prerequisite: MongoDB is unreachable ({type(exc).__name__}). Check "
                   "MONGODB_URI and the Atlas IP access list.") from None
    return client, client[db_name]


def check_atlas(db) -> None:
    try:
        list(db[EMBEDDINGS].list_search_indexes())
    except OperationFailure as exc:
        raise Stop("Prerequisite: this deployment is not Atlas, or your user lacks the privilege "
                   f"for Atlas Search index management (listSearchIndexes: code {exc.code}). "
                   "Stopping; no generic index or other vector store is used.") from None
    print("Deployment: Atlas with Search index management (listSearchIndexes works).")


def read_sections() -> list[dict]:
    records = []
    for cfg in ACTS.values():
        path = OUT_DIR / cfg["out"]
        if not path.exists():
            print(f"  skip: {path.name} not found")
            continue
        with path.open(encoding="utf-8") as fh:
            records.extend(json.loads(line) for line in fh)
    if not records:
        raise Stop("Prerequisite: no section JSONL in data/processed/. Run Story 2.1 first.")
    return records


def load_sections_and_chunks(db, records: list[dict]) -> None:
    now = datetime.now(UTC)
    sources: dict[str, dict] = {}
    for r in records:
        sid = source_id(r["source_pdf"], r["source_sha256"])
        sources.setdefault(sid, {
            "_id": sid, "source_id": sid, "source_pdf": r["source_pdf"],
            "source_sha256": r["source_sha256"], "act": r["act"], "act_label": r["act_label"],
            "status": r["status"], "source_status_version": r["source_status_version"],
            "parser": r["parser"], "parser_version": r["parser_version"], "created_at": now,
        })
    for act in {s["act"] for s in sources.values()}:
        old = [s["_id"] for s in db[SOURCES].find({"act": act}, {"_id": 1})
               if s["_id"] not in sources]
        if old:
            print(f"  changed source for {act}: new PDF hash; sections refreshed (old source {old})")
    for s in sources.values():
        db[SOURCES].update_one({"_id": s["_id"]}, {"$setOnInsert": s}, upsert=True)

    stored = {d["_id"]: d for d in db[SECTIONS].find({}, {"text_sha256": 1, "source_id": 1})}
    stored_ver: dict[str, set] = {}
    for d in db[CHUNKS].find({}, {"section_id": 1, "chunking_version": 1}):
        stored_ver.setdefault(d["section_id"], set()).add(d["chunking_version"])

    sec_ops, written, replaced = [], 0, 0
    for r in records:
        digest = text_sha256(r["text"])
        sid = r["section_id"]
        src = source_id(r["source_pdf"], r["source_sha256"])
        cur = stored.get(sid, {})
        same_text = cur.get("text_sha256") == digest and cur.get("source_id") == src
        if same_text and stored_ver.get(sid) == {CHUNKING_VERSION}:
            continue
        if sid in stored:
            reason = "text or source changed" if not same_text else "chunking version changed"
            print(f"  replacing chunks and embeddings for {sid}: {reason}")
            replaced += 1
        old = [c["_id"] for c in db[CHUNKS].find({"section_id": sid}, {"_id": 1})]
        if old:
            db[EMBEDDINGS].delete_many({"chunk_id": {"$in": old}})
            db[CHUNKS].delete_many({"section_id": sid})
        sec = {k: r[k] for k in (
            "section_id", "act", "act_label", "status", "chapter", "chapter_title",
            "section_number", "heading", "text", "source_pdf", "source_sha256", "parser",
            "parser_version", "source_status_version", "needs_review", "section_label",
            "page_start", "page_end") if k in r}
        sec.update(_id=sid, source_id=src, text_sha256=digest, access_level=ACCESS_LEVEL)
        sec_ops.append(ReplaceOne({"_id": sid}, sec, upsert=True))
        docs = []
        for i, piece in enumerate(split_text(r["text"])):
            cid = chunk_id(sid, CHUNKING_VERSION, i)
            docs.append({
                "_id": cid, "chunk_id": cid, "section_id": sid, "chunk_index": i,
                "chunking_version": CHUNKING_VERSION, "splitter": SPLITTER, "text": piece,
                "embedding_input_text": embedding_input(r["act_label"], r["heading"], piece),
                "act": r["act"], "chapter": r["chapter"], "status": r["status"],
                "access_level": ACCESS_LEVEL})
        db[CHUNKS].insert_many(docs)
        written += 1
        if len(sec_ops) >= 200:
            db[SECTIONS].bulk_write(sec_ops)
            sec_ops = []
    if sec_ops:
        db[SECTIONS].bulk_write(sec_ops)
    print(f"Sections: {written} written or refreshed ({replaced} replaced), "
          f"{len(records) - written} unchanged and reused.")
    print(f"Counts: sources={db[SOURCES].count_documents({})} "
          f"sections={db[SECTIONS].count_documents({})} chunks={db[CHUNKS].count_documents({})}")


def ensure_vector_index(db, dims: int) -> None:
    coll = db[EMBEDDINGS]
    existing = list(coll.list_search_indexes(VECTOR_INDEX_NAME))
    if not existing:
        coll.create_search_index(vector_index_model(dims))
        print(f"Vector index {VECTOR_INDEX_NAME}: created.")
        return
    want = vector_index_definition(dims)
    have = existing[0].get("latestDefinition") or existing[0].get("definition")
    if existing[0].get("type") != "vectorSearch" or have != want:
        raise Stop(f"Index {VECTOR_INDEX_NAME} exists with a different configuration; not "
                   f"touching it.\n  wanted: {want}\n  found:  {have} "
                   f"(type {existing[0].get('type')})")
    print(f"Vector index {VECTOR_INDEX_NAME}: reused (definition matches).")


def wait_for_index(db) -> bool:
    deadline = time.time() + INDEX_WAIT_SECONDS
    while time.time() < deadline:
        found = list(db[EMBEDDINGS].list_search_indexes(VECTOR_INDEX_NAME))
        if found and found[0].get("queryable"):
            print(f"Vector index {VECTOR_INDEX_NAME}: queryable.")
            return True
        time.sleep(INDEX_POLL_SECONDS)
    print("Index not ready yet; re-run to continue.")
    return False


def embed_missing(db, vclient, s) -> None:
    model, ver, dims = s.embedding_model, s.embedding_model_version, s.embedding_dimensions
    have = set(db[EMBEDDINGS].distinct("embedding_id"))
    todo = []
    for c in db[CHUNKS].find({}):
        eid = embedding_id(c["chunk_id"], model, ver, dims, c["embedding_input_text"])
        if eid not in have:
            todo.append((eid, c))
    other = db[EMBEDDINGS].count_documents({"$or": [
        {"embedding_model": {"$ne": model}}, {"embedding_model_version": {"$ne": ver}},
        {"dimensions": {"$ne": dims}}]})
    if other:
        print(f"Note: {other} embeddings use other model settings; reported, not deleted.")
    total = len(todo)
    if not total:
        print("Embeddings: all chunks already embedded; nothing to do.")
        return
    batches = -(-total // embedding.BATCH_SIZE)
    print(f"Embedding {total} chunks in {batches} batches of {embedding.BATCH_SIZE}, "
          f"{embedding.BATCH_DELAY_SECONDS}s apart: expect about "
          f"{batches * embedding.BATCH_DELAY_SECONDS // 60} minutes. Safe to interrupt and re-run.")
    done = 0
    for i in range(0, total, embedding.BATCH_SIZE):
        batch = todo[i:i + embedding.BATCH_SIZE]
        try:
            vecs = embedding.embed_texts(vclient, [c["embedding_input_text"] for _, c in batch],
                                         model, dims, "document")
        except embedding.EmbeddingStopped as exc:
            raise Stop(f"Stopped: {exc}. {total - done} chunks remain; re-run to resume.") from None
        db[EMBEDDINGS].insert_many([{
            "_id": eid, "embedding_id": eid, "chunk_id": c["chunk_id"], "embedding_model": model,
            "embedding_model_version": ver, "dimensions": dims, VECTOR_FIELD: v,
            "act": c["act"], "chapter": c["chapter"], "status": c["status"],
            "access_level": c["access_level"]} for (eid, c), v in zip(batch, vecs, strict=True)])
        done += len(batch)
        print(f"embedded {done}/{total} chunks, {total - done} remaining", flush=True)
        if done < total:
            embedding.delay_between_batches()


def sample_query(db, vclient, s) -> None:
    qv = embedding.embed_texts(vclient, [SAMPLE_QUESTION], s.embedding_model,
                               s.embedding_dimensions, "query")[0]
    hits = list(db[EMBEDDINGS].aggregate([
        {"$vectorSearch": {"index": VECTOR_INDEX_NAME, "path": VECTOR_FIELD, "queryVector": qv,
                           "numCandidates": 50, "limit": 3,
                           "filter": {"access_level": ACCESS_LEVEL}}},
        {"$project": {"chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}}]))
    if not hits:
        print("Sample query: verification pending (no hits yet; index may still be syncing).")
        return
    top = hits[0]
    chunk = db[CHUNKS].find_one({"_id": top["chunk_id"]})
    sec = db[SECTIONS].find_one({"_id": chunk["section_id"]})
    print(f"Sample query {SAMPLE_QUESTION!r} top result: chunk_id={top['chunk_id']} "
          f"section_id={sec['section_id']} heading={sec['heading']!r} score={top['score']:.4f}")


def run(use_test_db: bool, skip_embed: bool) -> None:
    s = get_settings()
    db_name = s.mongodb_test_db_name if use_test_db else s.mongodb_db_name
    client, db = connect(s.mongodb_uri, db_name)
    print(f"Connected. Database: {db_name}")
    check_atlas(db)
    ensure_collections(db)
    ensure_indexes(db)
    load_sections_and_chunks(db, read_sections())
    ensure_vector_index(db, s.embedding_dimensions)
    if skip_embed or not s.voyage_api_key:
        reason = "--skip-embed" if skip_embed else "VOYAGE_API_KEY is empty"
        print(f"Embedding skipped ({reason}); verification pending.")
        return
    vclient = voyageai.Client(api_key=s.voyage_api_key)
    embed_missing(db, vclient, s)
    print(f"Counts: embeddings={db[EMBEDDINGS].count_documents({})}")
    if wait_for_index(db):
        sample_query(db, vclient, s)
    else:
        print("Sample query: verification pending (index not ready).")
    client.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-test-db", action="store_true", help="use MONGODB_TEST_DB_NAME")
    ap.add_argument("--skip-embed", action="store_true", help="skip Voyage embedding")
    args = ap.parse_args()
    try:
        run(args.use_test_db, args.skip_embed)
    except Stop as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
