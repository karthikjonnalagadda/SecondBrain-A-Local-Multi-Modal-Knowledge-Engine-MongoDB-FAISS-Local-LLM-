# db.py
# MongoDB connection and simple helpers

from pymongo import MongoClient, ReturnDocument
import os
import uuid
from datetime import datetime

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["second_brain"]

# Collections
documents_col = db["documents"]   # high level uploaded files metadata
chunks_col = db["chunks"]         # chunk documents (text + metadata + _faiss_id)
_state_col = db["__internal_state"]  # store things like last_faiss_id

def new_doc_record(filename, dtype, source="upload", extra=None):
    doc = {
        "_id": str(uuid.uuid4()),
        "filename": filename,
        "type": dtype,
        "source": source,
        "created_at": datetime.utcnow().isoformat(),
        "extra": extra or {}
    }
    documents_col.insert_one(doc)
    return doc

def upsert_chunk(chunk_doc):
    """
    chunk_doc must include:
    {
      "_id": uuid,
      "doc_id": ...,
      "text": ...,
      "start_pos": int,
      "end_pos": int,
      "metadata": {...},
      "_faiss_id": int (optional)
    }
    """
    chunks_col.insert_one(chunk_doc)

def allocate_faiss_ids(n):
    """
    Returns a list of n new integer ids for FAISS and stores next counter.
    Uses a single document with key 'faiss_counter'.
    """
    row = _state_col.find_one_and_update(
        {"_id": "faiss_counter"},
        {"$inc": {"value": n}},
        upsert=True,
        return_document=ReturnDocument.BEFORE
    )
    if row is None:
        start = 1
    else:
        start = int(row.get("value", 0)) + 1
    # The find_one_and_update returned BEFORE previous value; compute allocated ids
    if row is None:
        # first allocation: we incremented from nothing to n, start should be 1
        ids = list(range(1, n+1))
    else:
        previous = int(row.get("value", 0))
        ids = list(range(previous+1, previous + 1 + n))
    return ids

def get_chunks_by_faiss_ids(faiss_ids):
    # map int ids to chunk documents
    docs = list(chunks_col.find({"_faiss_id": {"$in": list(faiss_ids)}}))
    # return mapping faiss_id -> chunk
    return {d["_faiss_id"]: d for d in docs}
