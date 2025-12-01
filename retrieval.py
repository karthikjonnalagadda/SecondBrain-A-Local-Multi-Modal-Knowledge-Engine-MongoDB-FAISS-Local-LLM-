# retrieval.py
# Query flow: embed query, query FAISS, fetch chunks from MongoDB, assemble context

from embeddings import embed_texts
from indexer import load_or_create_index, query_index
from db import get_chunks_by_faiss_ids
import numpy as np

def retrieve(query, top_k=5):
    """
    Returns list of dicts:
    [{
      "faiss_id": int,
      "score": float,
      "chunk": "<text>",
      "doc_id": "...",
      "filename": "...",
      "type": "...",
      "chunk_id": "...",
      "metadata": {...}
    }, ...]
    """
    emb = embed_texts([query])[0]  # (d,)
    index = load_or_create_index()
    ids, scores = query_index(index, emb, top_k=top_k)
    if not ids:
        return []
    # fetch chunk docs from MongoDB
    faiss_to_chunk = get_chunks_by_faiss_ids(ids)
    results = []
    for fid, score in zip(ids, scores):
        chunk_doc = faiss_to_chunk.get(fid)
        if not chunk_doc:
            continue
        # try to fetch filename/type from documents collection if possible
        doc_meta = None
        try:
            from db import documents_col
            doc_meta = documents_col.find_one({"_id": chunk_doc["doc_id"]})
        except Exception:
            doc_meta = None
        results.append({
            "faiss_id": fid,
            "score": float(score),
            "chunk": chunk_doc["text"],
            "doc_id": chunk_doc["doc_id"],
            "filename": doc_meta["filename"] if doc_meta else None,
            "type": doc_meta["type"] if doc_meta else None,
            "chunk_id": chunk_doc["_id"],
            "metadata": chunk_doc.get("metadata", {})
        })
    return results
