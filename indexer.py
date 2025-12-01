# indexer.py
# FAISS index management (create, add, query, persist)

import faiss
import os
import numpy as np
from embeddings import embed_texts

INDEX_PATH = "./faiss.index"
DIM = 384  # embedding dim for all-MiniLM-L6-v2

def load_or_create_index():
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        quant = None
        # Use IndexFlatIP (inner product) + IDMap for mapping int ids -> vectors
        index = faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
    return index

def save_index(index):
    faiss.write_index(index, INDEX_PATH)

def add_embeddings(index, embs: np.ndarray, ids: list):
    """
    embs: numpy float32 (n, d), assumed normalized for cosine similarity
    ids: list of ints length n
    """
    assert embs.shape[0] == len(ids)
    index.add_with_ids(embs, np.array(ids, dtype="int64"))
    save_index(index)

def query_index(index, q_emb: np.ndarray, top_k=5):
    """
    q_emb: numpy array shape (d,) or (1, d)
    returns: (ids, scores) both lists of length <= top_k
    """
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    q_emb = q_emb.astype("float32")
    D, I = index.search(q_emb, top_k)
    # D: distances (inner product scores), I: ids
    ids = I[0].tolist()
    scores = D[0].tolist()
    # remove -1 ids (no result)
    out = [(i, s) for i, s in zip(ids, scores) if i != -1]
    if not out:
        return [], []
    ids, scores = zip(*out)
    return list(ids), list(scores)
