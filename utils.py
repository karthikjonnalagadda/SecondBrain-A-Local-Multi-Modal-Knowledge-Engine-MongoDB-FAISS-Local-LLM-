# utils.py
import uuid
import os
from datetime import datetime

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def new_id():
    return str(uuid.uuid4())

def chunk_text(text, chunk_size=800, chunk_overlap=200):
    """Yield (chunk_text, start_pos, end_pos). Operates on characters."""
    if not text:
        return []
    start = 0
    n = len(text)
    chunks = []
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append((chunk.strip(), start, min(end, n)))
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
