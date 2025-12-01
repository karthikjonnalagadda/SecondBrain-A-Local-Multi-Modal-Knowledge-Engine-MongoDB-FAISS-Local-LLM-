# ingest.py
# PDF, audio, image OCR, web scraping ingestion helpers

import os
from pypdf import PdfReader
from db import new_doc_record, upsert_chunk, allocate_faiss_ids
from embeddings import embed_texts
from indexer import load_or_create_index, add_embeddings
import uuid
from datetime import datetime

# transcription (try faster-whisper, fallback to openai-whisper if available)
try:
    from faster_whisper import WhisperModel
    WHISPER_IMPL = "faster-whisper"
    whisper_model = WhisperModel("small", device="cpu")
except Exception:
    WHISPER_IMPL = "whisper"
    try:
        import whisper
        whisper_model = whisper.load_model("small")
    except Exception:
        whisper_model = None

# OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Simple chunker
def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        start += chunk_size - overlap
    return chunks

def ingest_pdf(path, filename=None, chunk_size=800):
    filename = filename or os.path.basename(path)
    doc = new_doc_record(filename, "pdf", source="upload", extra={"path": path})
    reader = PdfReader(path)
    full_text = ""
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            # attach page marker to text
            full_text += f"\n\n[page:{i+1}]\n" + txt
    chunks = chunk_text(full_text, chunk_size=chunk_size)
    # allocate FAISS ids
    faiss_ids = allocate_faiss_ids(len(chunks))
    embs = embed_texts([c[2] for c in chunks])
    index = load_or_create_index()
    # add to FAISS and MongoDB
    for (start, end, text_chunk), faiss_id, emb in zip(chunks, faiss_ids, embs):
        chunk_doc = {
            "_id": str(uuid.uuid4()),
            "doc_id": doc["_id"],
            "text": text_chunk,
            "start_pos": start,
            "end_pos": end,
            "metadata": {"page": None},  # page info is in the text itself [page:N]
            "_faiss_id": int(faiss_id),
            "created_at": datetime.utcnow().isoformat()
        }
        upsert_chunk(chunk_doc)
    # add all embeddings at once
    add_embeddings(index, embs, faiss_ids)
    return {"doc_id": doc["_id"], "ingested_chunks": len(chunks)}

def transcribe_audio(path):
    if WHISPER_IMPL == "faster-whisper" and whisper_model is not None:
        segments, info = whisper_model.transcribe(path, beam_size=5)
        text = " ".join([seg.text for seg in segments])
        return text
    elif WHISPER_IMPL == "whisper" and whisper_model is not None:
        res = whisper_model.transcribe(path)
        return res.get("text", "")
    else:
        raise RuntimeError("No transcription model available. Install faster-whisper or whisper.")

def ingest_audio(path, filename=None, chunk_size=800):
    filename = filename or os.path.basename(path)
    doc = new_doc_record(filename, "audio", source="upload", extra={"path": path})
    text = transcribe_audio(path)
    chunks = chunk_text(text, chunk_size=chunk_size)
    faiss_ids = allocate_faiss_ids(len(chunks))
    embs = embed_texts([c[2] for c in chunks])
    index = load_or_create_index()
    for (start, end, text_chunk), faiss_id in zip(chunks, faiss_ids):
        chunk_doc = {
            "_id": str(uuid.uuid4()),
            "doc_id": doc["_id"],
            "text": text_chunk,
            "start_pos": start,
            "end_pos": end,
            "metadata": {"timestamp": None},
            "_faiss_id": int(faiss_id),
            "created_at": datetime.utcnow().isoformat()
        }
        upsert_chunk(chunk_doc)
    add_embeddings(index, embs, faiss_ids)
    return {"doc_id": doc["_id"], "transcript_snippet": text[:500], "ingested_chunks": len(chunks)}

def ingest_image(path, filename=None, chunk_size=800):
    if not OCR_AVAILABLE:
        raise RuntimeError("pytesseract not available. Install tesseract and pytesseract.")
    filename = filename or os.path.basename(path)
    doc = new_doc_record(filename, "image", source="upload", extra={"path": path})
    from PIL import Image
    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    chunks = chunk_text(text, chunk_size=chunk_size)
    faiss_ids = allocate_faiss_ids(len(chunks))
    embs = embed_texts([c[2] for c in chunks])
    index = load_or_create_index()
    for (start, end, text_chunk), faiss_id in zip(chunks, faiss_ids):
        chunk_doc = {
            "_id": str(uuid.uuid4()),
            "doc_id": doc["_id"],
            "text": text_chunk,
            "start_pos": start,
            "end_pos": end,
            "metadata": {},
            "_faiss_id": int(faiss_id),
            "created_at": datetime.utcnow().isoformat()
        }
        upsert_chunk(chunk_doc)
    add_embeddings(index, embs, faiss_ids)
    return {"doc_id": doc["_id"], "ingested_chunks": len(chunks)}
