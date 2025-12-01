# app.py
# FastAPI server tying ingestion, retrieval and generation together.
# Usage:
# uvicorn app:app --reload --port 8000

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from ingest import ingest_pdf, ingest_audio, ingest_image
from retrieval import retrieve
from generator import generate_answer
from db import documents_col, chunks_col
from typing import Optional

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Second Brain (Mongo + FAISS, no-OpenAI)")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/ingest/pdf")
async def api_ingest_pdf(file: UploadFile = File(...)):
    try:
        dest = os.path.join(UPLOAD_DIR, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        res = ingest_pdf(dest, filename=file.filename)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/audio")
async def api_ingest_audio(file: UploadFile = File(...)):
    try:
        dest = os.path.join(UPLOAD_DIR, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        res = ingest_audio(dest, filename=file.filename)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/image")
async def api_ingest_image(file: UploadFile = File(...)):
    try:
        dest = os.path.join(UPLOAD_DIR, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        res = ingest_image(dest, filename=file.filename)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def api_query(payload: dict):
    """
    payload: {"q": "...", "top_k": 5}
    returns:
    {
      "answer": "...",
      "sources": [ {doc_id, chunk_id, filename, type, page, timestamp, snippet, score}, ... ]
    }
    """
    q = payload.get("q") or payload.get("query")
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'q' in payload")
    top_k = int(payload.get("top_k", 5))
    # 1) retrieve top chunks
    chunks = retrieve(q, top_k=top_k)
    if not chunks:
        return {"answer": "No relevant content found.", "sources": []}
    # 2) assemble context: include chunk snippets with simple separators and source labels
    context_parts = []
    for c in chunks:
        label = c.get("filename") or c.get("doc_id") or "unknown"
        context_parts.append(f"[{label}] {c['chunk']}")
    context_text = "\n\n".join(context_parts)
    prompt = f"""You are an assistant. Use ONLY the provided CONTEXT to answer the question. If the context does not contain the answer, say "I don't know."

CONTEXT:
{context_text}

QUESTION:
{q}

Answer concisely and at the end, list which sources (by filename) you used."""
    # 3) generate
    answer = generate_answer(prompt, max_length=256)
    # 4) build response with snippets for UI
    sources = []
    for c in chunks:
        sources.append({
            "doc_id": c.get("doc_id"),
            "chunk_id": c.get("chunk_id"),
            "filename": c.get("filename"),
            "type": c.get("type"),
            "page": c.get("metadata", {}).get("page"),
            "timestamp": c.get("metadata", {}).get("timestamp"),
            "snippet": c.get("chunk"),
            "score": c.get("score")
        })
    return {"answer": answer, "sources": sources}

@app.get("/")
async def root():
    return FileResponse("static/index.html")
