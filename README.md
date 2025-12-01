# 📘 SecondBrain – Local Multi-Modal Knowledge Engine  
### MongoDB • FAISS • Sentence-Transformers • Faster-Whisper • FLAN-T5 (Local LLM)  
**Fully Offline • No OpenAI • Privacy-Focused**

---

## 🚀 Overview
**SecondBrain** is a fully local AI knowledge engine that ingests and processes:

- 📄 PDF files  
- 🎧 Audio recordings (local transcription)  
- 🖼️ Images (OCR)  
- 🌐 Web URLs (scraping)  
- 📝 Plain text / notes  

It extracts text, chunks content, generates embeddings using **Sentence-Transformers**, stores metadata in **MongoDB**, indexes vectors in **FAISS**, and answers your questions using a **local LLM (FLAN-T5)** — all offline.

This acts as your personal **Second Brain**, with powerful multi-modal recall and summarization.

---

## 🌟 Features

### 🔹 Multi-Modal Ingestion
- PDF → text extraction  
- Audio → whisper transcription  
- Image → Tesseract OCR  
- URL → webpage/article scraper  
- Text → direct ingestion  

### 🔹 Vector Search with FAISS
- Embeddings via `all-MiniLM-L6-v2`  
- Fast cosine similarity search  
- Chunking with overlap  

### 🔹 Local LLM Q&A
- Runs entirely offline  
- Uses **FLAN-T5 Small**  
- Provides summarized answers  
- Includes **source citations**  

### 🔹 MongoDB Storage
- Full chunk metadata  
- Timestamps, file info  
- FAISS vector ID mapping  

### 🔹 Web UI
- Upload PDFs, audio, images  
- Ingest URLs & text  
- Ask natural language questions  
- View answer + sources  

---

## 🏗️ Architecture

┌────────────────────────────┐
│ Web UI │
│ (HTML + CSS + JS) │
└───────────────┬────────────┘
│
▼
┌───────────────┐
│ FastAPI │
│ (Backend) │
└───────┬───────┘
│
┌────────────┴────────────┐
│ │
▼ ▼
┌──────────────┐ ┌────────────────┐
│ Ingestion │ │ Retrieval + │
│ PDF/Audio/... │ │ Local LLM Q&A │
└───────┬──────┘ └───────┬────────┘
│ │
▼ ▼
┌─────────────────────┐ ┌─────────────────────┐
│ MongoDB │ │ FAISS │
│ (metadata + chunks) │ │ (vector index) │
└──────────────────────┘ └──────────────────────┘


---

## 🧰 Tech Stack

| Component | Technology |
|----------|------------|
| Backend | FastAPI (Python) |
| Storage | MongoDB |
| Vector Index | FAISS-CPU |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Audio Transcription | Faster-Whisper / Whisper |
| OCR | Tesseract |
| Local LLM | FLAN-T5 Small |
| Frontend | HTML + CSS + JavaScript |

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/karthikjonnalagadda/SecondBrain-A-Local-Multi-Modal-Knowledge-Engine-MongoDB-FAISS-Local-LLM-.git
cd SecondBrain-A-Local-Multi-Modal-Knowledge-Engine-MongoDB-FAISS-Local-LLM-

2️⃣ Create & Activate Virtual Env

python -m venv .venv
.\.venv\Scripts\activate     # Windows
# source .venv/bin/activate # Linux / Mac

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Install Tesseract OCR

Windows (Chocolatey):

choco install tesseract

▶️ Running the App
Start FastAPI

uvicorn app:app --reload --port 8000

Open Web UI

http://127.0.0.1:8000/

🖥️ UI Screenshots

Create a folder:

/screenshots

Add your images, then use:

### 📄 PDF Upload
![PDF Upload](screenshots/pdf_upload.png)

### 🎧 Audio Transcription
![Audio Upload](screenshots/audio_upload.png)

### 🖼️ Image OCR
![OCR](screenshots/ocr.png)

### 🌐 URL Ingestion
![URL](screenshots/url.png)

### 🤖 Query Answer
![Answer](screenshots/answer.png)

📡 API Endpoints
POST /ingest/pdf

Upload a PDF
Response:

{
  "doc_id": "...",
  "ingested_chunks": 42
}

POST /ingest/audio

Upload audio → whisper transcription

{
  "doc_id": "...",
  "transcript_snippet": "First part...",
  "ingested_chunks": 19
}

POST /ingest/image

Upload an image → OCR → ingest
POST /ingest/url

{ "url": "https://example.com/article" }

POST /ingest/text

{
  "text": "These are my notes...",
  "filename": "notes.txt"
}

POST /query

Ask a natural language question

{
  "q": "Summarize the key points",
  "top_k": 5
}

Response:

{
  "answer": "Summary...",
  "sources": [
    {
      "filename": "file.pdf",
      "snippet": "text chunk...",
      "page": 3,
      "score": 0.88
    }
  ]
}

📈 Future Improvements

    User accounts & search history

    Better UI (React + Tailwind)

    Support for LLaMA / Mistral local LLMs

    Timeline-based searching

    PDF table extraction

    Encrypt MongoDB storage

    Embed images (CLIP)

    Audio diarization

📝 License

MIT License
(Let me know if you want me to generate the LICENSE file.)
👤 Author

Karthik Jonnalagadda
GitHub: https://github.com/karthikjonnalagadda
