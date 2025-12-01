# second-brain-mongo-no-openai

Self-hosted minimal multi-modal personal knowledge base:
- PDF + Audio ingestion
- Chunking + embeddings (sentence-transformers)
- FAISS vector index
- MongoDB for metadata & chunk storage
- Local LLM (flan-t5-small) for synthesis — no paid APIs

## Requirements
- Python 3.9+
- MongoDB running locally or accessible via URI
- Optional: GPU + CUDA for faster transcription and generation

## Setup (Linux / macOS / WSL / Windows)
1. Clone repo and enter folder:
```bash
git clone <repo> second-brain-mongo-no-openai
cd second-brain-mongo-no-openai
