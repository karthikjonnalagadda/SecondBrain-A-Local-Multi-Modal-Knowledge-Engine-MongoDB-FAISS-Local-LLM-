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
## <img width="1836" height="936" alt="image" src="https://github.com/user-attachments/assets/8628efc0-6d15-457e-abaa-9cdf16aaccc4" />


### 🖼️ Home UI
<img width="1063" height="928" alt="image" src="https://github.com/user-attachments/assets/8d5993c4-1cdd-40ee-8eca-6e35cc351056" />


### 🧠 Query Answer
<img width="1331" height="315" alt="image" src="https://github.com/user-attachments/assets/3eb2dbc9-2fba-42d8-94a7-ccd90aff98db" />


### 📤 Upload Section
![Upload Area](screenshots/upload_area.png)
