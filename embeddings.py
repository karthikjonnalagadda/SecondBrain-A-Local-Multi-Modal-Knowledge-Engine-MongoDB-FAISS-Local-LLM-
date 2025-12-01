# embeddings.py
# sentence-transformers wrapper

from sentence_transformers import SentenceTransformer
import numpy as np

# Model: all-MiniLM-L6-v2 is small & fast (dim=384)
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def embed_texts(texts, normalize=True):
    """
    texts: list[str]
    returns np.ndarray shape (n, d) dtype float32
    """
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embs = embs.astype("float32")
    if normalize:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        embs = embs / norms
    return embs
