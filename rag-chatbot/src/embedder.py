# src/embedder.py
import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # free, fast, good quality
VECTORDB_DIR = "vectordb"

def load_chunks(chunks_path: str = "chunks/chunks.json") -> list:
    """Load chunks from JSON file."""
    with open(chunks_path) as f:
        return json.load(f)

def build_vector_store(chunks: list) -> None:
    """Convert chunks to embeddings and save in FAISS."""
    Path(VECTORDB_DIR).mkdir(exist_ok=True)

    print(f" Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Extract just the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    print(f"  Creating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32
    )
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    dimension = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (cosine after normalize)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, os.path.join(VECTORDB_DIR, "index.faiss"))

    # Save chunk text separately (FAISS only stores numbers, not text)
    with open(os.path.join(VECTORDB_DIR, "chunks_meta.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f" FAISS index saved! Total vectors: {index.ntotal}")

if __name__ == "__main__":
    chunks = load_chunks()
    build_vector_store(chunks)