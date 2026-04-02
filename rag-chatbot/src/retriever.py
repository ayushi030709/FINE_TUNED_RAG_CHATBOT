# src/retriever.py
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTORDB_DIR = "vectordb"

class Retriever:
    def __init__(self):
        print(" Loading retriever components...")
        
        # Load the same embedding model
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Load saved FAISS index
        self.index = faiss.read_index(f"{VECTORDB_DIR}/index.faiss")
        
        # Load saved chunk texts
        with open(f"{VECTORDB_DIR}/chunks_meta.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        
        print(f" Retriever ready! {self.index.ntotal} chunks available.")

    def retrieve(self, query: str, top_k: int = 4) -> list:
        """
        Find top_k most relevant chunks for the query.
        Returns list of chunks with similarity scores.
        """
        # Embed the question
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search FAISS
        scores, indices = self.index.search(query_embedding, top_k)

        # Collect results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1 means no result found
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                results.append(chunk)

        return results


# Test it
if __name__ == "__main__":
    r = Retriever()
    results = r.retrieve("How do I opt out of arbitration?")
    for res in results:
        print(f"\n[Score: {res['score']:.3f}]")
        print(res["text"][:200])