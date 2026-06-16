"""
Phase 3 – Dual Index Builder
Builds and saves both FAISS (semantic) and BM25 (lexical) indexes
over the chunked corpus.
"""

import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.config import (
    EMBEDDING_MODEL, INDEX_DIR, DATA_PROCESSED_DIR
)


# ── FAISS Index ─────────────────────────────────────────

def build_faiss_index(chunks: list[dict]) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    """
    Embed all chunks and build a FAISS inner-product index.

    Returns:
        (faiss_index, embeddings_array)
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # normalize for cosine similarity via inner product
        batch_size=64,
    )

    # Build FAISS index (inner product on normalized vectors = cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    print(f"FAISS index built: {index.ntotal} vectors, {dimension}d")
    return index, embeddings


def save_faiss_index(index: faiss.IndexFlatIP):
    """Save FAISS index to disk."""
    path = INDEX_DIR / "faiss.index"
    faiss.write_index(index, str(path))
    print(f"FAISS index saved to: {path}")


def load_faiss_index() -> faiss.IndexFlatIP:
    """Load FAISS index from disk."""
    path = INDEX_DIR / "faiss.index"
    return faiss.read_index(str(path))


# ── BM25 Index ──────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Build a BM25 index over the chunked corpus.

    Returns:
        BM25Okapi instance
    """
    print("Building BM25 index...")
    tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built over {len(tokenized_corpus)} documents")
    return bm25


def save_bm25_index(bm25: BM25Okapi):
    """Save BM25 index to disk via pickle."""
    path = INDEX_DIR / "bm25.pkl"
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved to: {path}")


def load_bm25_index() -> BM25Okapi:
    """Load BM25 index from disk."""
    path = INDEX_DIR / "bm25.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Build Both ──────────────────────────────────────────

def build_all_indexes(chunks: list[dict]):
    """Build and save both FAISS and BM25 indexes."""

    # FAISS
    faiss_index, _ = build_faiss_index(chunks)
    save_faiss_index(faiss_index)

    # BM25
    bm25 = build_bm25_index(chunks)
    save_bm25_index(bm25)

    # Also save the chunk metadata alongside indexes
    # (needed at query time to map index positions back to chunks)
    meta_path = INDEX_DIR / "chunk_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Chunk metadata saved to: {meta_path}")

    print("\n✓ All indexes built and saved.")


if __name__ == "__main__":
    chunks_path = DATA_PROCESSED_DIR / "chunks.json"
    if chunks_path.exists():
        with open(chunks_path) as f:
            chunks = json.load(f)
        build_all_indexes(chunks)
    else:
        print("Run chunking first to generate chunks.json")
