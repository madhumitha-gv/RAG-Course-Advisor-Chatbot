"""
Phase 4 – Hybrid Retriever
Queries both FAISS (semantic) and BM25 (lexical) indexes,
fuses results with Reciprocal Rank Fusion (RRF).
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL, INDEX_DIR,
    TOP_K_PER_RETRIEVER, TOP_K_FINAL, RRF_K
)
from src.index_builder import (
    load_faiss_index, load_bm25_index, _tokenize
)


class HybridRetriever:
    """
    Combines FAISS semantic search with BM25 lexical search
    using Reciprocal Rank Fusion.
    """

    def __init__(self):
        print("Loading retriever components...")

        # Load indexes
        self.faiss_index = load_faiss_index()
        self.bm25 = load_bm25_index()

        # Load chunk metadata
        meta_path = INDEX_DIR / "chunk_metadata.json"
        with open(meta_path) as f:
            self.chunks = json.load(f)

        # Load embedding model (for query encoding)
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"✓ Retriever ready — {len(self.chunks)} chunks indexed")

    def _search_faiss(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """
        Semantic search via FAISS.
        Returns list of (chunk_index, score) tuples.
        """
        query_embedding = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:  # FAISS returns -1 for empty slots
                results.append((int(idx), float(score)))
        return results

    def _search_bm25(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """
        Lexical search via BM25.
        Returns list of (chunk_index, score) tuples.
        """
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices by score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero matches
                results.append((int(idx), float(scores[idx])))
        return results

    def _reciprocal_rank_fusion(
        self,
        faiss_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """
        Fuse results from both retrievers using RRF.
        score = sum( 1 / (k + rank) ) across retrievers.
        """
        rrf_scores = {}

        # Score FAISS results by rank
        for rank, (idx, _) in enumerate(faiss_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

        # Score BM25 results by rank
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

        # Sort by fused score descending
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:TOP_K_FINAL]

    def retrieve(self, query: str) -> list[dict]:
        """
        Main retrieval method.
        Returns top-k chunks with metadata, ranked by RRF score.
        """
        # Get results from both retrievers
        faiss_results = self._search_faiss(query, TOP_K_PER_RETRIEVER)
        bm25_results = self._search_bm25(query, TOP_K_PER_RETRIEVER)

        # Fuse with RRF
        fused_results = self._reciprocal_rank_fusion(faiss_results, bm25_results)

        # Build output with chunk metadata
        retrieved = []
        for idx, rrf_score in fused_results:
            chunk = self.chunks[idx].copy()
            chunk["rrf_score"] = round(rrf_score, 6)
            retrieved.append(chunk)

        return retrieved


if __name__ == "__main__":
    retriever = HybridRetriever()

    # Test queries
    test_queries = [
        "What are the core courses for Data Science MS?",
        "DSCI-D590",
        "machine learning electives",
        "prerequisite requirements for Computer Science PhD",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        results = retriever.retrieve(q)
        for i, r in enumerate(results):
            print(f"\n  [{i+1}] Score: {r['rrf_score']}")
            print(f"      Program: {r['program']} ({r['level']})")
            print(f"      Heading: {r['heading']}")
            print(f"      Text:    {r['text'][:150]}...")
