"""GramHealth AI - Vector Store and Semantic Retrieval Module.
Implements FAISS IndexFlatIP (Inner Product) over L2-normalized dense embeddings,
providing mathematically exact Cosine Similarity search.
Persists the index and chunk metadata locally to eliminate redundant re-indexing.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np

import config
from src.chunking import Chunk
from src.embeddings import get_embedder


class FAISSVectorStore:
    """Manages persistent FAISS vector indexing and semantic retrieval."""

    def __init__(self, dimension: int = config.EMBEDDING_DIM):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: Optional[np.ndarray] = None,
        show_progress: bool = False,
    ):
        """Encodes and adds chunks to the FAISS index."""
        if not chunks:
            return

        if embeddings is None:
            embedder = get_embedder()
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_passages(texts, show_progress=show_progress)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        # FAISS requires float32
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Executes cosine similarity search for a query string.

        Returns list of (Chunk, similarity_score) sorted in descending order of relevance.
        """
        if len(self.chunks) == 0:
            return []

        embedder = get_embedder()
        query_vec = embedder.embed_query(query)
        return self.search_by_vector(query_vec, top_k=top_k, similarity_threshold=similarity_threshold)

    def search_by_vector(
        self,
        query_vec: np.ndarray,
        top_k: int = config.TOP_K,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Executes inner product (cosine similarity) search using a precomputed query vector."""
        if len(self.chunks) == 0:
            return []

        # Ensure shape (1, dim)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        query_vec = query_vec.astype(np.float32)
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, k)

        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            sim_score = float(score)
            if similarity_threshold is not None and sim_score < similarity_threshold:
                continue
            results.append((self.chunks[idx], sim_score))

        return results

    def save(self, directory: Path | str):
        """Serializes the FAISS index and chunk metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_file = directory / "faiss_index.bin"
        metadata_file = directory / "chunks_metadata.json"

        # Write FAISS index
        faiss.write_index(self.index, str(index_file))

        # Write chunk metadata and text
        serialized_chunks = [c.to_dict() for c in self.chunks]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(serialized_chunks, f, ensure_ascii=False, indent=2)

        print(f"[VectorStore] Saved {len(self.chunks)} chunks to {directory}")

    @classmethod
    def load(cls, directory: Path | str) -> "FAISSVectorStore":
        """Loads a persisted FAISS index and chunk metadata from disk."""
        directory = Path(directory)
        index_file = directory / "faiss_index.bin"
        metadata_file = directory / "chunks_metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                f"Vector store files not found in {directory}. Expected {index_file.name} and {metadata_file.name}."
            )

        instance = cls(dimension=config.EMBEDDING_DIM)
        instance.index = faiss.read_index(str(index_file))

        with open(metadata_file, "r", encoding="utf-8") as f:
            serialized_chunks = json.load(f)

        instance.chunks = [
            Chunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                metadata=item["metadata"],
                start_char=item.get("start_char", 0),
                end_char=item.get("end_char", 0),
            )
            for item in serialized_chunks
        ]

        print(f"[VectorStore] Successfully loaded {len(instance.chunks)} indexed chunks from {directory}")
        return instance


def get_default_vector_store() -> Optional[FAISSVectorStore]:
    """Loads the default persisted vector store if available."""
    try:
        return FAISSVectorStore.load(config.VECTORSTORE_DIR)
    except FileNotFoundError:
        return None
