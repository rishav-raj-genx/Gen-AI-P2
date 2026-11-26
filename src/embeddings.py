"""GramHealth AI - Multilingual Embedding Module.
Implements dense vector embeddings using intfloat/multilingual-e5-small.
Applies asymmetric 'query: ' and 'passage: ' prefixes and vector normalization
to ensure inner-product operations in FAISS compute exact Cosine Similarity.
"""

from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import config


class MultilingualEmbeddingModel:
    """Singleton wrapper for the multilingual-e5-small embedding model."""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MultilingualEmbeddingModel, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        # Determine best available compute device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"[Embeddings] Loading {config.EMBEDDING_MODEL_NAME} on device: {device}...")
        self._model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
        self.device = device
        self.dimension = config.EMBEDDING_DIM

    def embed_passages(
        self,
        passages: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """Embeds document passages with the mandatory 'passage: ' prefix.

        Returns an (N, 384) L2-normalized float32 numpy array.
        """
        if not passages:
            return np.empty((0, self.dimension), dtype=np.float32)

        prefixed_passages = [f"{config.PASSAGE_PREFIX}{p}" for p in passages]
        embeddings = self._model.encode(
            prefixed_passages,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single user query with the mandatory 'query: ' prefix.

        Returns a 1D (384,) L2-normalized float32 numpy array.
        """
        prefixed_query = f"{config.QUERY_PREFIX}{query.strip()}"
        embedding = self._model.encode(
            prefixed_query,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    def embed_queries_batch(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """Embeds a batch of queries with 'query: ' prefix."""
        prefixed_queries = [f"{config.QUERY_PREFIX}{q.strip()}" for q in queries]
        embeddings = self._model.encode(
            prefixed_queries,
            batch_size=batch_size,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


# Convenience module-level singleton instance
_embedder_instance = None


def get_embedder() -> MultilingualEmbeddingModel:
    """Returns the shared MultilingualEmbeddingModel instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = MultilingualEmbeddingModel()
    return _embedder_instance
