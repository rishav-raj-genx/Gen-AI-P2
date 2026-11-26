"""GramHealth AI - Multilingual Embeddings Unit Tests."""

import numpy as np
import pytest
import config
from src.embeddings import get_embedder


def test_embedding_dimension_and_normalization():
    embedder = get_embedder()
    assert embedder.dimension == config.EMBEDDING_DIM

    query = "What is the dose of iron tablets in pregnancy?"
    q_vec = embedder.embed_query(query)

    assert q_vec.shape == (384,)
    assert q_vec.dtype == np.float32

    # L2 norm of normalized vector should be approximately 1.0
    norm = np.linalg.norm(q_vec)
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_passage_embedding_batch():
    embedder = get_embedder()
    passages = [
        "Exclusive breastfeeding is recommended for the first six months.",
        "Low osmolarity ORS reduces the need for IV fluids in children with dehydration.",
    ]

    p_vecs = embedder.embed_passages(passages)
    assert p_vecs.shape == (2, 384)
    assert p_vecs.dtype == np.float32

    for vec in p_vecs:
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-3)
