"""GramHealth AI - Vector Store and Semantic Retrieval Unit Tests."""

import numpy as np
import pytest
import config
from src.chunking import Chunk
from src.retrieval import FAISSVectorStore


def test_faiss_inner_product_cosine_similarity():
    vstore = FAISSVectorStore(dimension=384)

    # Create synthetic normalized vectors
    vec1 = np.zeros((1, 384), dtype=np.float32)
    vec1[0, 0] = 1.0  # Unit vector along dimension 0

    vec2 = np.zeros((1, 384), dtype=np.float32)
    vec2[0, 1] = 1.0  # Orthogonal vector along dimension 1

    c1 = Chunk("c1", "Text 1", {"doc_id": "d1", "page_number": 1})
    c2 = Chunk("c2", "Text 2", {"doc_id": "d2", "page_number": 2})

    vstore.add_chunks([c1], embeddings=vec1)
    vstore.add_chunks([c2], embeddings=vec2)

    assert len(vstore) == 2

    # Query with identical direction to vec1
    q_vec = np.zeros(384, dtype=np.float32)
    q_vec[0] = 1.0

    results = vstore.search_by_vector(q_vec, top_k=2)
    assert len(results) == 2
    top_chunk, top_score = results[0]
    second_chunk, second_score = results[1]

    # Perfect match should yield score = 1.0
    assert top_chunk.chunk_id == "c1"
    assert np.isclose(top_score, 1.0, atol=1e-3)

    # Orthogonal match should yield score = 0.0
    assert second_chunk.chunk_id == "c2"
    assert np.isclose(second_score, 0.0, atol=1e-3)


def test_load_persisted_vectorstore():
    # Load actual indexed store from disk
    vstore = FAISSVectorStore.load(config.VECTORSTORE_DIR)
    assert len(vstore) > 0

    results = vstore.search("What is the dose of iron in pregnancy?", top_k=2)
    assert len(results) == 2
    assert results[0][1] > 0.80
    assert results[0][0].text != ""
