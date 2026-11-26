"""GramHealth AI - End-to-End RAG Pipeline Unit and Integration Tests."""

import pytest
from src.generation import RAGPipeline


@pytest.fixture(scope="module")
def rag_pipeline():
    return RAGPipeline()


def test_standard_retrieval_query_citations(rag_pipeline):
    query = "What is the recommended dose of iron and folic acid for pregnant women?"
    result = rag_pipeline.run(query, top_k=4)

    assert result["is_emergency"] is False
    assert result["is_out_of_scope"] is False
    assert len(result["retrieved_chunks"]) > 0
    assert len(result["citations"]) > 0

    first_cit = result["citations"][0]
    assert "doc_title" in first_cit
    assert "page_number" in first_cit
    assert "similarity_score" in first_cit
    assert first_cit["similarity_score"] >= 0.80


def test_emergency_interception(rag_pipeline):
    query = "Crushing chest pain and patient collapsed unconscious"
    result = rag_pipeline.run(query)

    assert result["is_emergency"] is True
    assert "108" in result["answer"]
    assert result["mode"] == "Emergency Guardrail"


def test_out_of_scope_query_insufficient_info(rag_pipeline):
    query = "How do I invest in cryptocurrency bitcoin blockchain?"
    result = rag_pipeline.run(query)

    assert result["is_out_of_scope"] is True
    assert "insufficient information" in result["answer"].lower()
