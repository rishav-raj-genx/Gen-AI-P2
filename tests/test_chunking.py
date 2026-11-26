"""GramHealth AI - Chunking Unit Tests."""

import pytest
from src.chunking import (
    Chunk,
    chunk_documents,
    compute_chunk_statistics,
    extract_overlap_samples,
    find_overlap,
    split_fixed_size,
)
from src.document_loader import DocumentPage


def test_split_fixed_size():
    text = "A" * 1600
    segments = split_fixed_size(text, chunk_size=800, chunk_overlap=0)
    assert len(segments) == 2
    assert len(segments[0][0]) == 800
    assert len(segments[1][0]) == 800


def test_chunk_documents_recursive_metadata_preservation():
    page = DocumentPage(
        text="Maternal health requires balanced nutrition.\n\nIron supplementation prevents severe anaemia in pregnant mothers.\n\nASHA workers provide community guidance.",
        metadata={
            "doc_id": "test_doc",
            "title": "Test Title",
            "page_number": 3,
            "organization": "WHO",
            "verified": True,
        },
    )

    chunks = chunk_documents([page], chunk_size=80, chunk_overlap=20, method="recursive")
    assert len(chunks) > 0

    for c in chunks:
        assert c.metadata["doc_id"] == "test_doc"
        assert c.metadata["page_number"] == 3
        assert c.metadata["organization"] == "WHO"
        assert c.metadata["verified"] is True
        assert "chunk_id" in c.metadata


def test_compute_chunk_statistics():
    c1 = Chunk("c1", "Hello world this is chunk one.", {"doc_id": "d1"})
    c2 = Chunk("c2", "Hello world this is chunk two and longer.", {"doc_id": "d1"})
    stats = compute_chunk_statistics([c1, c2])

    assert stats["total_chunks"] == 2
    assert stats["min_chars"] > 0
    assert stats["max_chars"] >= stats["min_chars"]
    assert stats["mean_chars"] > 0


def test_find_overlap():
    t1 = "Maternal nutrition is vital for child growth and cognitive brain development."
    t2 = "cognitive brain development. In rural communities, ASHA workers play an important role."
    overlap = find_overlap(t1, t2)
    assert "cognitive brain development." in overlap
