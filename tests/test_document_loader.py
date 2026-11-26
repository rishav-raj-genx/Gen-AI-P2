"""GramHealth AI - Document Loader Unit Tests."""

from pathlib import Path
import pytest
import config
from src.document_loader import clean_text, load_all_documents, load_pdf, load_uploaded_pdf


def test_clean_text():
    sample = "This is a  test\xa0with\u200b non-breaking   spaces and hy-\nphenation."
    cleaned = clean_text(sample)
    assert "\xa0" not in cleaned
    assert "\u200b" not in cleaned
    assert "hyphenation" in cleaned
    assert "  " not in cleaned


def test_load_all_documents():
    pages = load_all_documents(config.DOCUMENTS_DIR, config.SOURCES_JSON)
    assert len(pages) > 0, "Expected non-zero document pages loaded."

    first_page = pages[0]
    assert first_page.text is not None
    assert len(first_page.text) > 30
    assert "filename" in first_page.metadata
    assert "page_number" in first_page.metadata
    assert first_page.metadata["page_number"] >= 1
    assert first_page.metadata.get("verified") is True


def test_load_uploaded_pdf_marks_unverified():
    # Load first page of an existing PDF as mock upload
    sample_pdf = list(config.DOCUMENTS_DIR.glob("*.pdf"))[0]
    with open(sample_pdf, "rb") as f:
        pdf_bytes = f.read()

    uploaded_pages = load_uploaded_pdf(pdf_bytes, "test_upload.pdf")
    assert len(uploaded_pages) > 0
    assert uploaded_pages[0].metadata["verified"] is False
    assert "Unverified" in uploaded_pages[0].metadata["source_type"]
