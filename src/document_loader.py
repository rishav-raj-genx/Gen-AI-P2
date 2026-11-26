"""GramHealth AI - Document Loader Module.
Extracts and cleans text from authentic healthcare PDFs using PyMuPDF.
Preserves page-level metadata for transparent citation.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pymupdf


class DocumentPage:
    """Represents an extracted page with structured metadata."""

    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"<DocumentPage doc={self.metadata.get('doc_id')} page={self.metadata.get('page_number')} chars={len(self.text)}>"


def clean_text(text: str) -> str:
    """Cleans extracted PDF text to remove artifacts, excessive whitespace,

    and normalize punctuation while preserving medical terms.
    """
    if not text:
        return ""

    # Replace non-breaking spaces and unusual whitespace
    text = text.replace("\xa0", " ").replace("\u200b", "")

    # Normalize hyphens and dashes
    text = re.sub(r"[—–]", "-", text)

    # Rejoin words broken across lines by hyphens (e.g., "sup-\nplementation" -> "supplementation")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Replace multiple newlines with double newline to preserve paragraph separation
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces or tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading and trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join(lines).strip()

    return cleaned


def load_pdf(
    pdf_path: Path | str,
    metadata_override: Optional[Dict[str, Any]] = None
) -> List[DocumentPage]:
    """Loads a single PDF using PyMuPDF, extracts text per page,

    and attaches full provenance metadata.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    pages: List[DocumentPage] = []

    base_metadata = {
        "filename": pdf_path.name,
        "doc_id": pdf_path.stem,
        "title": pdf_path.stem.replace("_", " ").title(),
        "organization": "Official Healthcare Source",
        "original_url": "",
        "verified": True,
        "source_type": "Official Repository",
    }
    if metadata_override:
        base_metadata.update(metadata_override)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)

        # Skip pages with negligible text (e.g. blank pages, purely pictorial covers)
        if len(cleaned.strip()) < 30:
            continue

        page_meta = dict(base_metadata)
        page_meta["page_number"] = page_idx + 1  # 1-indexed for citation
        page_meta["char_count"] = len(cleaned)

        pages.append(DocumentPage(text=cleaned, metadata=page_meta))

    doc.close()
    return pages


def load_all_documents(
    documents_dir: Path | str,
    sources_json_path: Optional[Path | str] = None
) -> List[DocumentPage]:
    """Loads all PDF files in the documents directory with metadata from sources.json."""
    documents_dir = Path(documents_dir)
    sources_map: Dict[str, Dict[str, Any]] = {}

    if sources_json_path:
        sources_path = Path(sources_json_path)
        if sources_path.exists():
            with open(sources_path, "r", encoding="utf-8") as f:
                sources_list = json.load(f)
                for item in sources_list:
                    sources_map[item.get("filename", "")] = item

    all_pages: List[DocumentPage] = []
    pdf_files = sorted(list(documents_dir.glob("*.pdf")))

    for pdf_file in pdf_files:
        meta_override = sources_map.get(pdf_file.name, {})
        meta_dict = {
            "filename": pdf_file.name,
            "doc_id": meta_override.get("id", pdf_file.stem),
            "title": meta_override.get("title", pdf_file.stem.replace("_", " ").title()),
            "organization": meta_override.get("organization", "Official Healthcare Source"),
            "original_url": meta_override.get("original_url", ""),
            "topics": meta_override.get("topics", []),
            "verified": meta_override.get("verified", True),
            "source_type": "Official Repository",
        }
        pages = load_pdf(pdf_file, metadata_override=meta_dict)
        all_pages.extend(pages)

    return all_pages


def load_uploaded_pdf(file_bytes: bytes, filename: str) -> List[DocumentPage]:
    """Loads an uploaded PDF from in-memory bytes and marks it unverified."""
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    pages: List[DocumentPage] = []

    base_metadata = {
        "filename": filename,
        "doc_id": f"upload_{Path(filename).stem}",
        "title": Path(filename).stem.replace("_", " ").title(),
        "organization": "Community / User Upload",
        "original_url": "Uploaded via UI",
        "topics": ["User Document"],
        "verified": False,
        "source_type": "Unverified Community Upload",
    }

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)

        if len(cleaned.strip()) < 30:
            continue

        page_meta = dict(base_metadata)
        page_meta["page_number"] = page_idx + 1
        page_meta["char_count"] = len(cleaned)

        pages.append(DocumentPage(text=cleaned, metadata=page_meta))

    doc.close()
    return pages
