"""GramHealth AI - Chunking Module.
Implements Recursive Character Text Splitting with Overlapping Chunks,
Fixed-Size Chunking for academic comparison, metadata preservation,
and overlap analysis.
"""

import statistics
from typing import Any, Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.document_loader import DocumentPage


class Chunk:
    """Represents a text chunk with complete source provenance and boundary information."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any],
        start_char: int = 0,
        end_char: int = 0,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.start_char = start_char
        self.end_char = end_char

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "char_count": self.char_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    def __repr__(self) -> str:
        return (
            f"<Chunk id={self.chunk_id} doc={self.metadata.get('doc_id')} "
            f"page={self.metadata.get('page_number')} chars={len(self.text)}>"
        )


def split_fixed_size(text: str, chunk_size: int, chunk_overlap: int = 0) -> List[tuple[str, int, int]]:
    """Splits text rigidly into fixed-size character windows regardless of sentence or paragraph boundaries.

    Used as the baseline control for academic comparison.
    """
    if not text:
        return []

    chunks_with_spans = []
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size

    pos = 0
    text_len = len(text)
    while pos < text_len:
        end = min(pos + chunk_size, text_len)
        segment = text[pos:end]
        if segment.strip():
            chunks_with_spans.append((segment, pos, end))
        if end >= text_len:
            break
        pos += step

    return chunks_with_spans


def chunk_documents(
    pages: List[DocumentPage],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    method: str = "recursive",
    separators: Optional[List[str]] = None,
) -> List[Chunk]:
    """Splits a list of DocumentPages into Chunks while preserving metadata.

    Supports:
      - 'recursive': RecursiveCharacterTextSplitter (paragraphs -> newlines -> sentences -> words -> characters)
      - 'fixed': Fixed-size windowing (ignoring syntactic boundaries)
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    recursive_splitter = None
    if method == "recursive":
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            keep_separator=True,
        )

    all_chunks: List[Chunk] = []

    for page in pages:
        doc_id = page.metadata.get("doc_id", "doc")
        page_num = page.metadata.get("page_number", 1)
        page_text = page.text

        if not page_text or len(page_text.strip()) == 0:
            continue

        if method == "recursive":
            # Extract chunks with character offsets where possible
            text_chunks = recursive_splitter.split_text(page_text)
            curr_pos = 0
            for idx, chunk_text in enumerate(text_chunks):
                chunk_id = f"{doc_id}_p{page_num}_c{idx + 1}"
                # Locate character offset
                found_idx = page_text.find(chunk_text[:50], curr_pos) if len(chunk_text) >= 50 else page_text.find(chunk_text, curr_pos)
                start_c = found_idx if found_idx != -1 else curr_pos
                end_c = start_c + len(chunk_text)
                curr_pos = max(curr_pos, start_c + max(1, len(chunk_text) - chunk_overlap))

                chunk_meta = dict(page.metadata)
                chunk_meta.update({
                    "chunk_id": chunk_id,
                    "chunk_index": idx + 1,
                    "chunk_size_setting": chunk_size,
                    "chunk_overlap_setting": chunk_overlap,
                    "chunking_method": method,
                    "start_char": start_c,
                    "end_char": end_c,
                })

                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata=chunk_meta,
                        start_char=start_c,
                        end_char=end_c,
                    )
                )
        else:
            # Fixed-size chunking
            fixed_segments = split_fixed_size(page_text, chunk_size, chunk_overlap)
            for idx, (chunk_text, start_c, end_c) in enumerate(fixed_segments):
                chunk_id = f"{doc_id}_p{page_num}_fix_c{idx + 1}"
                chunk_meta = dict(page.metadata)
                chunk_meta.update({
                    "chunk_id": chunk_id,
                    "chunk_index": idx + 1,
                    "chunk_size_setting": chunk_size,
                    "chunk_overlap_setting": chunk_overlap,
                    "chunking_method": "fixed",
                    "start_char": start_c,
                    "end_char": end_c,
                })
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata=chunk_meta,
                        start_char=start_c,
                        end_char=end_c,
                    )
                )

    return all_chunks


def compute_chunk_statistics(chunks: List[Chunk]) -> Dict[str, Any]:
    """Computes descriptive statistical metrics across the chunk collection."""
    if not chunks:
        return {
            "total_chunks": 0,
            "min_chars": 0,
            "max_chars": 0,
            "mean_chars": 0.0,
            "median_chars": 0.0,
            "std_dev_chars": 0.0,
        }

    lengths = [c.char_count for c in chunks]
    return {
        "total_chunks": len(chunks),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "mean_chars": round(statistics.mean(lengths), 2),
        "median_chars": round(statistics.median(lengths), 2),
        "std_dev_chars": round(statistics.stdev(lengths), 2) if len(lengths) > 1 else 0.0,
    }


def find_overlap(text1: str, text2: str, max_lookback: int = 300) -> str:
    """Detects the exact overlapping suffix of text1 that forms the prefix of text2."""
    search_window = min(len(text1), max_lookback)
    tail = text1[-search_window:]
    for i in range(len(tail)):
        sub = tail[i:]
        if len(sub) >= 15 and text2.startswith(sub):
            return sub
    return ""


def extract_overlap_samples(chunks: List[Chunk], limit: int = 5) -> List[Dict[str, Any]]:
    """Identifies consecutive pairs of chunks within the same page/document

    and isolates their overlapping content for visual verification in the UI.
    """
    samples = []
    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]

        # Check if they belong to the same document and page
        if (
            c1.metadata.get("doc_id") == c2.metadata.get("doc_id")
            and c1.metadata.get("page_number") == c2.metadata.get("page_number")
        ):
            overlap_text = find_overlap(c1.text, c2.text)
            if overlap_text:
                samples.append({
                    "chunk1_id": c1.chunk_id,
                    "chunk2_id": c2.chunk_id,
                    "doc_title": c1.metadata.get("title"),
                    "page_number": c1.metadata.get("page_number"),
                    "chunk1_preview": c1.text[-120:],
                    "chunk2_preview": c2.text[:120],
                    "overlap_content": overlap_text,
                    "overlap_length": len(overlap_text),
                })
                if len(samples) >= limit:
                    break

    return samples
