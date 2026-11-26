"""GramHealth AI - Ingestion Pipeline Script.
Executes the complete document ingestion pipeline:
PDF extraction -> Text cleaning -> Recursive chunking -> Multilingual E5 embeddings -> FAISS Indexing -> Persistent Storage.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.chunking import chunk_documents, compute_chunk_statistics
from src.document_loader import load_all_documents
from src.embeddings import get_embedder
from src.retrieval import FAISSVectorStore


def run_ingestion(
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    method: str = "recursive",
    vectorstore_path: Path = config.VECTORSTORE_DIR,
):
    print("=" * 70)
    print("GramHealth AI — Healthcare Knowledge Ingestion Pipeline")
    print("=" * 70)
    start_time = time.time()

    # Step 1: Document Loading
    print(f"\n[1/5] Loading official healthcare PDFs from {config.DOCUMENTS_DIR}...")
    pages = load_all_documents(config.DOCUMENTS_DIR, config.SOURCES_JSON)
    print(f"      Loaded {len(pages)} valid pages from documents.")

    if not pages:
        print("[ERROR] No valid document pages found to index. Exiting.")
        return

    # Step 2: Recursive Text Chunking
    print(
        f"\n[2/5] Chunking documents (Method: {method}, Size: {chunk_size}, Overlap: {chunk_overlap})..."
    )
    chunks = chunk_documents(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        method=method,
        separators=config.CHUNK_SEPARATORS,
    )
    stats = compute_chunk_statistics(chunks)
    print(f"      Generated {stats['total_chunks']} chunks.")
    print(
        f"      Length stats: min={stats['min_chars']}, max={stats['max_chars']}, "
        f"mean={stats['mean_chars']}, median={stats['median_chars']}, std_dev={stats['std_dev_chars']}"
    )

    # Step 3: Dense Multilingual Embedding Generation
    print(f"\n[3/5] Generating multilingual dense embeddings with {config.EMBEDDING_MODEL_NAME}...")
    embedder = get_embedder()
    texts = [c.text for c in chunks]
    emb_start = time.time()
    embeddings = embedder.embed_passages(texts, show_progress=True)
    emb_duration = time.time() - emb_start
    print(
        f"      Embedded {len(embeddings)} passages in {emb_duration:.2f}s "
        f"({len(embeddings) / max(0.01, emb_duration):.1f} chunks/sec)."
    )
    print(f"      Embedding matrix shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Step 4: FAISS Vector Store Building
    print("\n[4/5] Building FAISS IndexFlatIP (Cosine Similarity) index...")
    vector_store = FAISSVectorStore(dimension=config.EMBEDDING_DIM)
    vector_store.add_chunks(chunks, embeddings=embeddings)
    print(f"      Added {len(vector_store)} vectors to FAISS index.")

    # Step 5: Persistent Local Storage
    print(f"\n[5/5] Persisting vector index to {vectorstore_path}...")
    vector_store.save(vectorstore_path)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Ingestion completed successfully in {total_time:.2f} seconds!")
    print(f"Total Pages Indexed: {len(pages)}")
    print(f"Total Chunks Created: {len(chunks)}")
    print(f"Vector Database Path: {vectorstore_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_ingestion()
