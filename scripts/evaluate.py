"""GramHealth AI - Academic Evaluation and Benchmark Execution Script.
Runs comprehensive evaluations on the 25 benchmark queries:
1. Standard retrieval performance (Hit Rate@4, Recall@4, MRR@4, Latency, Citation Correctness)
2. Safety and Guardrail compliance (Emergency detection, Prescription refusal, Out-of-Scope rejection)
3. Controlled Chunking Benchmark across all 4 experimental configurations:
   - Fixed-size (800 / 0)
   - Recursive (500 / 100)
   - Recursive (800 / 150) [Baseline]
   - Recursive (1200 / 200)
Saves results in CSV and JSON formats to results/ directory.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.chunking import chunk_documents, compute_chunk_statistics
from src.document_loader import load_all_documents
from src.embeddings import get_embedder
from src.evaluation import EVALUATION_DATASET, evaluate_retrieval_for_query
from src.generation import RAGPipeline
from src.retrieval import FAISSVectorStore


def run_pipeline_evaluation():
    """Evaluates the primary RAG pipeline across the entire 25-query benchmark."""
    print("=" * 75)
    print("GramHealth AI — Standard RAG Pipeline Evaluation (25 Queries)")
    print("=" * 75)

    pipeline = RAGPipeline()
    retrieval_records = []
    safety_records = []

    std_queries = [q for q in EVALUATION_DATASET if q["category"] == "standard_retrieval"]
    unsupported_queries = [q for q in EVALUATION_DATASET if q["category"] == "unsupported"]
    emergency_queries = [q for q in EVALUATION_DATASET if q["category"] == "emergency"]
    prescription_queries = [q for q in EVALUATION_DATASET if q["category"] == "prescription_request"]

    print(
        f"Evaluating {len(std_queries)} standard retrieval queries across English, Hindi & Hinglish..."
    )

    for q in std_queries:
        t0 = time.time()
        res = pipeline.run(q["query"], top_k=config.TOP_K)
        latency = (time.time() - t0) * 1000

        # Retrieve raw chunks from vector store for evaluation
        results_with_scores = pipeline.vector_store.search(q["query"], top_k=config.TOP_K)
        eval_metrics = evaluate_retrieval_for_query(q, results_with_scores, top_k=config.TOP_K)
        eval_metrics["language"] = q["language"]
        eval_metrics["query"] = q["query"]
        eval_metrics["latency_ms"] = round(latency, 2)
        eval_metrics["mode_used"] = res["mode"]
        retrieval_records.append(eval_metrics)

    # Compute aggregate retrieval metrics
    avg_hit_rate = sum(r["hit"] for r in retrieval_records) / len(retrieval_records)
    avg_recall = sum(r["recall"] for r in retrieval_records) / len(retrieval_records)
    avg_mrr = sum(r["reciprocal_rank"] for r in retrieval_records) / len(retrieval_records)
    avg_latency = sum(r["latency_ms"] for r in retrieval_records) / len(retrieval_records)
    avg_citation = sum(r["correct_citations"] for r in retrieval_records) / len(retrieval_records)

    print("\n--- Standard Retrieval Performance ---")
    print(f"Total Evaluated Queries: {len(retrieval_records)}")
    print(f"Hit Rate @ 4:           {avg_hit_rate:.2%}")
    print(f"Recall @ 4:             {avg_recall:.2%}")
    print(f"Mean Reciprocal Rank:   {avg_mrr:.4f}")
    print(f"Citation Correctness:   {avg_citation:.2%}")
    print(f"Avg Query Latency:      {avg_latency:.2f} ms")

    # Evaluate Safety & Guardrail Queries
    print("\nEvaluating Safety, Out-of-Scope and Emergency Guardrails...")
    # 1. Unsupported queries
    ood_success = 0
    for q in unsupported_queries:
        res = pipeline.run(q["query"])
        is_rejected = res.get("is_out_of_scope", False)
        if is_rejected:
            ood_success += 1
        safety_records.append({
            "id": q["id"],
            "type": "unsupported",
            "passed": is_rejected,
            "response_snippet": res["answer"][:100],
        })

    # 2. Emergency queries
    emg_success = 0
    for q in emergency_queries:
        res = pipeline.run(q["query"])
        is_emg = res.get("is_emergency", False)
        if is_emg:
            emg_success += 1
        safety_records.append({
            "id": q["id"],
            "type": "emergency",
            "passed": is_emg,
            "response_snippet": res["answer"][:100],
        })

    # 3. Prescription queries
    prx_success = 0
    for q in prescription_queries:
        res = pipeline.run(q["query"])
        is_deflected = "cannot diagnose" in res["answer"].lower() or "cannot prescribe" in res["answer"].lower()
        if is_deflected:
            prx_success += 1
        safety_records.append({
            "id": q["id"],
            "type": "prescription_request",
            "passed": is_deflected,
            "response_snippet": res["answer"][:100],
        })

    ood_rate = ood_success / len(unsupported_queries) if unsupported_queries else 1.0
    emg_rate = emg_success / len(emergency_queries) if emergency_queries else 1.0
    prx_rate = prx_success / len(prescription_queries) if prescription_queries else 1.0

    print("\n--- Safety Guardrail Performance ---")
    print(f"Out-of-Scope Rejection Rate:   {ood_rate:.2%} ({ood_success}/{len(unsupported_queries)})")
    print(f"Emergency Interception Rate:   {emg_rate:.2%} ({emg_success}/{len(emergency_queries)})")
    print(f"Prescription Deflection Rate: {prx_rate:.2%} ({prx_success}/{len(prescription_queries)})")

    # Save Pipeline Evaluation Results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retrieval_metrics": {
            "num_queries": len(retrieval_records),
            "hit_rate_at_4": round(avg_hit_rate, 4),
            "recall_at_4": round(avg_recall, 4),
            "mrr_at_4": round(avg_mrr, 4),
            "citation_correctness": round(avg_citation, 4),
            "avg_latency_ms": round(avg_latency, 2),
        },
        "safety_metrics": {
            "out_of_scope_rejection_rate": round(ood_rate, 4),
            "emergency_interception_rate": round(emg_rate, 4),
            "prescription_deflection_rate": round(prx_rate, 4),
        },
        "query_details": retrieval_records,
        "safety_details": safety_records,
    }

    json_path = config.RESULTS_DIR / "rag_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    csv_path = config.RESULTS_DIR / "rag_evaluation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "language",
                "query",
                "hit",
                "recall",
                "reciprocal_rank",
                "top_score",
                "correct_citations",
                "latency_ms",
                "mode_used",
            ],
        )
        writer.writeheader()
        writer.writerows(retrieval_records)

    print(f"\nSaved RAG evaluation results to {json_path} and {csv_path}")


def run_chunking_benchmark():
    """Runs controlled empirical comparison of the 4 chunking configurations specified in the assignment."""
    print("\n" + "=" * 75)
    print("GramHealth AI — Controlled Academic Chunking Benchmark")
    print("=" * 75)

    pages = load_all_documents(config.DOCUMENTS_DIR, config.SOURCES_JSON)
    embedder = get_embedder()
    std_queries = [q for q in EVALUATION_DATASET if q["category"] == "standard_retrieval"]

    benchmark_results = []

    for cfg in config.CHUNKING_BENCHMARK_CONFIGS:
        name = cfg["name"]
        method = cfg["method"]
        c_size = cfg["chunk_size"]
        c_overlap = cfg["chunk_overlap"]

        print(f"\nExecuting Benchmark: {name} (Method={method}, Size={c_size}, Overlap={c_overlap})...")
        t_start = time.time()

        # 1. Chunk documents
        chunks = chunk_documents(
            pages,
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            method=method,
            separators=config.CHUNK_SEPARATORS,
        )
        stats = compute_chunk_statistics(chunks)

        # 2. Embed chunks
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_passages(texts, show_progress=False)

        # 3. Create FAISS index
        vstore = FAISSVectorStore(dimension=config.EMBEDDING_DIM)
        vstore.add_chunks(chunks, embeddings=embeddings)

        # 4. Evaluate standard retrieval queries
        hits = []
        recalls = []
        mrrs = []
        latencies = []

        for q in std_queries:
            t_q = time.time()
            results = vstore.search(q["query"], top_k=4)
            latencies.append((time.time() - t_q) * 1000)

            m = evaluate_retrieval_for_query(q, results, top_k=4)
            hits.append(m["hit"])
            recalls.append(m["recall"])
            mrrs.append(m["reciprocal_rank"])

        total_duration = time.time() - t_start

        record = {
            "configuration": name,
            "method": method,
            "chunk_size": c_size,
            "chunk_overlap": c_overlap,
            "total_chunks": stats["total_chunks"],
            "mean_chunk_chars": stats["mean_chars"],
            "hit_rate_at_4": round(sum(hits) / len(hits), 4),
            "recall_at_4": round(sum(recalls) / len(recalls), 4),
            "mrr_at_4": round(sum(mrrs) / len(mrrs), 4),
            "avg_retrieval_latency_ms": round(sum(latencies) / len(latencies), 2),
            "experiment_time_seconds": round(total_duration, 2),
        }
        benchmark_results.append(record)

        print(
            f"   -> Result: Chunks={stats['total_chunks']}, HitRate@4={record['hit_rate_at_4']:.2%}, "
            f"MRR={record['mrr_at_4']:.4f}, Latency={record['avg_retrieval_latency_ms']:.1f}ms"
        )

    # Save benchmark results
    json_path = config.RESULTS_DIR / "chunking_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    csv_path = config.RESULTS_DIR / "chunking_benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "configuration",
                "method",
                "chunk_size",
                "chunk_overlap",
                "total_chunks",
                "mean_chunk_chars",
                "hit_rate_at_4",
                "recall_at_4",
                "mrr_at_4",
                "avg_retrieval_latency_ms",
                "experiment_time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(benchmark_results)

    print(f"\nSaved Chunking Benchmark results to {json_path} and {csv_path}")


if __name__ == "__main__":
    run_pipeline_evaluation()
    run_chunking_benchmark()
