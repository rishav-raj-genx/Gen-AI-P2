"""GramHealth AI - Generation and RAG Pipeline Module.
Implements:
1. Context construction from retrieved FAISS chunks
2. Ollama Llama 3.2 3B integration with strict grounding and citation instructions
3. Explicitly labeled 'Retrieval-Only Mode' when local LLM is offline
4. Latency measurement and citation metadata tracking
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

import config
from src.chunking import Chunk
from src.retrieval import FAISSVectorStore, get_default_vector_store
from src.safety import (
    check_diagnosis_or_prescription,
    check_emergency,
    format_insufficient_info_response,
    is_insufficient_information,
    sanitize_input,
)


def check_ollama_availability(
    base_url: str = config.OLLAMA_BASE_URL,
    model_name: str = config.OLLAMA_MODEL,
) -> Tuple[bool, str]:
    """Checks whether the Ollama server is reachable and whether the target model is installed."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Match base model name or tagged name
            matched = any(
                model_name in m or m.startswith(model_name.split(":")[0]) for m in models
            )
            if matched:
                return True, f"Ollama is running with {model_name} available."
            else:
                available_str = ", ".join(models) if models else "None"
                return (
                    False,
                    f"Ollama is running, but model '{model_name}' was not found. Available models: {available_str}",
                )
        return False, f"Ollama returned HTTP status {resp.status_code}."
    except requests.exceptions.RequestException as e:
        return False, f"Ollama server not reachable at {base_url} ({e.__class__.__name__})."


def construct_context(chunks_with_scores: List[Tuple[Chunk, float]]) -> str:
    """Formats retrieved chunks into a clean, structured context block with clear citation markers."""
    context_blocks = []
    for idx, (chunk, score) in enumerate(chunks_with_scores, start=1):
        doc_title = chunk.metadata.get("title", "Unknown Document")
        page_num = chunk.metadata.get("page_number", "?")
        org = chunk.metadata.get("organization", "Healthcare Organization")
        verified = "Verified Official" if chunk.metadata.get("verified", True) else "User Uploaded (Unverified)"

        header = f"[Source {idx} | Doc: '{doc_title}' | Page {page_num} | Org: {org} | Status: {verified} | Similarity: {score:.2f}]"
        block = f"{header}\n{chunk.text}\n"
        context_blocks.append(block)

    return "\n---\n".join(context_blocks)


def call_ollama_llm(
    prompt: str,
    base_url: str = config.OLLAMA_BASE_URL,
    model_name: str = config.OLLAMA_MODEL,
    timeout: int = config.OLLAMA_REQUEST_TIMEOUT,
) -> str:
    """Invokes the local Ollama LLM with streaming disabled for reliable single-turn response."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,  # Low temperature to minimize hallucination
            "top_p": 0.9,
            "num_predict": 750,
        },
    }
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def synthesize_retrieval_only(
    query: str,
    chunks_with_scores: List[Tuple[Chunk, float]],
) -> str:
    """Generates an educational, citation-backed response strictly extracted

    from the top retrieved chunks when the local LLM is offline.
    Explicitly labeled as 'Retrieval-Only Mode'.
    """
    if not chunks_with_scores:
        return "No relevant healthcare documents were found for this query."

    lines = [
        "### 📋 [Retrieval-Only Mode — Grounded Evidence Synthesis]\n",
        "> *Note: Local LLM (Ollama) is offline or not installed. GramHealth AI is operating in "
        "**Retrieval-Only Mode**, displaying direct evidence extracted from official healthcare documents.*",
        "\n**Key Medical Guidance Identified:**\n",
    ]

    # Extract primary statements from top chunks, prioritizing sentences matching query keywords
    query_words = set([w.lower() for w in re.findall(r"\w{3,}", query)])

    for idx, (chunk, score) in enumerate(chunks_with_scores[:3], start=1):
        doc_title = chunk.metadata.get("title", "Document")
        page_num = chunk.metadata.get("page_number", "N/A")
        org = chunk.metadata.get("organization", "Health Agency")

        raw_sentences = [s.strip() for s in chunk.text.replace("\n", " ").split(". ") if len(s.strip()) > 30]
        
        # Sort sentences by keyword relevance to query
        def sentence_score(s):
            s_words = set(re.findall(r"\w{3,}", s.lower()))
            return len(s_words.intersection(query_words))
        
        ranked_sentences = sorted(raw_sentences, key=sentence_score, reverse=True)
        # Select top 2-3 most relevant sentences, preserving their original order in text
        top_selected = ranked_sentences[:3] if ranked_sentences else []
        ordered_sentences = [s for s in raw_sentences if s in top_selected]

        highlight = ". ".join(ordered_sentences) if ordered_sentences else chunk.text[:350]
        if highlight and not highlight.endswith("."):
            highlight += "."

        lines.append(
            f"**{idx}. Evidence from {doc_title} (Page {page_num}, {org} — Relevance: {score:.1%}):**\n"
            f"> \"{highlight}\"\n"
        )

    lines.append(
        "\n**Recommended Action for Rural Communities:**\n"
        "• Review the exact citations and supporting excerpts below.\n"
        "• For personalized advice, consult your village **ASHA**, the **ANM** at the nearest Sub-Health Centre, "
        "or a Medical Officer at the Primary Health Centre (PHC)."
    )

    return "\n".join(lines)


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(self, vector_store: Optional[FAISSVectorStore] = None):
        self.vector_store = vector_store or get_default_vector_store()
        self.ollama_available, self.ollama_msg = check_ollama_availability()

    def run(
        self,
        query: str,
        top_k: int = config.TOP_K,
        force_retrieval_only: bool = False,
    ) -> Dict[str, Any]:
        """Executes the full RAG pipeline for a user query."""
        start_time = time.time()
        retrieval_time = 0.0
        generation_time = 0.0

        clean_query = sanitize_input(query)

        # 1. Emergency Safety Check
        is_emg, emg_response = check_emergency(clean_query)
        if is_emg:
            return {
                "query": query,
                "answer": emg_response,
                "is_emergency": True,
                "is_out_of_scope": False,
                "mode": "Emergency Guardrail",
                "retrieved_chunks": [],
                "citations": [],
                "context": "",
                "total_latency_seconds": round(time.time() - start_time, 3),
                "retrieval_latency_seconds": 0.0,
                "generation_latency_seconds": 0.0,
            }

        # 2. Prescription/Diagnosis Request Check
        is_diag, diag_response = check_diagnosis_or_prescription(clean_query)
        if is_diag:
            return {
                "query": query,
                "answer": diag_response,
                "is_emergency": False,
                "is_out_of_scope": False,
                "mode": "Safety Disclaimer",
                "retrieved_chunks": [],
                "citations": [],
                "context": "",
                "total_latency_seconds": round(time.time() - start_time, 3),
                "retrieval_latency_seconds": 0.0,
                "generation_latency_seconds": 0.0,
            }

        # 3. Vector Retrieval
        if self.vector_store is None or len(self.vector_store) == 0:
            return {
                "query": query,
                "answer": "⚠️ The vector knowledge base is empty. Please run ingestion to index documents.",
                "is_emergency": False,
                "is_out_of_scope": True,
                "mode": "Unindexed",
                "retrieved_chunks": [],
                "citations": [],
                "context": "",
                "total_latency_seconds": round(time.time() - start_time, 3),
                "retrieval_latency_seconds": 0.0,
                "generation_latency_seconds": 0.0,
            }

        ret_start = time.time()
        results = self.vector_store.search(clean_query, top_k=top_k)
        retrieval_time = time.time() - ret_start

        if not results:
            return {
                "query": query,
                "answer": format_insufficient_info_response(clean_query, 0.0),
                "is_emergency": False,
                "is_out_of_scope": True,
                "mode": "Insufficient Information",
                "retrieved_chunks": [],
                "citations": [],
                "context": "",
                "total_latency_seconds": round(time.time() - start_time, 3),
                "retrieval_latency_seconds": round(retrieval_time, 3),
                "generation_latency_seconds": 0.0,
            }

        max_score = results[0][1]

        # 4. Out-of-Scope / Low Confidence Detection
        if is_insufficient_information(max_score):
            return {
                "query": query,
                "answer": format_insufficient_info_response(clean_query, max_score),
                "is_emergency": False,
                "is_out_of_scope": True,
                "mode": "Insufficient Information",
                "retrieved_chunks": [
                    {"chunk": c.to_dict(), "similarity_score": round(s, 4)}
                    for c, s in results
                ],
                "citations": [],
                "context": construct_context(results),
                "total_latency_seconds": round(time.time() - start_time, 3),
                "retrieval_latency_seconds": round(retrieval_time, 3),
                "generation_latency_seconds": 0.0,
            }

        # 5. Build Context and Citations
        context_str = construct_context(results)
        citations = []
        for c, s in results:
            citations.append({
                "doc_title": c.metadata.get("title"),
                "filename": c.metadata.get("filename"),
                "page_number": c.metadata.get("page_number"),
                "organization": c.metadata.get("organization"),
                "original_url": c.metadata.get("original_url"),
                "similarity_score": round(s, 4),
                "excerpt": c.text[:220] + "..." if len(c.text) > 220 else c.text,
                "chunk_id": c.chunk_id,
                "verified": c.metadata.get("verified", True),
            })

        # 6. Answer Generation
        # Re-check Ollama status dynamically
        self.ollama_available, self.ollama_msg = check_ollama_availability()
        use_ollama = self.ollama_available and not force_retrieval_only

        gen_start = time.time()
        mode_used = "Ollama Llama 3.2 3B" if use_ollama else "Retrieval-Only Mode"

        if use_ollama:
            prompt = (
                "You are GramHealth AI, an evidence-based rural healthcare educational assistant.\n"
                "Answer the user's question accurately, empathetically, and simply for a rural community audience.\n\n"
                "STRICT INSTRUCTIONS:\n"
                "1. Base your answer EXCLUSIVELY on the provided [Healthcare Context]. Do NOT assume or invent information.\n"
                "2. For every factual recommendation or guideline, explicitly cite the document and page using [Doc: <Title>, Page <Page>].\n"
                "3. If the user asks in Hindi or Hinglish, answer in fluent, respectful Hindi/Hinglish as appropriate.\n"
                "4. Do NOT prescribe medications, specify commercial drug brand names, or diagnose conditions.\n"
                "5. Remind the user to seek in-person consultation with local ASHA/ANM workers or PHC doctors.\n\n"
                f"[Healthcare Context]:\n{context_str}\n\n"
                f"User Question: {clean_query}\n\n"
                "Answer:"
            )
            try:
                answer = call_ollama_llm(prompt)
            except Exception as e:
                # Graceful fallback to retrieval-only mode if LLM fails
                mode_used = "Retrieval-Only Mode (Ollama Call Failed)"
                answer = synthesize_retrieval_only(clean_query, results)
        else:
            answer = synthesize_retrieval_only(clean_query, results)

        generation_time = time.time() - gen_start
        total_time = time.time() - start_time

        return {
            "query": query,
            "answer": answer,
            "is_emergency": False,
            "is_out_of_scope": False,
            "mode": mode_used,
            "retrieved_chunks": [
                {"chunk": c.to_dict(), "similarity_score": round(s, 4)}
                for c, s in results
            ],
            "citations": citations,
            "context": context_str,
            "total_latency_seconds": round(total_time, 3),
            "retrieval_latency_seconds": round(retrieval_time, 3),
            "generation_latency_seconds": round(generation_time, 3),
        }
