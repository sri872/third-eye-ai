"""End-to-end local RAG pipeline for emergency-response knowledge questions."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.rag.generator import generate_grounded_answer
from core.rag.retriever import LocalTfidfRetriever

RAG_SAMPLE_QUESTIONS = (
    "What does the Indian Good Samaritan guidance say about helping a road-crash victim?",
    "What protections are described for a Good Samaritan at a hospital?",
    "What is the difference between India's 108 and 102 ambulance services?",
    "What does the WHO Global Plan recommend for post-crash emergency response?",
    "What does the Motor Vehicles Act say about Good Samaritan protection?",
)

DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "knowledge_base"
    / "emergency_response_india.json"
)


@lru_cache(maxsize=4)
def _load_retriever_cached(path: str, modified_ns: int, size: int) -> LocalTfidfRetriever:
    del modified_ns, size  # included in cache key to invalidate on corpus changes
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported RAG corpus version.")
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("The RAG corpus must contain a non-empty documents list.")
    return LocalTfidfRetriever(documents)


def load_rag_retriever(corpus_path: str | Path | None = None) -> LocalTfidfRetriever:
    path = Path(corpus_path or DEFAULT_CORPUS_PATH).resolve()
    stat = path.stat()
    return _load_retriever_cached(str(path), stat.st_mtime_ns, stat.st_size)


def answer_knowledge_question(
    query: str,
    *,
    top_k: int = 3,
    corpus_path: str | Path | None = None,
) -> dict[str, Any]:
    """Retrieve, synthesize, and cite source passages for a bounded knowledge query."""
    question = str(query or "").strip()
    if not question:
        raise ValueError("Select or enter a knowledge question.")
    if len(question) > 500:
        raise ValueError("Knowledge question must be 500 characters or fewer.")

    retriever = load_rag_retriever(corpus_path)
    passages = [p.as_dict() for p in retriever.retrieve(question, top_k=top_k)]
    generated = generate_grounded_answer(question, passages)

    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for passage in passages:
        source_id = passage["source_id"]
        if source_id in seen:
            continue
        seen.add(source_id)
        sources.append(
            {
                "source_id": source_id,
                "title": passage["title"],
                "agency": passage["agency"],
                "jurisdiction": passage["jurisdiction"],
                "url": passage["url"],
            }
        )

    return {
        "question": question,
        "answer": generated["answer"],
        "generation_mode": generated["generation_mode"],
        "model": generated["model"],
        "retrieved_passages": passages,
        "sources": sources,
        "retrieval_method": "local TF-IDF vector retrieval (scikit-learn)",
        "source_count": len(sources),
        "human_review_note": (
            "Reference information only. Not medical advice, legal advice, hospital-capacity "
            "verification, an emergency assessment, or a dispatch instruction. For immediate "
            "danger, contact the appropriate official emergency service directly."
        ),
    }
