"""Local retrieval-augmented knowledge assistant.

This namespace is intentionally separate from deterministic hospital ranking.
"""
from core.rag.pipeline import RAG_SAMPLE_QUESTIONS, answer_knowledge_question

__all__ = ["RAG_SAMPLE_QUESTIONS", "answer_knowledge_question"]
