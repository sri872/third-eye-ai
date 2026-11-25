"""Tests for the source-grounded emergency-response RAG assistant."""

from __future__ import annotations

import json

import pytest

from core.rag.generator import generate_grounded_answer
from core.rag.pipeline import (
    DEFAULT_CORPUS_PATH,
    RAG_SAMPLE_QUESTIONS,
    answer_knowledge_question,
    load_rag_retriever,
)
from core.rag.retriever import LocalTfidfRetriever


def test_curated_corpus_loads_and_retrieves_good_samaritan_source():
    retriever = load_rag_retriever()
    matches = retriever.retrieve("What rights does a Good Samaritan have at a hospital?", top_k=3)

    assert matches
    assert matches[0].source_id == "MORTH_GOOD_SAMARITAN_GUIDE"
    assert "personal details" in matches[0].text.lower() or "hospital" in matches[0].text.lower()
    assert matches[0].url.startswith("https://morth.nic.in/")


def test_nhm_108_and_102_question_retrieves_nhm_source():
    matches = load_rag_retriever().retrieve(
        "What is the difference between 108 emergency response and 102 patient transport?",
        top_k=4,
    )
    assert any(match.source_id == "NHM_AMBULANCE_108_102" for match in matches)


def test_retriever_rejects_empty_and_overlong_queries():
    retriever = load_rag_retriever()
    assert retriever.retrieve("   ") == []
    with pytest.raises(ValueError, match="500 characters"):
        retriever.retrieve("x" * 501)


def test_retriever_returns_no_passages_for_unrelated_question():
    retriever = LocalTfidfRetriever(
        [
            {
                "source_id": "road",
                "title": "Road response",
                "agency": "Example source",
                "url": "https://example.org/road",
                "chunks": [{"chunk_id": "road-1", "text": "Good Samaritans may help road crash victims."}],
            }
        ]
    )
    assert retriever.retrieve("Explain quantum chromodynamics and quark confinement.") == []


def test_answer_pipeline_cites_retrieved_sources_and_uses_explicit_fallback(monkeypatch):
    monkeypatch.delenv("THIRD_EYE_RAG_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("THIRD_EYE_RAG_OLLAMA_HOST", raising=False)

    result = answer_knowledge_question(RAG_SAMPLE_QUESTIONS[0], top_k=3)

    assert result["retrieval_method"].startswith("local TF-IDF")
    assert result["retrieved_passages"]
    assert result["sources"]
    assert "EXTRACTIVE RAG" in result["generation_mode"]
    assert "[1]" in result["answer"]
    assert all(source["url"].startswith("https://") for source in result["sources"])
    assert "not medical advice" in result["human_review_note"].lower()


def test_no_matching_passage_does_not_generate_an_unsupported_answer():
    result = generate_grounded_answer(
        "Explain quantum chromodynamics and quark confinement.",
        [],
    )
    assert result["generation_mode"] == "NO SUPPORTED SOURCE FOUND"
    assert "do not contain enough relevant information" in result["answer"]


def test_ollama_is_optional_and_fallback_is_not_mislabelled(monkeypatch):
    monkeypatch.setenv("THIRD_EYE_RAG_OLLAMA_MODEL", "model-not-configured")
    monkeypatch.setenv("THIRD_EYE_RAG_OLLAMA_HOST", "http://127.0.0.1:1")
    result = generate_grounded_answer(
        "What is the Good Samaritan protection?",
        [
            {
                "text": "The cited source describes a Good Samaritan protection.",
                "title": "Source title",
                "agency": "Official agency",
                "url": "https://example.org/source",
            }
        ],
    )
    assert result["generation_mode"] == "EXTRACTIVE FALLBACK — LOCAL OLLAMA UNAVAILABLE"
    assert "cited source" in result["answer"]


def test_rag_corpus_is_versioned_and_has_source_metadata():
    corpus = json.loads(DEFAULT_CORPUS_PATH.read_text(encoding="utf-8"))
    assert corpus["version"] == 1
    assert len(corpus["documents"]) >= 3
    for document in corpus["documents"]:
        assert document["source_id"]
        assert document["title"]
        assert document["agency"]
        assert document["url"].startswith("https://")
        assert document["chunks"]
        assert all(chunk["text"].strip() for chunk in document["chunks"])



def test_hospitals_tab_exposes_rag_question_dropdown_and_answer(monkeypatch):
    """The knowledge assistant is available without a media upload and preserves six tabs."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("THIRD_EYE_RAG_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("THIRD_EYE_RAG_OLLAMA_HOST", raising=False)
    app_file = Path(__file__).resolve().parents[1] / "app" / "main.py"
    at = AppTest.from_file(str(app_file), default_timeout=300).run()
    assert not at.exception, [item.value for item in at.exception]
    assert len(at.tabs) == 6

    question_select = next(
        item for item in at.tabs[4].selectbox
        if item.label == "Emergency knowledge question"
    )
    assert question_select.options[0] == "Choose a question…"
    question_select.select(RAG_SAMPLE_QUESTIONS[2]).run()

    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]
    tab_text = " ".join(
        node.value
        for kind in ("markdown", "caption", "info", "text")
        for node in at.tabs[4].get(kind)
    )
    assert "EXTRACTIVE RAG" in tab_text
    assert "Retrieved evidence" in tab_text
    assert "National Health Mission" in tab_text
