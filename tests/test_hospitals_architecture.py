"""Architecture regressions for deterministic hospital ranking and separate knowledge RAG."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("app", "core", "db", "scripts")
FORBIDDEN_HOSPITAL_LEGACY_TOKENS = (
    "HospitalIndex",
    "HospitalRetriever",
    "get_embedding_model",
    "build_hospital_profile_text",
    "formulate_retrieval_query",
    "retrieve_suggestions",
    "query_override",
    "semantic_score",
    "score_hospital_candidate",
    "faiss-cpu",
    "faiss",
    "sentence-transformers",
    "sentence_transformers",
    "SentenceTransformer",
)


def _active_source_files():
    for directory in SOURCE_DIRS:
        yield from (ROOT / directory).rglob("*.py")


def _from_imports(source: str) -> set[tuple[str, str]]:
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.add((module, alias.name))
    return imports


def test_knowledge_rag_namespace_exists_as_a_separate_feature():
    package = importlib.import_module("core.rag")
    pipeline = importlib.import_module("core.rag.pipeline")
    assert hasattr(pipeline, "answer_knowledge_question")
    assert hasattr(pipeline, "RAG_SAMPLE_QUESTIONS")
    assert callable(package.answer_knowledge_question)


def test_hospital_ranking_does_not_import_or_depend_on_knowledge_rag():
    hospital_files = list((ROOT / "core" / "hospitals").glob("*.py"))
    assert hospital_files
    for path in hospital_files:
        source = path.read_text(encoding="utf-8")
        assert "core.rag" not in source
        assert "sentence_transformers" not in source.lower()
        assert "faiss" not in source.lower()


def test_new_hospital_package_exposes_shared_public_api():
    package = importlib.import_module("core.hospitals")
    assert hasattr(package, "HospitalCatalog")
    assert hasattr(package, "HospitalRanker")
    assert hasattr(package, "classify_hospital_record")
    assert hasattr(package, "dispatch_eligible_facilities")


def test_removed_hospital_retrieval_apis_and_dependencies_cannot_reappear():
    violations = []
    for path in _active_source_files():
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        for token in FORBIDDEN_HOSPITAL_LEGACY_TOKENS:
            if token.lower() in lowered:
                violations.append(f"{path.relative_to(ROOT)}: {token}")
    assert violations == [], "Removed hospital retrieval surfaces reintroduced: " + ", ".join(violations)


def test_removed_vector_dependencies_are_absent_from_requirements():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "faiss" not in requirements
    assert "sentence-transformers" not in requirements
    assert "sentence_transformers" not in requirements
    assert "scikit-learn" in requirements


def test_app_and_evaluator_share_the_one_hospital_ranker():
    app_path = ROOT / "app" / "main.py"
    evaluator_path = ROOT / "scripts" / "evaluate_hospital_ranking.py"
    app_source = app_path.read_text(encoding="utf-8")
    evaluator_source = evaluator_path.read_text(encoding="utf-8")

    expected_import = ("core.hospitals.ranking", "HospitalRanker")
    assert expected_import in _from_imports(app_source)
    assert expected_import in _from_imports(evaluator_source)
    assert "ranker.rank_suggestions(" in app_source
    assert "ranker.rank_suggestions(" in evaluator_source
    assert "def rank_suggestions" not in evaluator_source
    assert "def haversine_distance_km" not in evaluator_source
