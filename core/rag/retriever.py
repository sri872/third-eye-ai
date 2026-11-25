"""Local TF-IDF retrieval over a curated, source-attributed document corpus.

Hospital eligibility/ranking does not use this index. The retriever is only for
knowledge questions that can be answered from the bundled source documents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RetrievedPassage:
    chunk_id: str
    source_id: str
    title: str
    agency: str
    url: str
    jurisdiction: str
    text: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "title": self.title,
            "agency": self.agency,
            "url": self.url,
            "jurisdiction": self.jurisdiction,
            "text": self.text,
            "score": round(float(self.score), 4),
        }


class LocalTfidfRetriever:
    """Build a deterministic local index and return the most relevant passages."""

    def __init__(self, documents: Iterable[dict[str, Any]]) -> None:
        self._chunks: list[dict[str, Any]] = []
        for document in documents:
            common = {
                "source_id": str(document["source_id"]),
                "title": str(document["title"]),
                "agency": str(document["agency"]),
                "url": str(document["url"]),
                "jurisdiction": str(document.get("jurisdiction", "not specified")),
            }
            for chunk in document.get("chunks", []):
                text = str(chunk.get("text", "")).strip()
                if text:
                    self._chunks.append(
                        {
                            **common,
                            "chunk_id": str(chunk.get("chunk_id", f"{common['source_id']}-{len(self._chunks) + 1}")),
                            "text": text,
                        }
                    )

        if not self._chunks:
            raise ValueError("The RAG knowledge corpus contains no non-empty passages.")

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            dtype=np.float32,
        )
        self._matrix = self._vectorizer.fit_transform(
            [chunk["text"] for chunk in self._chunks]
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPassage]:
        """Retrieve relevant passages; omit zero/near-zero lexical matches."""
        query = str(query or "").strip()
        if not query:
            return []
        if len(query) > 500:
            raise ValueError("Knowledge question must be 500 characters or fewer.")
        if top_k < 1 or top_k > 10:
            raise ValueError("top_k must be between 1 and 10.")

        query_vector = self._vectorizer.transform([query])
        if query_vector.nnz == 0:
            return []

        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked_indices = np.argsort(scores)[::-1]
        matches: list[RetrievedPassage] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score < 0.035:
                break
            chunk = self._chunks[int(index)]
            matches.append(
                RetrievedPassage(
                    chunk_id=chunk["chunk_id"],
                    source_id=chunk["source_id"],
                    title=chunk["title"],
                    agency=chunk["agency"],
                    url=chunk["url"],
                    jurisdiction=chunk["jurisdiction"],
                    text=chunk["text"],
                    score=score,
                )
            )
            if len(matches) >= top_k:
                break
        return matches
