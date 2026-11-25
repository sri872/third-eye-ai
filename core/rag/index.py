"""FAISS semantic index for OpenStreetMap hospital profiles.

Uses sentence-transformers ('all-MiniLM-L6-v2') to encode rich clinical and facility
profiles into 384-dimensional dense vectors indexed by FAISS.
Lazy-loads embedding models once for zero redundant overhead.
"""

import os
import sys

# Prevent OpenMP library runtime conflict between PyTorch and FAISS on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Ensure huggingface_hub uses local cache immediately without network retry delays
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path
import threading
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from core.schemas import Hospital

_MODEL = None
_MODEL_LOCK = threading.Lock()
MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_model():
    """Lazy load SentenceTransformer model singleton."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer
                _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def build_hospital_profile_text(hospital: Hospital) -> str:
    """Build structured semantic description of hospital for dense vector indexing."""
    parts = [f"Facility: {hospital.name}."]
    if hospital.address:
        parts.append(f"Location: {hospital.address}.")
    if hospital.specialties:
        parts.append(f"Medical specialties: {hospital.specialties}.")
    else:
        parts.append("Specialties: General hospital / clinic services.")
    
    if hospital.emergency == "yes":
        parts.append(
            "Emergency services: emergency department reported in OSM tags; "
            "24/7 operation not independently verified."
        )
    elif hospital.emergency == "no":
        parts.append("Emergency services: No emergency department.")
    else:
        parts.append("Emergency services: Not specified.")

    # Bed availability honesty
    parts.append("Bed availability: Real-time bed availability is not determined from public map records.")
    return " ".join(parts)


class HospitalIndex:
    """In-memory FAISS Index with cosine similarity on normalized embeddings."""

    def __init__(self, hospitals: Optional[List[Hospital]] = None):
        self.hospitals: List[Hospital] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.dimension = 384
        if hospitals:
            self.build(hospitals)

    def build(self, hospitals: List[Hospital]) -> int:
        """Encode hospital profiles and build FAISS IndexFlatIP (cosine similarity)."""
        if not hospitals:
            raise ValueError("Cannot build index from empty hospital list.")

        self.hospitals = list(hospitals)
        texts = [build_hospital_profile_text(h) for h in self.hospitals]

        model = get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Normalize vectors for cosine similarity via inner product (IndexFlatIP)
        faiss.normalize_L2(embeddings)
        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        return self.index.ntotal

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Hospital, float]]:
        """Retrieve top_k hospitals by semantic cosine similarity."""
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("HospitalIndex is empty. Call build() first.")

        model = get_embedding_model()
        query_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_vec)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results: List[Tuple[Hospital, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.hospitals):
                # Clamp score to [-1.0, 1.0] for floating point rounding
                norm_score = float(max(-1.0, min(1.0, score)))
                results.append((self.hospitals[idx], norm_score))

        return results
