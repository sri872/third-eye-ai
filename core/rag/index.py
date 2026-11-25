"""Deterministic in-memory holder for OpenStreetMap hospital records.

The legacy module path is retained to avoid breaking older imports. No semantic
embeddings, vector index, or retrieval-augmented generation is implemented.
"""
from typing import List, Optional, Tuple
from core.schemas import Hospital

def get_embedding_model():
    """Compatibility guard for the removed embedding-based implementation."""
    raise RuntimeError("Embedding retrieval has been removed; no embedding model is required.")

def build_hospital_profile_text(hospital: Hospital) -> str:
    """Build source-grounded descriptive text for display/debugging only."""
    parts = [f"Facility: {hospital.name}."]
    if hospital.address:
        parts.append(f"Location: {hospital.address}.")
    if hospital.specialties:
        parts.append(f"Specialties reported in OSM: {hospital.specialties}.")
    else:
        parts.append("Specialties: not specified in OSM.")
    if hospital.emergency == "yes":
        parts.append("Emergency: yes (source-reported; not independently verified).")
    elif hospital.emergency == "no":
        parts.append("Emergency: no (source-reported).")
    else:
        parts.append("Emergency: not specified in OSM.")
    parts.append("Bed availability: not determined.")
    return " ".join(parts)

class HospitalIndex:
    """Lightweight collection of records; ranking occurs in HospitalRetriever."""
    def __init__(self, hospitals: Optional[List[Hospital]] = None):
        self.hospitals: List[Hospital] = list(hospitals or [])

    def build(self, hospitals: List[Hospital]) -> int:
        if not hospitals:
            raise ValueError("Cannot build index from empty hospital list.")
        self.hospitals = list(hospitals)
        return len(self.hospitals)

    def search(self, query: str = "", top_k: int = 10) -> List[Tuple[Hospital, float]]:
        """Return candidates with a neutral legacy score; no text similarity is computed."""
        if not self.hospitals:
            raise RuntimeError("HospitalIndex is empty. Call build() first.")
        return [(hospital, 0.0) for hospital in self.hospitals[:max(0, int(top_k))]]
