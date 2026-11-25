"""Domain-specific hospital intelligence for Project Third-Eye.

This package ingests OpenStreetMap facility data, classifies it from source tags,
and applies transparent deterministic ranking rules. It does not use retrieval-
augmented generation, embedding models, vector indexes, or semantic search.
"""

from core.hospitals.catalog import HospitalCatalog
from core.hospitals.classification import (
    FacilityClassification,
    classify_facility,
    classify_hospital_record,
    dispatch_eligible_facilities,
)
from core.hospitals.ranking import (
    HospitalRanker,
    describe_assumed_needs,
    haversine_distance_km,
)

__all__ = [
    "FacilityClassification",
    "HospitalCatalog",
    "HospitalRanker",
    "classify_facility",
    "classify_hospital_record",
    "describe_assumed_needs",
    "dispatch_eligible_facilities",
    "haversine_distance_km",
]
