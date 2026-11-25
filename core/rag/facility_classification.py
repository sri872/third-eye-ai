"""Conservative classification of OpenStreetMap facilities for emergency dispatch.

The classifier uses only two kinds of evidence: the OSM tags attached to a record
and unambiguous words in the facility's own name. It never uses distance to an
incident, partial word matches, or guessed medical capability. A mis-tagged
building (for example an architecture studio tagged ``healthcare=hospital``) is
rejected when its name identifies it as a non-clinical facility.

Categories are deliberately kept in this module rather than in the Hospital
schema: they are derived facts, not new source records, and therefore do not
change the existing database rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Optional

CATEGORY_HOSPITAL = "hospital"
CATEGORY_TRAUMA_CENTRE = "trauma centre"
CATEGORY_CLINIC = "clinic"
CATEGORY_OTHER_HEALTHCARE = "other healthcare"
CATEGORY_UNRELATED = "unrelated / unclassified"

# Smaller numbers mean higher priority for emergency dispatch ranking.
CATEGORY_PRIORITY = {
    CATEGORY_TRAUMA_CENTRE: 0,
    CATEGORY_HOSPITAL: 1,
    CATEGORY_CLINIC: 2,
    CATEGORY_OTHER_HEALTHCARE: 3,
    CATEGORY_UNRELATED: 4,
}

_NON_MEDICAL_MARKERS = (
    "architecture",
    "architectural",
    "architect",
    "design",
    "deisgn",  # present as a misspelling in the real OSM cache
    "university",
    "school",
    "hostel",
    "campus",
    "office",
    "shop",
    "store",
    "showroom",
    "parking",
    "library",
    "faculty",
)

_NON_MEDICAL_BUILDINGS = {
    "office",
    "retail",
    "commercial",
    "university",
    "school",
    "warehouse",
}

_OUTPATIENT_MARKERS = (
    "clinic",
    "dentist",
    "dental",
    "pharmacy",
    "chemist",
    "laboratory",
    "pathology",
    "veterinary",
)

_OUTPATIENT_TAGS = {
    "clinic",
    "dentist",
    "pharmacy",
    "laboratory",
    "veterinary",
}

_OTHER_HEALTHCARE_MARKERS = ("lab", "labs", "vet", "ambulance station", "blood bank")


@dataclass(frozen=True)
class FacilityClassification:
    """Derived, source-based classification of one facility."""

    category: str
    is_healthcare: bool
    dispatch_eligible: bool
    reason: Optional[str]


def _word_match(text: str, marker: str) -> bool:
    """Match a whole word/phrase; avoids partial matches such as 'design' in 'designation'."""
    return re.search(r"\b" + re.escape(marker) + r"\b", text) is not None


def _normalized_name(name: str) -> str:
    """Normalize known spacing variants without inventing any missing words."""
    text = str(name or "").strip().lower()
    return re.sub(r"\bcivilhospital\b", "civil hospital", text)


def _has_hospital_signal(name: str) -> bool:
    return (
        _word_match(name, "hospital")
        or _word_match(name, "nursing home")
        or _word_match(name, "medical college")
    )


def _has_trauma_signal(name: str) -> bool:
    tokens = ("centre", "center", "hospital", "institute", "base", "unit")
    return _word_match(name, "trauma") and any(_word_match(name, token) for token in tokens)


def classify_facility(
    name: str,
    raw_tags: Optional[Dict[str, Any]] = None,
) -> FacilityClassification:
    """Classify a facility using its name and OSM tags only.

    Returns a conservative classification. ``dispatch_eligible`` is True only
    for an unambiguous hospital or trauma centre; clinics and other healthcare
    services are kept as healthcare records but are not emergency destinations.
    """
    text = _normalized_name(name)
    tags = raw_tags or {}
    amenity = str(tags.get("amenity") or "").strip().lower()
    healthcare = str(tags.get("healthcare") or "").strip().lower()

    # An explicit non-clinical building tag contradicts a stray healthcare tag.
    if str(tags.get("office") or "").strip() or str(tags.get("shop") or "").strip():
        return FacilityClassification(
            category=CATEGORY_UNRELATED,
            is_healthcare=False,
            dispatch_eligible=False,
            reason="Non-medical OSM office/shop tag; not a healthcare facility",
        )
    building = str(tags.get("building") or "").strip().lower()
    if building in _NON_MEDICAL_BUILDINGS and not _has_hospital_signal(text):
        return FacilityClassification(
            category=CATEGORY_UNRELATED,
            is_healthcare=False,
            dispatch_eligible=False,
            reason=f"Non-medical OSM building tag ('{building}'); not a healthcare facility",
        )

    # Names win over unreliable tags: architecture/design/university/office/shop sites
    # must never become hospitals just because a mapper mis-tagged them.
    for marker in _NON_MEDICAL_MARKERS:
        if _word_match(text, marker):
            return FacilityClassification(
                category=CATEGORY_UNRELATED,
                is_healthcare=False,
                dispatch_eligible=False,
                reason=f"Non-medical facility name suggests non-clinical site (contains '{marker}')",
            )
    if _word_match(text, "college"):
        medical_college = bool(re.search(r"\b(medical|nursing)\b", text))
        if not medical_college or not _has_hospital_signal(text):
            return FacilityClassification(
                category=CATEGORY_UNRELATED,
                is_healthcare=False,
                dispatch_eligible=False,
                reason="Non-medical facility name suggests an educational site (contains 'college')",
            )

    # Outpatient and non-hospital healthcare services are not emergency destinations.
    for marker in _OUTPATIENT_MARKERS:
        if _word_match(text, marker):
            return FacilityClassification(
                category=CATEGORY_CLINIC,
                is_healthcare=True,
                dispatch_eligible=False,
                reason=(
                    f"Facility name indicates outpatient clinic, dental, or lab rather than "
                    f"hospital (contains '{marker}')"
                ),
            )
    for marker in _OTHER_HEALTHCARE_MARKERS:
        if _word_match(text, marker) and not _has_hospital_signal(text):
            return FacilityClassification(
                category=CATEGORY_OTHER_HEALTHCARE,
                is_healthcare=True,
                dispatch_eligible=False,
                reason=f"Facility is a non-hospital healthcare service (contains '{marker}')",
            )

    if amenity in _OUTPATIENT_TAGS or healthcare in _OUTPATIENT_TAGS:
        tag = amenity if amenity in _OUTPATIENT_TAGS else healthcare
        return FacilityClassification(
            category=CATEGORY_CLINIC,
            is_healthcare=True,
            dispatch_eligible=False,
            reason=f"OSM tag '{tag}' marks an outpatient/non-hospital healthcare service",
        )

    tagged_hospital = amenity == "hospital" or healthcare == "hospital"
    tagged_trauma = healthcare == "trauma" or str(tags.get("emergency") or "").lower() == "trauma"

    if tagged_trauma or _has_trauma_signal(text):
        return FacilityClassification(
            category=CATEGORY_TRAUMA_CENTRE,
            is_healthcare=True,
            dispatch_eligible=True,
            reason=None,
        )
    if tagged_hospital or _has_hospital_signal(text):
        return FacilityClassification(
            category=CATEGORY_HOSPITAL,
            is_healthcare=True,
            dispatch_eligible=True,
            reason=None,
        )
    if healthcare in {"doctor", "centre", "center", "yes"} or amenity in {"doctors", "health_post"}:
        return FacilityClassification(
            category=CATEGORY_OTHER_HEALTHCARE,
            is_healthcare=True,
            dispatch_eligible=False,
            reason=f"OSM tag does not establish a hospital ({amenity or healthcare})",
        )

    return FacilityClassification(
        category=CATEGORY_UNRELATED,
        is_healthcare=False,
        dispatch_eligible=False,
        reason="Missing hospital tag (amenity/healthcare=hospital) and clear hospital name keyword",
    )


def dispatch_eligible_facilities(hospitals, classification_by_id=None):
    """Return only facilities whose name/tags establish a hospital or trauma centre."""
    known = classification_by_id or {}
    eligible = []
    for hospital in hospitals:
        classification = known.get(hospital.hospital_id) or classify_facility(hospital.name)
        if classification.dispatch_eligible and hospital.excluded_from_dispatch_reason is None:
            eligible.append(hospital)
    return eligible
