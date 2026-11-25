"""Hospital retrieval, need-formulation, and proximity re-ranking.

Extracts clinical requirements from IncidentReport objects, retrieves candidate
hospitals by source-backed eligibility, category, emergency tags, and straight-line Haversine
distance with explicit clinical honesty notes.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from core.rag.facility_classification import (
    CATEGORY_PRIORITY,
    classify_facility,
)
from core.rag.index import HospitalIndex
from core.schemas import Hospital, IncidentReport


OSM_ATTRIBUTION = "© OpenStreetMap contributors"
DISTANCE_LABEL = "straight-line, not driving distance (Haversine from SIMULATED demo location)"
ROAD_ROUTE_UNAVAILABLE = "unavailable — no configured road-routing provider"
LIMITATIONS_NOTE = (
    "Road distance and travel time are unavailable: no reliable routing provider is configured "
    "for this tab. Distances are straight-line (Haversine) measurements from the SIMULATED demo "
    "location and are NOT driving distances. Hospital bed availability, real-time wait times, "
    "ambulance dispatch status, independently verified 24/7 emergency departments, and "
    "unlisted specialties are NOT established by the OSM cache and are reported as not determined. "
    "Ranking scores are transparent heuristics, not verified medical suitability."
)


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth in kilometers.
    
    Uses standard Haversine formula with mean Earth radius R = 6371.0 km.
    """
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 3)


UNSUITABLE_SINGLE_SPECIALTIES = {
    "eye": {
        "keywords": ["eye", "drishti", "ophthalm", "netra"],
        "incident_triggers": ["eye", "vision", "blind", "cornea"],
    },
    "dental": {
        "keywords": ["dental", "dentist", "tooth", "teeth"],
        "incident_triggers": ["dental", "tooth", "teeth", "jaw"],
    },
    "maternity": {
        "keywords": ["maternity", "gynaec", "gynec", "obstetric", "fertility"],
        "incident_triggers": ["pregnant", "pregnancy", "maternity", "labor", "delivery", "fetus", "unborn"],
    },
    "children": {
        "keywords": ["children", "paediatric", "pediatric", "kids", "neonatal", "infant"],
        "incident_triggers": ["child", "infant", "baby", "pediatric", "kid", "toddler"],
    },
    "heart/cardiac": {
        "keywords": ["heart", "cardiac", "cardio"],
        "incident_triggers": ["heart", "cardiac", "chest pain", "angina", "infarction"],
    },
    "skin": {
        "keywords": ["skin", "derma", "dermatology"],
        "incident_triggers": ["skin rash", "dermatitis"],
    },
    "ENT": {
        "keywords": ["ent", "ear nose throat"],
        "incident_triggers": ["ent", "ear", "throat"],
    },
    "ayurvedic": {
        "keywords": ["ayurved", "ayush"],
        "incident_triggers": ["ayurved"],
    },
    "homeopathic": {
        "keywords": ["homeopath"],
        "incident_triggers": ["homeopath"],
    },
    "physiotherapy": {
        "keywords": ["physio", "physiotherapy", "rehabilitation"],
        "incident_triggers": ["physio", "rehabilitation"],
    },
}


def evaluate_facility_suitability(
    hospital: Hospital,
    incident_text: str,
) -> Tuple[float, Optional[str]]:
    """Determine if a facility has a single specialty unrelated to trauma.
    
    If unsuited to this incident, returns transparent down-rank penalty and rule description.
    """
    text_lower = incident_text.lower()
    # Only OSM specialty tags establish a specialty. A name such as "Children Hospital"
    # is not treated as proof of a paediatric emergency service.
    source_specialties = (hospital.specialties or "").lower()

    for spec_name, spec_info in UNSUITABLE_SINGLE_SPECIALTIES.items():
        is_spec = any(k in source_specialties for k in spec_info["keywords"])
        if is_spec:
            is_needed = any(t in text_lower for t in spec_info["incident_triggers"])
            if not is_needed:
                rule_desc = (
                    f"Down-ranked by suitability rule: source specialty '{spec_name}' is listed in "
                    "OpenStreetMap but the incident does not report a matching need."
                )
                return 0.30, rule_desc

    return 0.0, None


def formulate_retrieval_query(
    incident: Optional[IncidentReport] = None,
    urgency: Optional[str] = None,
    hazards: Optional[List[str]] = None,
    injuries: Optional[List[str]] = None,
    verdict: Optional[str] = None,
) -> str:
    """Formulate a clinical search query from an IncidentReport or explicit triage inputs.
    
    Adheres strictly to Task 0d:
    - Labels output as 'ASSUMED NEEDS (rule-based mapping from caller keywords)'
    - Strictly conservative: only maps keywords caller actually said
    - No unsupported expansions (e.g. smoke does NOT expand to chemical decontamination)
    - Always includes the generic need 'emergency / trauma department'
    """
    if incident is not None:
        urgency = urgency or incident.urgency
        hazards = hazards or incident.hazards
        injuries = injuries or incident.injuries
        verdict = verdict or incident.verdict

    urgency = urgency or "urgent"
    hazards = hazards or []
    injuries = injuries or []

    caller_tokens = (" ".join(injuries) + " " + " ".join(hazards)).lower()
    assumed_needs = ["emergency / trauma department"]

    if any(k in caller_tokens for k in ["head", "brain", "skull", "concussion", "unconscious", "coma"]):
        assumed_needs.append("head trauma management / neuro ICU")
    if any(k in caller_tokens for k in ["burn", "fire", "scald"]):
        assumed_needs.append("burn treatment unit")
    if any(k in caller_tokens for k in ["fracture", "bone", "broken", "amputation"]):
        assumed_needs.append("orthopedic trauma care")
    if any(k in caller_tokens for k in ["child", "pediatric", "infant", "baby"]):
        assumed_needs.append("pediatric emergency care")
    if any(k in caller_tokens for k in ["chest", "cardiac", "heart"]):
        assumed_needs.append("cardiac resuscitation")
    if any(k in caller_tokens for k in ["eye", "blind", "vision"]):
        assumed_needs.append("ophthalmology care")
    if any(k in caller_tokens for k in ["pregnant", "pregnancy", "maternity", "labor"]):
        assumed_needs.append("obstetric emergency care")
    if any(k in caller_tokens for k in ["chemical", "toxic", "acid", "poison"]):
        assumed_needs.append("chemical decontamination")

    needs_str = "; ".join(assumed_needs)
    return f"ASSUMED NEEDS (rule-based mapping from caller keywords): {needs_str}. Urgency: {urgency}."


def emergency_suitability(hospital: Hospital) -> Dict[str, Any]:
    """Report emergency suitability from the OSM emergency tag only.

    Nothing here is independently verified, and 24/7 operation is never inferred.
    """
    tag = (hospital.emergency or "").strip().lower()
    if tag == "yes":
        return {
            "status": "SOURCE_CONFIRMED",
            "tier": 0,
            "recommended_for_emergency": True,
            "label": "OSM emergency=yes (source-reported; not independently verified; 24/7 not established)",
        }
    if tag == "no":
        return {
            "status": "REPORTED_NO",
            "tier": 2,
            "recommended_for_emergency": False,
            "label": "OSM emergency=no — no emergency department reported in the source",
        }
    return {
        "status": "NOT_SPECIFIED",
        "tier": 1,
        "recommended_for_emergency": False,
        "label": "not specified in OpenStreetMap — emergency department unavailable/unverified",
    }


def listed_specialties(hospital: Hospital) -> Dict[str, Any]:
    """Return only specialties actually present in the source record."""
    raw = (hospital.specialties or "").strip()
    if not raw:
        return {
            "items": [],
            "categories": [],
            "display": "not specified in OpenStreetMap",
        }
    items = [part.strip() for part in re.split(r"[;,]", raw) if part.strip()]
    categories = []
    checks = (
        ("trauma", ("trauma",)),
        ("neurology", ("neuro",)),
        ("emergency", ("emergency",)),
        ("cardiology", ("cardiac", "cardio", "heart")),
        ("paediatrics", ("paediatric", "pediatric", "neonat")),
        ("gynaecology", ("gynaec", "gynec", "obstet")),
        ("orthopaedics", ("ortho",)),
        ("surgery", ("surg",)),
    )
    for label, keywords in checks:
        if any(any(keyword in item.lower() for keyword in keywords) for item in items):
            categories.append(label)
    return {
        "items": items,
        "categories": categories,
        "display": raw,
    }


def candidate_rank_key(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
    """Rank source-confirmed emergency hospitals before proximity."""
    return (
        candidate["emergency_tier"],
        candidate["category_tier"],
        -candidate["combined_score"],
        candidate["distance_km"],
        candidate["hospital"].name.lower(),
    )


def score_hospital_candidate(
    semantic_score: float,
    distance_km: float,
    emergency_tag: Optional[str] = None,
    suitability_penalty: float = 0.0,
) -> Tuple[float, float, float]:
    """Calculate a deterministic proximity/emergency-tag heuristic.

    ``semantic_score`` remains in the signature for backwards compatibility and
    is intentionally ignored. The returned third value is always 0.0.
    """
    proximity_score = 1.0 / (1.0 + (max(0.0, distance_km) / 8.0))
    emergency_bonus = 0.10 if emergency_tag and emergency_tag.lower() == "yes" else 0.0
    combined_score = max(0.02, 0.80 * proximity_score + emergency_bonus - suitability_penalty)
    return round(combined_score, 4), round(proximity_score, 4), 0.0

def explain_match(
    hospital: Hospital,
    distance_km: float,
    semantic_score: float,
    query_text: str,
    suitability_rule: Optional[str] = None,
    category: Optional[str] = None,
    emergency_info: Optional[Dict[str, Any]] = None,
    specialties_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Explain a recommendation without presenting a heuristic as clinical verification."""
    reasons = []
    reasons.append(f"Facility category '{category or classify_facility(hospital.name).category}' from OSM name/tags")
    reasons.append(
        (emergency_info or emergency_suitability(hospital))["label"]
    )
    specialties = specialties_info or listed_specialties(hospital)
    if specialties["items"]:
        reasons.append(f"specialties listed in OSM: {specialties['display']}")
    else:
        reasons.append("no specialties specified in OpenStreetMap; none assumed")
    reasons.append(f"{distance_km:.1f} km straight-line from the SIMULATED demo location")
    reasons.append("ranking uses source tags, facility category and proximity; no semantic similarity is computed")
    if suitability_rule:
        reasons.append(suitability_rule)
    reasons.append("transparent ranking heuristic, not verified medical suitability")
    return "; ".join(reasons)


class HospitalRetriever:
    """High-level retriever using deterministic source/tag/category/proximity ranking."""

    def __init__(self, index: HospitalIndex):
        self.index = index

    def retrieve_suggestions(
        self,
        incident_lat: float,
        incident_lon: float,
        incident: Optional[IncidentReport] = None,
        query_override: Optional[str] = None,
        top_candidates: int = 30,
        final_top_k: int = 3,
    ) -> Dict[str, Any]:
        """Retrieve and re-rank eligible hospital suggestions for an incident.

        Irrelevant/unclassified facilities are removed before scoring. Ordering
        prioritizes source-confirmed emergency hospitals and facility category
        over straight-line proximity; scores remain labelled as heuristics.
        """
        query = query_override or formulate_retrieval_query(incident=incident)
        indexed_records = getattr(self.index, "hospitals", None)
        candidate_limit = len(indexed_records) if indexed_records is not None else top_candidates
        raw_candidates = self.index.search(query, top_k=candidate_limit)

        scored_candidates = []
        excluded_count = 0
        incident_context_text = query
        if incident:
            incident_context_text += f" {' '.join(incident.injuries)} {' '.join(incident.hazards)}"

        for hospital, _legacy_score in raw_candidates:
            classification = classify_facility(hospital.name)
            if hospital.excluded_from_dispatch_reason is not None or not classification.dispatch_eligible:
                # Never rank architecture departments, universities, shops, offices,
                # clinics or other unrelated facilities as emergency destinations.
                excluded_count += 1
                continue

            dist = haversine_distance_km(incident_lat, incident_lon, hospital.latitude, hospital.longitude)
            suitability_penalty, suit_rule = evaluate_facility_suitability(
                hospital=hospital,
                incident_text=incident_context_text,
            )
            combined, prox_score, norm_sem = score_hospital_candidate(
                semantic_score=0.0,
                distance_km=dist,
                emergency_tag=hospital.emergency,
                suitability_penalty=suitability_penalty,
            )
            emergency_info = emergency_suitability(hospital)
            specialties_info = listed_specialties(hospital)
            why = explain_match(
                hospital=hospital,
                distance_km=dist,
                semantic_score=norm_sem,
                query_text=query,
                suitability_rule=suit_rule,
                category=classification.category,
                emergency_info=emergency_info,
                specialties_info=specialties_info,
            )

            scored_candidates.append({
                "hospital": hospital,
                "distance_km": dist,
                "distance_label": DISTANCE_LABEL,
                "facility_category": classification.category,
                "category_tier": CATEGORY_PRIORITY.get(classification.category, 4),
                "emergency_status": emergency_info["status"],
                "emergency_suitability": emergency_info["label"],
                "emergency_tier": emergency_info["tier"],
                "recommended_for_emergency": emergency_info["recommended_for_emergency"],
                "specialties_listed": specialties_info["items"],
                "specialty_categories": specialties_info["categories"],
                "phone": hospital.contact_number if hospital.contact_number else "not listed",
                "phone_status": "source phone number from OSM" if hospital.contact_number else "missing in OSM; unverified",
                "emergency": hospital.emergency if hospital.emergency else "not specified",
                "specialties": hospital.specialties if hospital.specialties else "not specified in OpenStreetMap",
                "address": hospital.address if hospital.address else "not determined in OpenStreetMap",
                "why_suggested": why,
                "suitability_rule": suit_rule,
                "bed_availability": "not determined",
                "road_distance_km": ROAD_ROUTE_UNAVAILABLE,
                "driving_time_min": ROAD_ROUTE_UNAVAILABLE,
                "score_label": "deterministic proximity/category/emergency-tag heuristic, not verified medical suitability",
                "attribution": OSM_ATTRIBUTION,
                "semantic_score": None,
                "proximity_score": prox_score,
                "combined_score": combined,
            })

        scored_candidates.sort(key=candidate_rank_key)
        top_suggestions = scored_candidates[:final_top_k]

        return {
            "query_used": query,
            "incident_coordinates": {"latitude": incident_lat, "longitude": incident_lon},
            "incident_coordinates_status": "SIMULATED demo coordinates; no GPS verification",
            "suggestions": top_suggestions,
            "eligible_facility_count": len(scored_candidates),
            "excluded_facility_count": excluded_count,
            "source_confirmed_emergency_count": sum(
                1 for item in scored_candidates if item["emergency_tier"] == 0
            ),
            "road_distance_km": ROAD_ROUTE_UNAVAILABLE,
            "driving_time_min": ROAD_ROUTE_UNAVAILABLE,
            "limitations_note": LIMITATIONS_NOTE,
            "attribution": OSM_ATTRIBUTION,
        }
