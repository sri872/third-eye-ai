"""Deterministic hospital ranking, need descriptions and distance helpers.

Candidate facilities are listed from the catalog and ordered by three explicit
rules: the source-reported emergency tag tier, the source-derived facility
category tier, and finally a transparent proximity/suitability heuristic. Any
heuristic number is for ordering only and is never a clinical suitability rating.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from core.hospitals.catalog import HospitalCatalog
from core.hospitals.classification import (
    CATEGORY_PRIORITY,
    classify_facility,
    classify_hospital_record,
)
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
    """Calculate the great-circle distance between two points in kilometers.

    Uses the standard Haversine formula with mean Earth radius R = 6371.0 km.
    The result is a straight-line distance, never a driving distance.
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
    """Apply a transparent down-rank rule for an unrelated single specialty.

    Only source-tagged specialties can trigger this rule. A facility name such as
    "Children Hospital" is not treated as proof of a paediatric emergency service.
    """
    text_lower = incident_text.lower()
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


def describe_assumed_needs(
    incident: Optional[IncidentReport] = None,
    urgency: Optional[str] = None,
    hazards: Optional[List[str]] = None,
    injuries: Optional[List[str]] = None,
    verdict: Optional[str] = None,
) -> str:
    """Describe rule-based assumed needs using only words the caller reported.

    This is an explanatory display string, not a clinical query or a model input.
    It never expands a keyword into an unstated capability or service.
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
    """Report emergency status from the OSM emergency tag only.

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
    """Rank by emergency-tag tier, facility-category tier, heuristic, distance, name."""
    return (
        candidate["emergency_tier"],
        candidate["category_tier"],
        -candidate["combined_score"],
        candidate["distance_km"],
        candidate["hospital"].name.lower(),
    )


def score_candidate_heuristic(
    distance_km: float,
    emergency_tag: Optional[str] = None,
    suitability_penalty: float = 0.0,
) -> Tuple[float, float]:
    """Calculate the secondary proximity/emergency-tag ordering heuristic.

    Returns ``(combined_score, proximity_score)``. This number determines ranking
    order only after the emergency-tag and facility-category tiers are applied;
    it is not a clinical suitability rating.
    """
    proximity_score = 1.0 / (1.0 + (max(0.0, distance_km) / 8.0))
    emergency_bonus = 0.10 if emergency_tag and emergency_tag.lower() == "yes" else 0.0
    combined_score = max(0.02, 0.80 * proximity_score + emergency_bonus - suitability_penalty)
    return round(combined_score, 4), round(proximity_score, 4)


def explain_ranking(
    hospital: Hospital,
    distance_km: float,
    suitability_rule: Optional[str] = None,
    category: Optional[str] = None,
    emergency_info: Optional[Dict[str, Any]] = None,
    specialties_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Explain an ordering decision without claiming clinical verification."""
    reasons = []
    reasons.append(f"Facility category '{category or classify_facility(hospital.name).category}' from OSM name/tags")
    reasons.append((emergency_info or emergency_suitability(hospital))["label"])
    specialties = specialties_info or listed_specialties(hospital)
    if specialties["items"]:
        reasons.append(f"specialties listed in OSM: {specialties['display']}")
    else:
        reasons.append("no specialties specified in OpenStreetMap; none assumed")
    reasons.append(f"{distance_km:.1f} km straight-line from the SIMULATED demo location")
    reasons.append("ranking uses source tags, facility category and proximity; no model-based text search is used")
    if suitability_rule:
        reasons.append(suitability_rule)
    reasons.append("transparent ranking heuristic, not verified medical suitability")
    return "; ".join(reasons)


class HospitalRanker:
    """Rank listed hospital candidates with transparent deterministic rules."""

    def __init__(self, catalog: HospitalCatalog):
        self.catalog = catalog

    def rank_suggestions(
        self,
        incident_lat: float,
        incident_lon: float,
        incident: Optional[IncidentReport] = None,
        assumed_needs: Optional[str] = None,
        top_candidates: int = 30,
        final_top_k: int = 3,
    ) -> Dict[str, Any]:
        """List eligible candidates and rank them by the documented policy.

        Ordering: source-reported emergency tag tier, then source-derived
        facility category tier, then the secondary proximity/suitability
        heuristic. Irrelevant or unclassified facilities are excluded first.
        """
        needs_text = assumed_needs or describe_assumed_needs(incident=incident)
        candidate_limit = len(self.catalog) if hasattr(self.catalog, "__len__") else top_candidates
        candidates = self.catalog.list_candidates(limit=candidate_limit)

        scored_candidates = []
        excluded_count = 0
        incident_context_text = needs_text
        if incident:
            incident_context_text += f" {' '.join(incident.injuries)} {' '.join(incident.hazards)}"

        for hospital in candidates:
            classification = classify_hospital_record(hospital)
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
            combined, prox_score = score_candidate_heuristic(
                distance_km=dist,
                emergency_tag=hospital.emergency,
                suitability_penalty=suitability_penalty,
            )
            emergency_info = emergency_suitability(hospital)
            specialties_info = listed_specialties(hospital)
            why = explain_ranking(
                hospital=hospital,
                distance_km=dist,
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
                "ranking_priority": {
                    "emergency_tag_tier": emergency_info["tier"],
                    "facility_category_tier": CATEGORY_PRIORITY.get(classification.category, 4),
                    "secondary_proximity_heuristic": combined,
                },
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
                "proximity_score": prox_score,
                "combined_score": combined,
            })

        scored_candidates.sort(key=candidate_rank_key)
        top_suggestions = scored_candidates[:final_top_k]

        return {
            "assumed_needs": needs_text,
            "ranking_method": (
                "OSM emergency-tag tier first, facility category tier second, "
                "secondary proximity heuristic third; not a clinical suitability rating"
            ),
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
