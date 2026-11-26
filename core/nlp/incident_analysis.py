"""Deterministic Incident Analysis and Urgency Classification.

Applies transparent rule-based logic to classify incident types, compute urgency,
and detect contradictions in caller reports without black-box ML models.
"""

from typing import Dict, Any, List
import re
from core.nlp.entity_extraction import extract_entities, _is_locally_negated


def analyze_incident(text: str) -> Dict[str, Any]:
    """Perform deterministic incident analysis on caller text.

    Args:
        text: Incident report description.

    Returns:
        Analysis dictionary containing incident type, urgency level,
        rationale, and contradiction flags.
    """
    entities = extract_entities(text)
    norm = entities["normalized_text"]

    # 1. Incident Classification
    incident_types: List[str] = []

    # Overturned / Rollover
    if any(h["category"] == "overturned_vehicle" for h in entities["hazards"]) or "rolled over" in norm:
        incident_types.append("ROLLOVER_COLLISION")

    # Fire / Hazardous condition
    if any(h["category"] in ("fire", "fuel_leak", "gas_leak") for h in entities["hazards"]):
        incident_types.append("HAZARDOUS_FIRE_OR_SPILL")

    # Multi-vehicle collision
    if entities["total_vehicles_reported"] >= 2 or "two car" in norm or "multi" in norm or "pileup" in norm:
        incident_types.append("MULTI_VEHICLE_COLLISION")
    elif entities["total_vehicles_reported"] == 1:
        incident_types.append("SINGLE_VEHICLE_INCIDENT")

    # Pedestrian involvement
    if "pedestrian" in norm or "walker" in norm or "crossing road" in norm:
        incident_types.append("PEDESTRIAN_INVOLVED")

    # Obstruction / Traffic
    if any(h["category"] == "road_obstruction" for h in entities["hazards"]):
        incident_types.append("TRAFFIC_BLOCKAGE")

    if not incident_types:
        incident_types.append("GENERAL_ROAD_INCIDENT")

    # Generic collision language is useful when a caller says "vehicles
    # collided" without naming a count or body style. Respect local negation.
    collision_terms = ("accident", "crash", "crashed", "collision", "collided", "struck", "hit")
    explicit_collision = any(
        not _is_locally_negated(norm, match.start())
        for term in collision_terms
        for match in re.finditer(rf"\b{term}\b", norm)
    )
    if explicit_collision and not any(
        kind in incident_types
        for kind in ("MULTI_VEHICLE_COLLISION", "ROLLOVER_COLLISION")
    ):
        incident_types.append("VEHICLE_COLLISION")

    # 2. Urgency Determination using explicit transparent rules
    # CRITICAL: Fatalities, unconscious victims, active fire, trapped occupants, severe head trauma
    critical_injuries = {
        "unconscious", "fatality", "critical_condition", "trapped_occupant",
        "head_injury", "breathing_difficulty", "seizure", "chest_pain"
    }
    critical_hazards = {"fire", "gas_leak", "electrical_hazard"}

    has_critical_inj = any(i["category"] in critical_injuries for i in entities["injuries"])
    has_critical_haz = any(h["category"] in critical_hazards for h in entities["hazards"])

    # HIGH: Fractures, heavy bleeding, overturned vehicle, fuel leak
    high_injuries = {"fracture", "bleeding", "serious_injury", "burn"}
    high_hazards = {"fuel_leak", "overturned_vehicle", "smoke", "oil_spill", "road_obstruction"}

    has_high_inj = any(i["category"] in high_injuries for i in entities["injuries"])
    has_high_haz = any(h["category"] in high_hazards for h in entities["hazards"])

    if has_critical_inj or has_critical_haz:
        urgency = "CRITICAL"
        urgency_reason = "Critical life-safety indicators detected (unconscious/trapped victim, fire, or severe injury)."
    elif has_high_inj or has_high_haz:
        urgency = "HIGH"
        urgency_reason = "Serious injury or major hazard detected (fracture, bleeding, fuel spill, or overturned vehicle)."
    elif entities["injuries"] or entities["hazards"]:
        urgency = "MEDIUM"
        urgency_reason = "Moderate injury or road hazard reported (minor injuries or traffic disruption)."
    else:
        urgency = "LOW"
        urgency_reason = "No injuries, severe hazards, or critical conditions reported by caller."

    # 3. Contradiction Detection
    contradictions: List[str] = []

    # Only flag a contradiction when a general denial conflicts with a
    # positive item, or the same specific category is both denied and reported.
    positive_injury_categories = {
        item["category"] for item in entities.get("injuries", [])
    }
    negated_injury_categories = set(entities.get("negated_injury_categories", []))
    if (
        entities.get("general_injury_denial") and positive_injury_categories
    ) or positive_injury_categories.intersection(negated_injury_categories):
        contradictions.append(
            "Contradiction: An injury denial conflicts with a separately reported injury cue."
        )

    positive_hazard_categories = {
        item["category"] for item in entities.get("hazards", [])
    }
    negated_hazard_categories = set(entities.get("negated_hazard_categories", []))
    if (
        entities.get("general_hazard_denial") and positive_hazard_categories
    ) or positive_hazard_categories.intersection(negated_hazard_categories):
        contradictions.append(
            "Contradiction: A hazard denial conflicts with a separately reported hazard cue."
        )

    # Check: Single car reported vs two-vehicle collision mention
    if entities["total_vehicles_reported"] == 1 and ("head-on" in norm or "collided with each other" in norm or "both cars" in norm):
        contradictions.append(
            "Contradiction: Single vehicle claimed but collision syntax implies multiple vehicles."
        )

    return {
        "entities": entities,
        "primary_incident_type": incident_types[0],
        "all_incident_types": incident_types,
        "urgency_level": urgency,
        "urgency_reason": urgency_reason,
        "contradictions_detected": contradictions,
        "has_contradictions": len(contradictions) > 0,
    }
