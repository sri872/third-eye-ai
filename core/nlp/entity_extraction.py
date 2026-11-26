"""Deterministic Entity and Keyword Extraction.

Extracts vehicles, locations, injuries, hazards, and numerical quantities
from incident descriptions using regular expressions and keyword dictionaries.
"""

import re
from typing import Dict, Any, List, Optional
from core.nlp.text_normalization import normalize_text, tokenize_words


# Dictionaries for entity extraction
VEHICLE_KEYWORDS = {
    "car": "car",
    "sedan": "car",
    "hatchback": "car",
    "suv": "suv",
    "jeep": "suv",
    "truck": "truck",
    "lorry": "truck",
    "semi": "truck",
    "bus": "bus",
    "minibus": "bus",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bike": "motorcycle",
    "two-wheeler": "motorcycle",
    "scooter": "scooter",
    "scooty": "scooter",
    "auto": "auto_rickshaw",
    "rickshaw": "auto_rickshaw",
    "auto-rickshaw": "auto_rickshaw",
    "van": "van",
    "ambulance": "ambulance",
    "bicycle": "bicycle",
}

COLOR_KEYWORDS = [
    "white", "black", "red", "silver", "grey", "gray", "blue", "yellow",
    "green", "golden", "maroon", "orange", "dark"
]

INJURY_KEYWORDS = {
    "unconscious": "unconscious",
    "fainted": "unconscious",
    "passed out": "unconscious",
    "fatal": "fatality",
    "dead": "fatality",
    "killed": "fatality",
    "critical": "critical_condition",
    "serious": "serious_injury",
    "severe": "serious_injury",
    "bleeding": "bleeding",
    "blood": "bleeding",
    "fracture": "fracture",
    "broken bone": "fracture",
    "head injury": "head_injury",
    "trapped": "trapped_occupant",
    "stuck inside": "trapped_occupant",
    "injured": "general_injury",
    "hurt": "general_injury",
    "minor injury": "minor_injury",
    "scratches": "minor_injury",
    "cuts": "minor_injury",
}

HAZARD_KEYWORDS = {
    "fire": "fire",
    "flames": "fire",
    "burning": "fire",
    "smoke": "smoke",
    "smoking": "smoke",
    "fuel leak": "fuel_leak",
    "petrol leak": "fuel_leak",
    "diesel leak": "fuel_leak",
    "gas leak": "gas_leak",
    "oil spill": "oil_spill",
    "chemical": "hazardous_material",
    "overturned": "overturned_vehicle",
    "flipped": "overturned_vehicle",
    "upside down": "overturned_vehicle",
    "blocked road": "road_obstruction",
    "traffic jam": "road_obstruction",
    "highway blocked": "road_obstruction",
    "power line": "electrical_hazard",
    "live wire": "electrical_hazard",
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}


def extract_entities(text: str) -> Dict[str, Any]:
    """Extract structured entities from caller incident text using deterministic rules.

    Args:
        text: Raw incident narrative or transcript.

    Returns:
        Structured entity dictionary.
    """
    norm = normalize_text(text)
    tokens = tokenize_words(norm)

    # 1. Extract Vehicles with modifier colors and quantities
    vehicles_found: List[Dict[str, Any]] = []

    # Search for patterns like "two white cars", "a red truck", "speeding bus"
    # Match optional count, optional adjectives/colors, and the vehicle keyword
    for word, canonical_type in VEHICLE_KEYWORDS.items():
        pattern = rf"\b((?:(?:[0-9]+|one|two|three|four|five|several|multiple)\s+)?(?:[a-z]+\s+)?{word}s?)\b"
        matches = re.finditer(pattern, norm)
        for match in matches:
            phrase = match.group(0)
            # Count extraction
            count = 1
            for nw, val in NUMBER_WORDS.items():
                if nw in phrase:
                    count = val
                    break
            num_digit = re.search(r"\b([0-9]+)\b", phrase)
            if num_digit:
                count = int(num_digit.group(1))

            # Color extraction
            color = None
            for col in COLOR_KEYWORDS:
                if col in phrase:
                    color = col
                    break

            vehicles_found.append({
                "raw_phrase": phrase,
                "type": canonical_type,
                "color": color,
                "count": count
            })

    # Deduplicate overlapping vehicle matches (e.g. 'car' vs 'two cars')
    deduped_vehicles = []
    seen_phrases = set()
    for v in vehicles_found:
        if v["raw_phrase"] not in seen_phrases:
            seen_phrases.add(v["raw_phrase"])
            deduped_vehicles.append(v)
    vehicles_found = deduped_vehicles

    # Total reported vehicle count
    total_vehicles = sum(v["count"] for v in vehicles_found) if vehicles_found else 0

    # 2. Extract Locations
    locations_found = _extract_locations(norm)

    # 3. Extract Injuries (checking for negation, e.g. 'no injuries')
    injuries_found = []
    has_negated_injuries = bool(re.search(r"\b(no|zero|without|not)\s+(injur|fatal|hurt|casualt|victim|bleed)", norm))

    if not has_negated_injuries:
        for phrase, canonical_inj in INJURY_KEYWORDS.items():
            if re.search(rf"\b{phrase}\b", norm):
                injuries_found.append({
                    "keyword": phrase,
                    "category": canonical_inj
                })

    # 4. Extract Hazards (checking for negation)
    hazards_found = []
    has_negated_hazards = bool(re.search(r"\b(no|without)\s+(fire|smoke|leak|flame|hazard)", norm))

    if not has_negated_hazards:
        for phrase, canonical_haz in HAZARD_KEYWORDS.items():
            if re.search(rf"\b{phrase}\b", norm):
                hazards_found.append({
                    "keyword": phrase,
                    "category": canonical_haz
                })

    # 5. Extract Reported People / Casualties count
    people_count = None
    people_match = re.search(r"\b([0-9]+|one|two|three|four|five|six)\s+(people|persons|passengers|victims|casualties|occupants)\b", norm)
    if people_match:
        val_str = people_match.group(1)
        people_count = NUMBER_WORDS.get(val_str, int(val_str) if val_str.isdigit() else 1)

    return {
        "raw_text": text,
        "normalized_text": norm,
        "vehicles": vehicles_found,
        "total_vehicles_reported": total_vehicles,
        "locations": locations_found,
        "injuries": injuries_found,
        "hazards": hazards_found,
        "people_count_reported": people_count,
        "has_negated_injuries": has_negated_injuries,
        "has_negated_hazards": has_negated_hazards,
    }


def _extract_locations(text: str) -> List[str]:
    """Extract location references using deterministic keyword and prepositional patterns."""
    locations: List[str] = []

    # Known regional landmarks and corridor references
    known_landmarks = [
        "phagwara", "banga road", "gt road", "g.t. road", "nh44", "nh 44",
        "national highway", "highway", "flyover", "bypass", "chaheru",
        "sugar mill", "railway crossing", "bus stand", "civil hospital",
        "parhar hospital", "gandhi hospital", "jalandhar road", "petrol pump"
    ]

    for lm in known_landmarks:
        if re.search(rf"\b{lm}\b", text):
            locations.append(lm.title())

    # Prepositional location phrases: "near ...", "at ...", "on ..."
    prep_matches = re.findall(r"\b(?:near|at|on|towards)\s+([a-z0-9\s]{3,25}?)(?:,|\.|\band\b|\bwith\b|\bwhere\b|$)", text)
    for pm in prep_matches:
        cleaned_pm = pm.strip()
        if len(cleaned_pm) > 3 and cleaned_pm not in [l.lower() for l in locations]:
            # Filter out non-location words
            if not any(stop in cleaned_pm for stop in ["the car", "a vehicle", "someone", "people"]):
                locations.append(cleaned_pm.title())

    return list(dict.fromkeys(locations))  # Deduplicate preserving order
