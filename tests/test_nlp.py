"""Tests for Deterministic NLP Incident Analysis.

Verifies text normalization, entity extraction, incident classification,
urgency scoring, contradiction detection, and structured reporting.
"""

import pytest

from core.nlp.text_normalization import normalize_text, tokenize_words, split_sentences
from core.nlp.entity_extraction import extract_entities
from core.nlp.incident_analysis import analyze_incident
from core.nlp.report_generation import generate_structured_report


def test_normalize_text_and_typos():
    """Test text cleaning and phonetic typo correction."""
    raw = "Major   acident on hiway, ambulence needed immediately! Vehical is overturnd."
    normalized = normalize_text(raw)
    assert "accident" in normalized
    assert "highway" in normalized
    assert "ambulance" in normalized
    assert "vehicle" in normalized
    assert "overturned" in normalized
    assert "   " not in normalized


def test_tokenize_words():
    """Test word tokenization."""
    words = tokenize_words("Two cars, one red bus.")
    assert words == ["two", "cars", "one", "red", "bus"]


def test_entity_extraction_vehicles_and_locations():
    """Test extracting vehicles and location keywords."""
    text = "A speeding red truck collided with two white cars on GT Road near Phagwara."
    entities = extract_entities(text)
    assert entities["total_vehicles_reported"] >= 3
    assert any(v["type"] == "truck" for v in entities["vehicles"])
    assert any(v["type"] == "car" for v in entities["vehicles"])
    assert "Gt Road" in entities["locations"] or "Phagwara" in entities["locations"]


def test_entity_extraction_injuries_and_hazards():
    """Test extracting injury and hazard categories."""
    text = "Head-on crash with unconscious driver, severe bleeding, and petrol leak."
    entities = extract_entities(text)
    inj_cats = [i["category"] for i in entities["injuries"]]
    haz_cats = [h["category"] for h in entities["hazards"]]
    assert "unconscious" in inj_cats
    assert "bleeding" in inj_cats
    assert "fuel_leak" in haz_cats


def test_negation_handling():
    """Test that explicit negation prevents false injury or hazard alerts."""
    text = "Fender bender on Banga Road. No injuries and no fire."
    entities = extract_entities(text)
    assert entities["has_negated_injuries"] is True
    assert entities["has_negated_hazards"] is True
    assert len(entities["injuries"]) == 0
    assert len(entities["hazards"]) == 0


def test_urgency_levels():
    """Test urgency scoring rules."""
    # Critical: Unconscious or fatal
    crit = analyze_incident("Car crash on highway, passenger is unconscious and trapped.")
    assert crit["urgency_level"] == "CRITICAL"

    # High: Fracture or bleeding or fuel leak
    high = analyze_incident("Motorcycle accident with fractured leg and bleeding on road.")
    assert high["urgency_level"] == "HIGH"

    # Low: No injuries or minor scratches
    low = analyze_incident("Minor car bumper scratch in parking lot. No injuries.")
    assert low["urgency_level"] == "LOW"


def test_contradiction_detection():
    """Test detection of contradictory claims in incident text."""
    contradictory_text = "The driver says no injuries, but the passenger has severe bleeding and is unconscious."
    res = analyze_incident(contradictory_text)
    assert res["has_contradictions"] is True
    assert len(res["contradictions_detected"]) > 0


def test_structured_report_generation():
    """Test report generation combining CV, OCR, and NLP."""
    nlp_res = analyze_incident("Two cars collided on GT Road. Driver has broken bone.")
    cv_diag = {"width": 1280, "height": 720, "lighting_condition": "NORMAL_DAYLIGHT", "sharpness_laplacian_var": 210.5}
    cv_cands = [{"bbox": [100, 100, 200, 150]}, {"bbox": [400, 120, 180, 140]}]
    ocr_res = {"status": "VERIFIED", "raw_text": "TS07JA0566", "normalized_text": "TS07JA0566", "mean_similarity": 0.88, "is_verified": True, "validation": {"state_name": "Telangana"}}

    report = generate_structured_report(
        cv_diagnostics=cv_diag,
        cv_candidates=cv_cands,
        plate_ocr_result=ocr_res,
        nlp_analysis=nlp_res,
    )
    assert "markdown_report" in report
    md = report["markdown_report"]
    assert "[REAL CV]" in md
    assert "[CLASSICAL OCR]" in md
    assert "[CALLER REPORTED]" in md
    assert "TS07JA0566" in md
    assert "AGREED: Visual candidate count (2) matches caller report (2)" in md


def test_negation_is_scoped_to_the_matching_injury_mention():
    """A general denial must not erase a later positive injury assertion."""
    result = analyze_incident(
        "Caller first says no injuries, but the driver is bleeding."
    )
    entities = result["entities"]
    assert entities["has_negated_injuries"] is True
    assert any(item["category"] == "bleeding" for item in entities["injuries"])
    assert result["has_contradictions"] is True
    assert result["urgency_level"] == "HIGH"


def test_negation_does_not_cross_contrast_into_an_unrelated_hazard():
    """'No fire, but fuel leak' reports a fuel leak without reporting fire."""
    result = analyze_incident("There is no fire, but a petrol leak beside the car.")
    entities = result["entities"]
    assert not any(item["category"] == "fire" for item in entities["hazards"])
    assert any(item["category"] == "fuel_leak" for item in entities["hazards"])
    assert not result["has_contradictions"]
    assert result["urgency_level"] == "HIGH"


def test_same_hazard_both_denied_and_reported_is_a_contradiction():
    result = analyze_incident("Caller says no fire, but witnesses report fire.")
    assert result["has_contradictions"] is True
    assert any(item["category"] == "fire" for item in result["entities"]["hazards"])
