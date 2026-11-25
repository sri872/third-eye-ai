"""Tab 5 regressions: honest facility classification, ranking, location and routing.

These tests deliberately avoid the embedding model: a tiny fake index returns
controlled candidates so ranking and display rules can be checked directly.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.dispatch_ui import DISPATCH_SUMMARY_ENV
from app.hospital_ui import (
    general_emergency_rows,
    hospital_number_lines,
    recommendation_rows,
)
from core.rag.emergency_numbers import get_emergency_numbers
from core.rag.facility_classification import (
    CATEGORY_CLINIC,
    CATEGORY_HOSPITAL,
    CATEGORY_TRAUMA_CENTRE,
    CATEGORY_UNRELATED,
    classify_facility,
    dispatch_eligible_facilities,
)
from core.rag.hospital_data import fetch_or_load_hospitals
from core.rag.location_consistency import compare_locations
from core.rag.retriever import HospitalRetriever
from core.schemas import Hospital

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_FILE = REPO_ROOT / "app" / "main.py"
OSM_CACHE = REPO_ROOT / "data" / "hospitals_osm.json"


class FakeIndex:
    """Minimal index stub; no model loading and no database access."""

    def __init__(self, candidates):
        self.candidates = list(candidates)

    def search(self, query, top_k=10):
        return self.candidates[:top_k]


def _hospital(
    identifier: str,
    name: str,
    lat: float,
    lon: float,
    *,
    emergency: str | None = None,
    phone: str | None = None,
    specialties: str | None = None,
) -> Hospital:
    return Hospital(
        hospital_id=identifier,
        name=name,
        latitude=lat,
        longitude=lon,
        emergency=emergency,
        contact_number=phone,
        specialties=specialties,
        data_source="REAL",
    )


def _retrieve(candidates, lat=31.2142, lon=75.7785, query="emergency trauma"):
    retriever = HospitalRetriever(FakeIndex(candidates))
    return retriever.retrieve_suggestions(
        incident_lat=lat,
        incident_lon=lon,
        query_override=query,
        final_top_k=3,
    )


# 1. Irrelevant facility exclusion
def test_lpu_architecture_and_unrelated_facilities_never_qualified() -> None:
    elements = json.loads(OSM_CACHE.read_text(encoding="utf-8"))["elements"]
    lpu = next(element for element in elements if str(element.get("id")) == "832187216")
    assert lpu["tags"]["name"] == "LPU Architecture Deisgn"
    # The real cache mis-tags this building as healthcare=hospital.
    assert lpu["tags"]["healthcare"] == "hospital"

    result = classify_facility(lpu["tags"]["name"], lpu["tags"])
    assert result.category == CATEGORY_UNRELATED
    assert result.is_healthcare is False
    assert result.dispatch_eligible is False
    assert "Non-medical" in result.reason

    for name in (
        "LPU Architecture Design Department",
        "Lovely Professional University Campus",
        "Civil Hospital Business Office",
        "Medical Store near Civil Hospital",
        "Hospital Design Block",
    ):
        classified = classify_facility(name, {"healthcare": "hospital"})
        assert classified.dispatch_eligible is False, name
        assert classified.category == CATEGORY_UNRELATED, name

    # Classification cannot use proximity: no coordinates are part of its input.
    parameters = inspect.signature(classify_facility).parameters
    assert not any(
        key in parameters for key in ("latitude", "longitude", "distance_km", "incident_lat")
    )


# 2. Valid hospital / trauma / clinic classification
def test_valid_healthcare_categories_are_distinguished() -> None:
    hospital = classify_facility("Government Medical College and Hospital")
    assert hospital.category == CATEGORY_HOSPITAL
    assert hospital.dispatch_eligible is True
    assert hospital.reason is None

    concatenated = classify_facility("civilhospital")
    assert concatenated.category == CATEGORY_HOSPITAL
    assert concatenated.dispatch_eligible is True

    trauma = classify_facility("Civil Trauma Centre", {"healthcare": "hospital"})
    assert trauma.category == CATEGORY_TRAUMA_CENTRE
    assert trauma.dispatch_eligible is True

    clinic = classify_facility("Dr Aman Clinic")
    assert clinic.category == CATEGORY_CLINIC
    assert clinic.is_healthcare is True
    assert clinic.dispatch_eligible is False


# 3. Real cache: raw count stays intact, irrelevant records are excluded before ranking
def test_real_cache_keeps_raw_count_and_attribution_but_excludes_irrelevant() -> None:
    hospitals, stats, source = fetch_or_load_hospitals(save_to_db=False)
    assert stats["total_count"] == 115
    assert all(hospital.attribution == "© OpenStreetMap contributors" for hospital in hospitals)

    eligible = dispatch_eligible_facilities(hospitals)
    assert len(eligible) < len(hospitals)
    assert not any("Architecture" in hospital.name or "Deisgn" in hospital.name for hospital in eligible)
    assert not any("University" in hospital.name for hospital in eligible)
    assert not any("Clinic" in hospital.name for hospital in eligible)
    assert "CACHE" in source or "OVERPASS" in source


# 4. Missing phone / emergency / specialty fields
def test_missing_fields_are_explicit_and_never_assumed() -> None:
    candidate = _hospital("osm-node-1", "Gandhi Hospital (P) Ltd", 31.2166, 75.7767)
    result = _retrieve([(candidate, 0.8)])
    suggestion = result["suggestions"][0]

    assert suggestion["phone"] == "not listed"
    assert "missing" in suggestion["phone_status"]
    assert suggestion["emergency"] == "not specified"
    assert "unavailable/unverified" in suggestion["emergency_suitability"]
    assert suggestion["specialties"] == "not specified in OpenStreetMap"
    assert suggestion["specialties_listed"] == []
    assert suggestion["recommended_for_emergency"] is False
    assert suggestion["attribution"] == "© OpenStreetMap contributors"
    assert "none assumed" in suggestion["why_suggested"]


# 5. Emergency tag honesty
def test_emergency_tag_is_source_reported_and_not_24x7_verified() -> None:
    candidate = _hospital(
        "osm-node-2", "Modern Hospital, Jalandhar", 31.1, 75.6, emergency="yes"
    )
    suggestion = _retrieve([(candidate, 0.5)])["suggestions"][0]
    assert suggestion["emergency_status"] == "SOURCE_CONFIRMED"
    assert "not independently verified" in suggestion["emergency_suitability"]
    assert "24/7 not established" in suggestion["emergency_suitability"]
    assert "verified 24/7 emergency department" not in suggestion["why_suggested"]


# 6. Ranking priority
def test_source_confirmed_emergency_outweighs_close_unverified_facility() -> None:
    near_unverified = _hospital("osm-node-3", "Gandhi Hospital (P) Ltd", 31.2143, 75.7786)
    far_confirmed = _hospital(
        "osm-node-4", "Modern Hospital, Jalandhar", 31.45, 75.95, emergency="yes"
    )
    # Fake search deliberately returns the near, semantically stronger facility first.
    result = _retrieve([(near_unverified, 0.95), (far_confirmed, 0.20)])
    names = [item["hospital"].name for item in result["suggestions"]]

    assert names[0] == "Modern Hospital, Jalandhar"
    assert result["suggestions"][0]["distance_km"] > result["suggestions"][1]["distance_km"]
    assert "heuristic" in result["suggestions"][0]["score_label"]


def test_trauma_centre_ranks_before_plain_hospital_with_same_emergency_tag() -> None:
    hospital = _hospital("osm-node-5", "Civil Hospital Phagwara", 31.23, 75.77, emergency="yes")
    trauma = _hospital("osm-node-6", "Civil Trauma Centre", 31.25, 75.79, emergency="yes")
    result = _retrieve([(hospital, 0.9), (trauma, 0.3)])
    assert result["suggestions"][0]["facility_category"] == CATEGORY_TRAUMA_CENTRE
    assert result["suggestions"][1]["facility_category"] == CATEGORY_HOSPITAL


# 7. Location consistency
def test_location_mismatch_keeps_selected_and_caller_locations_separate() -> None:
    selected = {
        "name": "NH-44 GT Road near Jagatjit Sugar Mill",
        "latitude": 31.2142,
        "longitude": 75.7785,
    }
    caller = "MG Road near Trinity Metro station"
    check = compare_locations(selected, caller)

    assert check.status == "LOCATION MISMATCH"
    assert check.selected_name == selected["name"]
    assert check.selected_coordinates.startswith("31.2142, 75.7785")
    assert check.caller_reported == caller
    assert "not substituted" in check.detail.lower()
    assert "no device gps verification" in check.gps_status.lower()

    missing = compare_locations(selected, None)
    assert missing.status == "CALLER LOCATION NOT REPORTED"


# 8. Routing limitations
def test_road_distance_and_travel_time_are_not_fabricated() -> None:
    candidate = _hospital("osm-node-7", "Civil Hospital Phagwara", 31.23, 75.77)
    result = _retrieve([(candidate, 0.7)])
    suggestion = result["suggestions"][0]

    assert result["road_distance_km"].startswith("unavailable")
    assert result["driving_time_min"].startswith("unavailable")
    assert suggestion["road_distance_km"].startswith("unavailable")
    assert suggestion["driving_time_min"].startswith("unavailable")
    assert "road distance and travel time are unavailable" in result["limitations_note"].lower()
    assert "not driving distances" in result["limitations_note"].lower()
    assert "straight-line" in suggestion["distance_label"]


# 9. Readable card rendering and contact separation
def test_card_rows_and_contact_rows_are_readable_and_labelled() -> None:
    candidate = _hospital("osm-node-8", "Civil Hospital Phagwara", 31.23, 75.77)
    suggestion = _retrieve([(candidate, 0.7)])["suggestions"][0]
    row = recommendation_rows([suggestion])[0]

    assert row["Facility"] == "Civil Hospital Phagwara"
    assert row["Category"] == CATEGORY_HOSPITAL
    assert "straight-line, not driving distance" in row["Straight-line distance"].lower()
    assert "SIMULATED" in row["Straight-line distance"]
    assert "not listed" in row["Phone (hospital-specific, OSM)"]
    assert row["Road distance / driving time"].startswith("unavailable")
    assert "NOT RECOMMENDED" in row["Emergency suitability"]
    assert "heuristic" in row["Rank basis"]
    assert hospital_number_lines([suggestion])[0].endswith("(missing in OSM; unverified)")

    general = general_emergency_rows(get_emergency_numbers(urgency=None, hazards=[]))
    numbers = {item["Number"] for item in general}
    assert {"112", "108"}.issubset(numbers)
    assert all(item["Status"] == "verified" for item in general)
    assert all("hospital" not in item["Service"].lower() for item in general)


# 10. No-upload UI still shows dedicated limitations and general numbers
def test_tab5_no_upload_shows_limitations_and_general_emergency_numbers(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DISPATCH_SUMMARY_ENV, str(tmp_path / "missing_summary.json"))
    at = AppTest.from_file(str(APP_FILE), default_timeout=300).run()
    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]

    hospitals_tab = next(tab for tab in at.tabs if tab.label.startswith("5."))
    text = " ".join(
        str(getattr(node, "value", ""))
        for kind in ("markdown", "text", "caption", "info", "warning")
        for node in hospitals_tab.get(kind)
    )
    table_text = " ".join(
        str(node.value.to_string()) for node in hospitals_tab.table
    )
    text = f"{text} {table_text}"
    assert "General —" in text
    assert "112" in text and "108" in text
    assert "NOT hospital-specific" in text
    assert "Hospital-specific" in text
    assert "not reported" in text
    assert "no device GPS verification" in text
    assert "road distance and estimated driving time are unavailable" in text
