"""Unit and integration tests for deterministic hospital suggestions and geographic routing.

Tests:
1. Haversine distance calculation with known geographic benchmarks.
2. Re-ranking: closer facilities receive proximity boost over distant facilities.
3. Hospital schema validation: coordinate bounds, bed constraints, optional beds.
4. Database persistence: saving and retrieving Hospital records in SQLite.
5. Missing phone behavior: reports 'not listed', never crashes or fabricates numbers.
6. Bed availability honesty: verifies 'not determined' is strictly maintained.
7. Overpass API failure fallback: graceful fallback to local cache upon network error.
8. Emergency numbers verification and triage mapping.
"""

from pathlib import Path
import sqlite3
from unittest.mock import patch
import pytest

from core.rag.emergency_numbers import EMERGENCY_SERVICES, get_emergency_numbers
from core.rag.hospital_data import (
    DEFAULT_CACHE_PATH,
    compute_coverage_statistics,
    fetch_or_load_hospitals,
    parse_osm_element_to_hospital,
)
from core.rag.index import HospitalIndex, get_embedding_model
from core.rag.retriever import (
    HospitalRetriever,
    explain_match,
    formulate_retrieval_query,
    haversine_distance_km,
    score_hospital_candidate,
)
from core.schemas import Hospital
from db.database import get_hospitals, init_db, save_hospitals


@pytest.fixture
def temp_hospital_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_hospital_search.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def sample_hospitals() -> list[Hospital]:
    return [
        Hospital(
            hospital_id="osm-node-101",
            name="Phagwara Emergency Civil Hospital",
            latitude=31.2200,
            longitude=75.7700,
            emergency="yes",
            specialties="emergency;trauma;general",
            contact_number="01824-222222",
            address="GT Road, Phagwara",
            data_source="REAL",
        ),
        Hospital(
            hospital_id="osm-node-102",
            name="Sharma Pediatric Hospital",
            latitude=31.2250,
            longitude=75.7750,
            emergency="no",
            specialties="pediatrics;neonatal",
            contact_number=None,
            address="Model Town, Phagwara",
            data_source="REAL",
        ),
        Hospital(
            hospital_id="osm-node-103",
            name="Distant Trauma Research Center",
            latitude=31.4500,
            longitude=75.9500,
            emergency="yes",
            specialties="trauma;neuro;cardiac",
            contact_number="+91-98765-43210",
            address="Suburban Outpost",
            data_source="REAL",
        ),
    ]


# 1. Haversine Distance Test
def test_haversine_distance_accuracy():
    # London (51.5074, -0.1278) to Paris (48.8566, 2.3522) ~ 343.5 km
    dist = haversine_distance_km(51.5074, -0.1278, 48.8566, 2.3522)
    assert 340.0 <= dist <= 346.0

    # Distance to identical point is zero
    assert haversine_distance_km(31.2200, 75.7700, 31.2200, 75.7700) == 0.0

    # Phagwara center to Chaheru (LPU) ~ 7.25 km
    phagwara_to_lpu = haversine_distance_km(31.2207, 75.7696, 31.2532, 75.7035)
    assert 7.0 <= phagwara_to_lpu <= 8.5


# 2. Re-Ranking Proximity Boost Test
def test_deterministic_proximity_and_emergency_tag_ranking():
    # Candidate A: same legacy score argument, distance 2 km
    combined_a, prox_a, legacy_a = score_hospital_candidate(semantic_score=0.8, distance_km=2.0)
    # Candidate B: same legacy score argument, distance 20 km
    combined_b, prox_b, legacy_b = score_hospital_candidate(semantic_score=0.8, distance_km=20.0)

    assert prox_a > prox_b
    assert combined_a > combined_b
    assert legacy_a == legacy_b == 0.0
    assert combined_a - combined_b > 0.15  # Notable proximity advantage

    # Emergency tag boost test
    combined_emerg, _, _ = score_hospital_candidate(semantic_score=0.5, distance_km=5.0, emergency_tag="yes")
    combined_no_emerg, _, _ = score_hospital_candidate(semantic_score=0.5, distance_km=5.0, emergency_tag="no")
    assert combined_emerg > combined_no_emerg


# 3. Hospital Schema Validation Test
def test_hospital_schema_validation():
    # Valid Hospital with None beds
    h = Hospital(
        hospital_id="osm-node-1",
        name="Valid Hospital",
        latitude=31.22,
        longitude=75.77,
        data_source="REAL",
    )
    assert h.total_beds is None
    assert h.available_beds is None
    assert h.data_source == "REAL"

    # Invalid latitude raises ValueError
    with pytest.raises(ValueError):
        Hospital(
            hospital_id="osm-node-2",
            name="Invalid Lat Hospital",
            latitude=195.0,
            longitude=75.77,
            data_source="REAL",
        )

    # Invalid available_beds > total_beds raises ValueError
    with pytest.raises(ValueError, match="available_beds .* cannot exceed total_beds"):
        Hospital(
            hospital_id="osm-node-3",
            name="Bed Violation Hospital",
            latitude=31.22,
            longitude=75.77,
            total_beds=50,
            available_beds=60,
            data_source="SIMULATED",
        )


# 4. Database Persistence Test
def test_save_and_retrieve_hospital_records(temp_hospital_db: Path, sample_hospitals: list[Hospital]):
    count = save_hospitals(sample_hospitals, db_path=temp_hospital_db)
    assert count == 3

    rows = get_hospitals(db_path=temp_hospital_db)
    assert len(rows) == 3

    # Check stored values
    hosp_map = {r["hospital_id"]: r for r in rows}
    assert hosp_map["osm-node-101"]["name"] == "Phagwara Emergency Civil Hospital"
    assert hosp_map["osm-node-101"]["emergency"] == "yes"
    assert hosp_map["osm-node-101"]["total_beds"] is None
    assert hosp_map["osm-node-102"]["contact_number"] is None


# 5. Missing Phone Number Test
def test_missing_phone_reports_not_listed(sample_hospitals: list[Hospital]):
    index = HospitalIndex(sample_hospitals)
    retriever = HospitalRetriever(index)

    # Search near pediatric hospital
    res = retriever.retrieve_suggestions(
        incident_lat=31.2250,
        incident_lon=75.7750,
        query_override="Child fever and infant pediatric emergency",
        final_top_k=3,
    )
    assert len(res["suggestions"]) == 3

    # Find the hospital without phone
    ped_sug = next(s for s in res["suggestions"] if s["hospital"].hospital_id == "osm-node-102")
    assert ped_sug["phone"] == "not listed"
    assert ped_sug["hospital"].contact_number is None


# 6. Bed Availability Honesty Test
def test_bed_availability_honesty(sample_hospitals: list[Hospital]):
    index = HospitalIndex(sample_hospitals)
    retriever = HospitalRetriever(index)

    res = retriever.retrieve_suggestions(
        incident_lat=31.2200,
        incident_lon=75.7700,
        query_override="Emergency acute trauma crash",
        final_top_k=3,
    )
    for sug in res["suggestions"]:
        assert sug["bed_availability"] == "not determined"
        assert sug["hospital"].total_beds is None
        assert sug["hospital"].available_beds is None
        assert "straight-line, not driving distance" in sug["distance_label"]
        assert sug["attribution"] == "© OpenStreetMap contributors"


# 7. Overpass Network Failure Cache Fallback Test
def test_overpass_failure_fallback():
    # Mock fetch_from_overpass to simulate timeout/connection failure
    with patch("core.rag.hospital_data.fetch_from_overpass") as mock_fetch:
        mock_fetch.return_value = (None, "502 Gateway Timeout: Server Overloaded")
        hospitals, stats, source = fetch_or_load_hospitals(force_refresh=True, save_to_db=False)

        # Must fall back gracefully to local cache (115 after deduplication)
        assert len(hospitals) == 115
        assert "CACHE_FALLBACK" in source
        assert "502 Gateway Timeout" in source
        assert stats["total_count"] == 115


# 8. Emergency Numbers Verification Test
def test_emergency_numbers_verified():
    # Check that universal 112 and medical 108 are always present
    contacts = get_emergency_numbers(urgency="critical", hazards=["fire", "fuel spill"])
    numbers = [c["number"] for c in contacts]
    assert "112" in numbers
    assert "108" in numbers
    assert "101" in numbers  # Fire service added due to fire hazard
    assert "100" in numbers  # Police service

    for contact in contacts:
        assert contact["source_url"].startswith("https://")
        assert len(contact["official_source"]) > 5


def test_embedding_retrieval_is_disabled():
    with pytest.raises(RuntimeError, match="Embedding retrieval has been removed"):
        get_embedding_model()
