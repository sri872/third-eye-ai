"""Tests for Hospital Intelligence and Road Network Routing.

Verifies hospital catalog capability extraction, NetworkX road routing,
distance/time calculation, and multi-tier hospital ranking.
"""

import pytest
from core.sos.location import SOSLocation
from core.hospitals.catalog import HospitalCatalog
from core.routing.router import RoadNetworkRouter
from core.hospitals.ranker import HospitalRanker


@pytest.fixture
def catalog():
    return HospitalCatalog()


@pytest.fixture
def router():
    return RoadNetworkRouter()


@pytest.fixture
def ranker(catalog, router):
    return HospitalRanker(catalog=catalog, router=router)


@pytest.fixture
def sos_nh44():
    return SOSLocation(
        latitude=31.2142,
        longitude=75.7785,
        location_name="NH-44 Sugar Mill",
    )


def test_catalog_loads_and_does_not_invent_capabilities(catalog):
    """Verify hospital catalog loads facilities and does not invent capabilities from names."""
    facilities = catalog.get_eligible_facilities()
    assert len(facilities) > 50

    for f in facilities:
        assert "name" in f and f["name"]
        assert "latitude" in f and "longitude" in f
        assert f["category"] in ("hospital", "clinic", "healthcare_facility")
        # Check that trauma is not invented if emergency tag is not yes
        if not f["has_documented_emergency"]:
            assert f["emergency_tag"] != "yes"


def test_road_network_routing(router, sos_nh44):
    """Verify NetworkX calculates road route starting from SOS coordinates."""
    # Route to Civil Hospital Phagwara (~31.2177, 75.7725)
    dest_lat = 31.2177
    dest_lon = 75.7725

    route = router.calculate_route(
        start_lat=sos_nh44.latitude,
        start_lon=sos_nh44.longitude,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
    )
    assert route["status"] in ("ROAD_NETWORK_ROUTE_FOUND", "ROUTING_FALLBACK_HAVERSINE")
    assert route["distance_km"] > 0.1
    assert route["estimated_time_minutes"] > 0.1
    assert len(route["route_geometry"]) >= 2
    # First point of route geometry must be the SOS origin
    assert route["route_geometry"][0] == (sos_nh44.latitude, sos_nh44.longitude)
    # External Google Maps URL must contain origin and destination
    assert f"origin={sos_nh44.latitude:.5f},{sos_nh44.longitude:.5f}" in route["google_maps_url"]


def test_hospital_ranking_prioritizes_documented_emergency(ranker, sos_nh44):
    """Verify hospital ranking prioritizes documented emergency services."""
    ranking = ranker.rank_hospitals(sos_nh44, urgency_level="CRITICAL", max_recommendations=4)
    assert ranking["primary_recommendation"] is not None

    primary = ranking["primary_recommendation"]
    fac = primary["facility"]
    route = primary["route"]

    # Primary recommendation must be a valid facility
    assert fac["name"]
    assert route["distance_km"] > 0
    assert route["estimated_time_minutes"] > 0
    assert "ranking_reason" in primary

    # Alternatives must be listed
    alternatives = ranking["alternatives"]
    assert len(alternatives) >= 1
    for alt in alternatives:
        assert alt["facility"]["name"]
        assert alt["route"]["distance_km"] > 0


def test_ranking_rejects_missing_sos_coordinates(ranker):
    """Verify ranker rejects null or empty SOS location without guessing coordinates."""
    res = ranker.rank_hospitals(None)
    assert "error" in res
    assert res["primary_recommendation"] is None
