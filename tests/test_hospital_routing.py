"""Hospital catalog and directed road-routing regression tests."""
from __future__ import annotations

import networkx as nx
import pytest

from core.sos.location import SOURCE_SIMULATED_DEMO, create_sos_location
from core.hospitals.catalog import HospitalCatalog
from core.hospitals.ranker import HospitalRanker
from core.routing.router import RoadNetworkRouter, RouteUnavailableError


def make_graph():
    graph = nx.MultiDiGraph()
    graph.add_node("a", x=75.7700, y=31.2200)
    graph.add_node("b", x=75.7710, y=31.2200)
    graph.add_node("c", x=75.7720, y=31.2200)
    graph.add_node("d", x=75.7700, y=31.2220)
    graph.add_edge("a", "b", key="ab", length=100.0, travel_time=10.0)
    graph.add_edge("b", "c", key="bc-fast", length=200.0, travel_time=20.0)
    graph.add_edge("b", "c", key="bc-short-slow", length=150.0, travel_time=50.0)
    graph.add_edge("a", "d", key="ad", length=500.0, travel_time=50.0)
    graph.add_edge("d", "c", key="dc", length=500.0, travel_time=50.0)
    return graph


def make_sos(lat=31.2200, lon=75.7700):
    location, message = create_sos_location(lat, lon, source=SOURCE_SIMULATED_DEMO)
    assert location is not None, message
    return location


def test_catalog_loads_only_source_tagged_hospitals():
    catalog = HospitalCatalog()
    facilities = catalog.get_eligible_facilities()
    assert len(facilities) > 0
    assert all(item["category"] == "hospital" for item in facilities)
    assert all(item["source_url"].startswith("https://www.openstreetmap.org/") for item in facilities)
    assert all("India" in item["address"] or "Address not listed" in item["address"] for item in facilities)
    for item in facilities:
        assert item["has_documented_emergency"] is (item["emergency_tag"] == "yes")


def test_road_route_uses_directed_graph_and_edge_travel_time():
    router = RoadNetworkRouter(graph=make_graph())
    route = router.calculate_route(31.2200, 75.7700, 31.2200, 75.7720)
    assert route["status"] == "ROAD_NETWORK_ROUTE_FOUND"
    assert route["is_road_network"] is True
    assert route["route_geometry"][0] == (31.2200, 75.7700)
    assert route["route_geometry"][-1] == (31.2200, 75.7720)
    assert route["road_graph_distance_km"] == pytest.approx(0.3)
    assert route["estimated_time_minutes"] == pytest.approx(0.5, abs=0.1)
    assert route["route_source"] == "SAVED_OPENSTREETMAP_DIRECTED_GRAPH"


def test_road_router_does_not_reverse_one_way_edges():
    router = RoadNetworkRouter(graph=make_graph())
    with pytest.raises(RouteUnavailableError, match="No directed road path"):
        router.calculate_route(31.2200, 75.7720, 31.2200, 75.7700)


def test_road_router_rejects_coordinates_outside_graph_coverage():
    router = RoadNetworkRouter(graph=make_graph(), max_snap_distance_m=500)
    with pytest.raises(RouteUnavailableError, match="outside saved road-graph coverage"):
        router.calculate_route(25.0, 80.0, 31.2200, 75.7720)


def test_ranker_filters_clinics_and_downranks_unmatched_specialty_hints():
    class FakeCatalog:
        def get_eligible_facilities(self, require_hospital_tier=True):
            return [
                {
                    "id": "hospital-1", "name": "General Hospital",
                    "latitude": 31.2200, "longitude": 75.7720,
                    "category": "hospital", "has_documented_emergency": False,
                    "speciality": None, "emergency_tag": None, "phone": None,
                    "address": "Test Road, Phagwara, Punjab, India",
                },
                {
                    "id": "hospital-2", "name": "Eye Hospital",
                    "latitude": 31.2200, "longitude": 75.7720,
                    "category": "hospital", "has_documented_emergency": True,
                    "speciality": "ophthalmology", "emergency_tag": "yes", "phone": None,
                    "address": "Test Road, Phagwara, Punjab, India",
                },
                {
                    "id": "clinic-1", "name": "Dental Clinic",
                    "latitude": 31.2200, "longitude": 75.7720,
                    "category": "clinic", "has_documented_emergency": False,
                    "speciality": "dental", "emergency_tag": None, "phone": None,
                    "address": "Test Road, Phagwara, Punjab, India",
                },
            ]

    class FakeRouter:
        def calculate_route(self, start_lat, start_lon, dest_lat, dest_lon):
            return {
                "is_road_network": True,
                "status": "ROAD_NETWORK_ROUTE_FOUND",
                "distance_km": 1.0,
                "estimated_time_minutes": 2.0,
                "google_maps_url": "https://maps.example/route",
                "route_source": "TEST_GRAPH",
            }

    ranker = HospitalRanker(catalog=FakeCatalog(), router=FakeRouter())
    result = ranker.rank_hospitals(make_sos(), incident_text="two cars collided on highway")
    assert result["primary_recommendation"]["facility"]["name"] == "General Hospital"
    assert all(
        item["facility"]["category"] == "hospital"
        for item in [result["primary_recommendation"], *result["alternatives"]]
    )


def test_ranker_skips_hospital_without_a_directed_road_route():
    class FakeCatalog:
        def get_eligible_facilities(self, require_hospital_tier=True):
            return [
                {
                    "id": "reachable", "name": "Reachable Hospital",
                    "latitude": 31.22, "longitude": 75.77, "category": "hospital",
                    "has_documented_emergency": False, "speciality": None,
                    "emergency_tag": None, "phone": None, "address": "Phagwara, India",
                },
                {
                    "id": "unreachable", "name": "Unreachable Hospital",
                    "latitude": 31.23, "longitude": 75.78, "category": "hospital",
                    "has_documented_emergency": False, "speciality": None,
                    "emergency_tag": None, "phone": None, "address": "Phagwara, India",
                },
            ]

    class FakeRouter:
        def calculate_route(self, start_lat, start_lon, dest_lat, dest_lon):
            if dest_lat > 31.225:
                raise RouteUnavailableError("No directed road path")
            return {
                "is_road_network": True,
                "distance_km": 1.2,
                "estimated_time_minutes": 3.0,
                "google_maps_url": "https://maps.example/route",
                "route_source": "TEST_GRAPH",
            }

    result = HospitalRanker(
        catalog=FakeCatalog(), router=FakeRouter()
    ).rank_hospitals(make_sos())
    assert result["primary_recommendation"]["facility"]["name"] == "Reachable Hospital"
    assert result["unroutable_hospitals"][0]["facility"] == "Unreachable Hospital"


def test_ranker_blocks_stale_sos_location():
    old = "2020-01-01T00:00:00+00:00"
    location, message = create_sos_location(
        31.22, 75.77, timestamp=old, source=SOURCE_SIMULATED_DEMO
    )
    assert location is not None, message
    result = HospitalRanker(
        catalog=object(), router=object()
    ).rank_hospitals(location)
    assert result["error"].startswith("SOS location is older than 15 minutes")
    assert result["primary_recommendation"] is None


def test_ranker_rejects_missing_sos_coordinates():
    result = HospitalRanker(catalog=object(), router=object()).rank_hospitals(None)
    assert "error" in result
    assert result["primary_recommendation"] is None


def test_critical_incident_prioritizes_documented_emergency_tag():
    class FakeCatalog:
        def get_eligible_facilities(self, require_hospital_tier=True):
            return [
                {
                    "id": "general", "name": "General Hospital",
                    "latitude": 31.2200, "longitude": 75.7720,
                    "category": "hospital", "has_documented_emergency": False,
                    "speciality": None, "emergency_tag": None, "phone": None,
                    "address": "General Road, Phagwara, Punjab, India",
                },
                {
                    "id": "tagged", "name": "Emergency Tagged Hospital",
                    "latitude": 31.2200, "longitude": 75.7720,
                    "category": "hospital", "has_documented_emergency": True,
                    "speciality": "ophthalmology", "emergency_tag": "yes", "phone": None,
                    "address": "Hospital Road, Phagwara, Punjab, India",
                },
            ]

    class FakeRouter:
        def calculate_route(self, start_lat, start_lon, dest_lat, dest_lon):
            return {
                "is_road_network": True,
                "distance_km": 1.0,
                "estimated_time_minutes": 2.0,
                "google_maps_url": "https://maps.example/route",
                "route_source": "TEST_GRAPH",
            }

    result = HospitalRanker(
        catalog=FakeCatalog(), router=FakeRouter()
    ).rank_hospitals(make_sos(), urgency_level="CRITICAL")
    assert result["primary_recommendation"]["facility"]["name"] == "Emergency Tagged Hospital"
    assert "CRITICAL" in result["ranking_method"]
