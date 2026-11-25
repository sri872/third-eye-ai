"""Unit and integration test suite for routing, dispatch, and hospital deduplication.

Verifies:
1. A* shortest-path equivalence to NetworkX across 20 random node pairs.
2. Route continuity: all consecutive nodes form valid directed edges in G.
3. Driving distance >= straight-line Haversine distance.
4. Nearest-ambulance choice algorithm on controlled graph.
5. SQLite schema and persistence for RouteResult and Ambulance.
6. Deduplication and suitability rules from Phase 5 review (Task 0).
7. Deterministic trajectory simulation.
"""

import json
from pathlib import Path
import random
import networkx as nx
import pytest

from core.hospitals.data import deduplicate_hospitals
from core.hospitals.ranking import (
    describe_assumed_needs,
    evaluate_facility_suitability,
    haversine_distance_km,
)
from core.routing.astar import astar_shortest_path, extract_route_coordinates
from core.routing.dispatch import (
    get_candidate_hospitals,
    load_simulated_ambulances,
    plan_emergency_dispatch,
    route_geometry_audit,
)
from core.routing.road_graph import (
    get_graph_stats,
    get_nearest_node,
    haversine_distance_m,
    load_graphml_network,
    prepare_network_graph,
)
from core.routing.simulate import simulate_dispatch_trajectory, simulate_leg_trajectory
from core.schemas import Ambulance, Hospital, RouteResult
from db.database import (
    get_ambulances,
    get_route_results,
    init_db,
    save_ambulances,
    save_route_result,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def road_graph():
    """Load cached road network graph."""
    return prepare_network_graph()


class TestAStarRouting:
    """Tests verifying A* algorithm correctness and admissibility."""

    def test_astar_matches_networkx_on_20_random_pairs(self, road_graph):
        """Verify A* produces travel time matching NetworkX shortest path within 1e-4s."""
        scc = max(nx.strongly_connected_components(road_graph), key=len)
        nodes = list(scc)
        random.seed(42)

        tested_pairs = 0
        for _ in range(30):
            s = random.choice(nodes)
            t = random.choice(nodes)
            if s == t:
                continue

            a_path, a_time, a_dist = astar_shortest_path(road_graph, s, t, weight="travel_time")
            nx_time = nx.shortest_path_length(road_graph, s, t, weight="travel_time")

            assert abs(a_time - nx_time) < 1e-4, f"Mismatch on {s} -> {t}: A*={a_time}, NX={nx_time}"
            assert len(a_path) >= 2
            assert a_path[0] == s
            assert a_path[-1] == t
            assert a_dist > 0.0

            tested_pairs += 1
            if tested_pairs >= 20:
                break

        assert tested_pairs == 20

    def test_route_continuity(self, road_graph):
        """Verify that every consecutive node pair in the returned path is a valid edge."""
        scc = max(nx.strongly_connected_components(road_graph), key=len)
        nodes = list(scc)
        s, t = nodes[0], nodes[10]

        path, travel_time, distance = astar_shortest_path(road_graph, s, t)
        assert len(path) >= 2

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            assert road_graph.has_edge(u, v), f"Edge ({u}, {v}) does not exist in graph"

    def test_astar_identical_source_and_target(self, road_graph):
        """Verify path between a node and itself is trivial."""
        node = next(iter(road_graph.nodes))
        path, time_s, dist_m = astar_shortest_path(road_graph, node, node)
        assert path == [node]
        assert time_s == 0.0
        assert dist_m == 0.0

    def test_astar_disconnected_returns_empty(self):
        """Verify unreachable targets return empty path and infinite time."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=75.0, y=31.0)
        g.add_node(2, x=75.1, y=31.1)
        path, time_s, dist_m = astar_shortest_path(g, 1, 2)
        assert path == []
        assert time_s == float("inf")
        assert dist_m == 0.0


class TestRoadNetworkProperties:
    """Tests verifying graph properties and metric relationships."""

    def test_driving_distance_greater_than_or_equal_straight_line(self, road_graph):
        """Verify driving distance is always >= straight-line Haversine distance."""
        scc = max(nx.strongly_connected_components(road_graph), key=len)
        nodes = list(scc)
        random.seed(99)

        for _ in range(15):
            s = random.choice(nodes)
            t = random.choice(nodes)
            if s == t:
                continue

            path, time_s, dist_m = astar_shortest_path(road_graph, s, t)
            sl_dist_m = haversine_distance_m(
                road_graph.nodes[s]["y"], road_graph.nodes[s]["x"],
                road_graph.nodes[t]["y"], road_graph.nodes[t]["x"],
            )
            # Driving distance must be >= straight-line distance (with small epsilon for float precision)
            assert dist_m >= sl_dist_m - 0.5, f"Driving ({dist_m}m) < straight-line ({sl_dist_m}m)"

    def test_graph_stats_metadata(self, road_graph):
        """Verify graph summary stats include required OSM attribution."""
        stats = get_graph_stats(road_graph)
        assert stats["node_count"] > 1000
        assert stats["edge_count"] > 2000
        assert stats["total_road_length_km"] > 500.0
        assert stats["attribution"] == "© OpenStreetMap contributors"
        assert "Estimated from OSM speed data" in stats["disclaimer"]


class TestDispatchEngine:
    """Tests verifying dispatch logic and candidate selection."""

    def test_nearest_ambulance_selection_on_mock_graph(self):
        """Test that the ambulance with the lowest driving ETA is strictly chosen."""
        g = nx.MultiDiGraph()
        # Incident at (31.0, 75.0)
        g.add_node(100, y=31.0, x=75.0)
        # Ambulance 1 at (31.01, 75.0) - fast direct road (60s)
        g.add_node(1, y=31.01, x=75.0)
        g.add_edge(1, 100, length=1000.0, speed_kph=60.0, travel_time=60.0)

        # Ambulance 2 at (31.005, 75.0) - closer straight-line but congested slow road (180s)
        g.add_node(2, y=31.005, x=75.0)
        g.add_edge(2, 100, length=500.0, speed_kph=10.0, travel_time=180.0)

        # Hospital at (31.02, 75.0)
        g.add_node(200, y=31.02, x=75.0)
        g.add_edge(100, 200, length=2000.0, speed_kph=40.0, travel_time=180.0)

        ambulances = [
            {"ambulance_id": "amb-01", "call_sign": "AMB-1", "latitude": 31.01, "longitude": 75.0, "status": "available", "data_source": "SIMULATED"},
            {"ambulance_id": "amb-02", "call_sign": "AMB-2", "latitude": 31.005, "longitude": 75.0, "status": "available", "data_source": "SIMULATED"},
        ]
        hospitals = [
            Hospital(hospital_id="hosp-01", name="Trauma Base", latitude=31.02, longitude=75.0, data_source="REAL")
        ]

        result = plan_emergency_dispatch(
            incident_lat=31.0,
            incident_lon=75.0,
            accident_id="test-acc-01",
            graph=g,
            ambulances=ambulances,
            candidate_hospitals=hospitals,
        )

        # Amb-01 has 60s ETA vs Amb-02's 180s ETA -> Amb-01 must be chosen despite Amb-02 being physically closer
        assert result.ambulance_id == "amb-01"
        assert result.ambulance_eta_minutes == 1.0
        assert result.hospital_id == "hosp-01"

    def test_full_dispatch_on_demo_incident(self, road_graph):
        """Test end-to-end dispatch on demo incident location."""
        result = plan_emergency_dispatch(
            incident_lat=31.2142,
            incident_lon=75.7785,
            accident_id="loc-phagwara-nh44-sugarmill",
            caller_text="High-speed multi-car crash, heavy blood loss, head trauma.",
            graph=road_graph,
        )

        assert result.ambulance_id.startswith("amb-")
        assert result.hospital_id is not None
        assert result.ambulance_eta_minutes > 0.0
        assert result.hospital_transit_minutes > 0.0
        assert result.distance_km > 0.0
        assert result.driving_vs_straight_ratio >= 1.0
        assert len(result.alternative_ambulances) == 6
        assert len(result.alternative_hospitals) == 8
        assert result.attribution == "© OpenStreetMap contributors"
        assert result.data_source == "SIMULATED"


class TestSimulationTrajectory:
    """Tests for deterministic vehicle trajectory simulation."""

    def test_simulate_leg_trajectory(self):
        """Verify trajectory generator produces intermediate progress snapshots."""
        waypoints = [
            [31.2142, 75.7785],
            [31.2155, 75.7790],
            [31.2170, 75.7800],
        ]
        total_time_s = 60.0
        snapshots = simulate_leg_trajectory(waypoints, total_time_s, step_seconds=10.0)

        assert len(snapshots) >= 7  # 0, 10, 20, 30, 40, 50, 60s
        assert snapshots[0]["progress_pct"] == 0.0
        assert snapshots[-1]["progress_pct"] == 100.0
        for s in snapshots:
            assert s["data_source"] == "SIMULATED"
            assert "latitude" in s and "longitude" in s

    def test_simulate_dispatch_trajectory_covers_both_legs_once(self):
        """Both stored legs are replayed; the shared incident point appears once."""
        first = [[31.0000, 75.0000], [31.0010, 75.0000]]
        second = [[31.0010, 75.0000], [31.0020, 75.0000]]
        snapshots = simulate_dispatch_trajectory(
            first, 10.0, second, 20.0, step_seconds=10.0
        )

        assert [s["elapsed_sec"] for s in snapshots] == [0.0, 10.0, 20.0, 30.0]
        assert [s["leg"] for s in snapshots] == [
            "leg1_ambulance_to_incident",
            "leg1_ambulance_to_incident",
            "leg2_incident_to_hospital",
            "leg2_incident_to_hospital",
        ]
        assert snapshots[-1]["latitude"] == pytest.approx(31.0020)
        assert snapshots[-1]["progress_pct"] == 100.0
        assert all(s["data_source"] == "SIMULATED" for s in snapshots)


class TestTask0ReviewFixes:
    """Verifies that Phase 5 review fixes (Task 0) are in effect."""

    def test_hospital_deduplication(self):
        """Verify that nearby duplicate hospitals within 150m are merged."""
        from core.hospitals.data import parse_osm_element_to_hospital
        hospitals_path = PROJECT_ROOT / "data" / "hospitals_osm.json"
        with open(hospitals_path, "r", encoding="utf-8") as f:
            raw_elements = json.load(f).get("elements", [])

        parsed = [parse_osm_element_to_hospital(e) for e in raw_elements]
        parsed = [h for h in parsed if h is not None]
        assert len(parsed) == 116

        deduped, merged_count = deduplicate_hospitals(parsed, distance_threshold_km=0.150)
        # Raw elements had 116, deduped has 115, exactly 1 merged
        assert len(deduped) == 115
        assert merged_count == 1
        # Civil hospital in Phagwara merged: exactly one Phagwara civil hospital, lowercase 'civilhospital' is merged
        civil_phagwara = [h for h in deduped if "civil" in h.name.lower() and "phagwara" in h.name.lower()]
        assert len(civil_phagwara) == 1
        assert not any(h.name == "civilhospital" for h in deduped)

    def test_suitability_filter_downranks_unrelated_specialty(self):
        """Verify eye hospital is downranked when caller does not report eye trauma."""
        eye_hosp = Hospital(
            hospital_id="test-eye",
            name="Mitra Eye Hospital",
            latitude=31.215,
            longitude=75.777,
            specialties="ophthalmology",
            data_source="REAL",
        )
        penalty, rule = evaluate_facility_suitability(eye_hosp, "chest trauma and fractures")
        assert penalty == 0.30
        assert rule is not None
        assert "Down-ranked by suitability rule" in rule

    def test_assumed_needs_format(self):
        """Verify the explanatory assumed-needs string strictly says ASSUMED NEEDS."""
        q = describe_assumed_needs(injuries=["broken arm", "bleeding"])
        assert q.startswith("ASSUMED NEEDS (rule-based mapping from caller keywords):")
        assert "emergency / trauma department" in q


class TestDatabaseRoutingPersistence:
    """Tests for SQLite persistence of RouteResult and Ambulance entities."""

    def test_save_and_retrieve_ambulances(self, tmp_path):
        """Verify saving and retrieving ambulances from SQLite."""
        db_file = tmp_path / "test_amb.db"
        init_db(db_file)

        ambs = [
            {
                "ambulance_id": "amb-t1",
                "call_sign": "AMB-T1",
                "base_hospital": "Test Base",
                "latitude": 31.22,
                "longitude": 75.77,
                "status": "available",
                "equipment_level": "Advanced",
                "data_source": "SIMULATED",
            }
        ]
        count = save_ambulances(ambs, db_path=db_file)
        assert count == 1

        rows = get_ambulances(db_path=db_file)
        assert len(rows) == 1
        assert rows[0]["ambulance_id"] == "amb-t1"
        assert rows[0]["call_sign"] == "AMB-T1"
        assert rows[0]["base_hospital"] == "Test Base"
        assert rows[0]["data_source"] == "SIMULATED"

    def test_save_and_retrieve_route_result(self, tmp_path):
        """Verify saving and retrieving RouteResult from SQLite."""
        db_file = tmp_path / "test_route.db"
        init_db(db_file)

        route = RouteResult(
            route_id="route-test-01",
            ambulance_id="amb-t1",
            hospital_id="hosp-t1",
            accident_id="acc-t1",
            distance_km=5.4,
            estimated_time_minutes=8.2,
            route_waypoints=[[31.22, 75.77], [31.23, 75.78]],
            ambulance_eta_minutes=3.5,
            hospital_transit_minutes=4.7,
            straight_line_km=3.8,
            driving_vs_straight_ratio=1.42,
            attribution="© OpenStreetMap contributors",
            disclaimer="Estimated from OSM speed data, no live traffic",
            data_source="SIMULATED",
        )
        saved_id = save_route_result(route, db_path=db_file)
        assert saved_id == "route-test-01"

        rows = get_route_results(db_path=db_file)
        assert len(rows) == 1
        assert rows[0]["route_id"] == "route-test-01"
        assert rows[0]["distance_km"] == 5.4


class TestPhase6bMedicalValidityRule:
    """Phase 6b regression tests enforcing medical destination validity and corridor limits."""

    def test_regression_lpu_architecture_never_selected(self, road_graph):
        """Regression test: osm-way-832187216 (LPU Architecture Deisgn) must never be selected."""
        from core.hospitals.data import parse_osm_element_to_hospital, validate_hospital_for_dispatch
        hospitals_path = PROJECT_ROOT / "data" / "hospitals_osm.json"
        with open(hospitals_path, "r", encoding="utf-8") as f:
            raw_elements = json.load(f).get("elements", [])

        # Find the real problem record from the cache
        problem_elem = next((e for e in raw_elements if str(e.get("id")) == "832187216"), None)
        assert problem_elem is not None, "Real record osm-way-832187216 must exist in cache"

        parsed = parse_osm_element_to_hospital(problem_elem)
        assert parsed is not None
        assert parsed.name == "LPU Architecture Deisgn"

        # Verify validity rule directly rejects it
        is_valid, reason = validate_hospital_for_dispatch(parsed, problem_elem.get("tags", {}))
        assert not is_valid
        assert reason is not None
        assert "Non-medical" in reason
        assert parsed.excluded_from_dispatch_reason is not None

        # Verify dispatch on LPU Chaheru scenario never selects it
        res = plan_emergency_dispatch(
            incident_lat=31.2532,
            incident_lon=75.7035,
            accident_id="loc-phagwara-lpu-chaheru",
            graph=road_graph,
        )
        assert res.hospital_id != "osm-way-832187216"
        assert "Architecture" not in res.chosen_hospital["name"]
        for alt in res.alternative_hospitals:
            assert alt["hospital_id"] != "osm-way-832187216"

    def test_exclusion_reasons_and_categories(self):
        """Test transparent classification and reason tracking across facility types."""
        from core.hospitals.data import validate_hospital_for_dispatch

        # Non-medical building
        h_non_med = Hospital(
            hospital_id="t-1",
            name="University College of Fine Arts",
            latitude=31.2,
            longitude=75.7,
            data_source="REAL",
        )
        valid, reason = validate_hospital_for_dispatch(h_non_med)
        assert not valid
        assert "Non-medical" in reason

        # Allowed medical college exception
        h_med_coll = Hospital(
            hospital_id="t-2",
            name="Government Medical College and Hospital",
            latitude=31.2,
            longitude=75.7,
            data_source="REAL",
        )
        valid, reason = validate_hospital_for_dispatch(h_med_coll)
        assert valid
        assert reason is None

        # Outpatient clinic
        h_clinic = Hospital(
            hospital_id="t-3",
            name="City Dental Clinic and Lab",
            latitude=31.2,
            longitude=75.7,
            data_source="REAL",
        )
        valid, reason = validate_hospital_for_dispatch(h_clinic)
        assert not valid
        assert "clinic" in reason.lower() or "dental" in reason.lower()

        # Missing hospital tags and keywords
        h_vague = Hospital(
            hospital_id="t-4",
            name="Sanjivani Health Care",
            latitude=31.2,
            longitude=75.7,
            data_source="REAL",
        )
        valid, reason = validate_hospital_for_dispatch(h_vague)
        assert not valid
        assert "Missing hospital tag" in reason

        # Valid hospital
        h_hosp = Hospital(
            hospital_id="t-5",
            name="Civil Hospital Phagwara",
            latitude=31.23,
            longitude=75.76,
            data_source="REAL",
        )
        valid, reason = validate_hospital_for_dispatch(h_hosp)
        assert valid
        assert reason is None

    def test_graph_corridor_hospital_coverage(self):
        """Verify at least 3 valid hospitals are inside the corridor and none outside are chosen."""
        from core.routing.dispatch import get_candidate_hospitals
        corridor_hospitals = get_candidate_hospitals(corridor_only=True)
        assert len(corridor_hospitals) >= 3, f"Must have >= 3 valid hospitals, found {len(corridor_hospitals)}"
        # Verify all are valid
        for h in corridor_hospitals:
            assert h.excluded_from_dispatch_reason is None

    def test_route_results_replace_old_rows_without_duplicates(self, tmp_path):
        """Verify re-saving route result for same accident_id replaces old record."""
        db_file = tmp_path / "test_replace.db"
        init_db(db_file)

        route1 = RouteResult(
            route_id="route-old-hash-111",
            ambulance_id="amb-01",
            hospital_id="hosp-01",
            accident_id="loc-test-01",
            distance_km=4.0,
            estimated_time_minutes=6.0,
            data_source="SIMULATED",
        )
        save_route_result(route1, db_path=db_file)
        assert len(get_route_results(db_path=db_file)) == 1

        # Save updated route for same accident_id
        route2 = RouteResult(
            route_id="route-new-deterministic",
            ambulance_id="amb-02",
            hospital_id="hosp-02",
            accident_id="loc-test-01",
            distance_km=3.5,
            estimated_time_minutes=5.0,
            ambulance_eta_minutes=2.1,
            hospital_transit_minutes=2.9,
            data_source="SIMULATED",
        )
        save_route_result(route2, db_path=db_file)
        rows = get_route_results(db_path=db_file)
        assert len(rows) == 1
        assert rows[0]["route_id"] == "route-new-deterministic"
        assert rows[0]["ambulance_id"] == "amb-02"
        assert rows[0]["hospital_id"] == "hosp-02"
        assert rows[0]["ambulance_eta_minutes"] == 2.1
        assert rows[0]["hospital_transit_minutes"] == 2.9
        assert rows[0]["data_source"] == "SIMULATED"


class TestPhase6dDispatchSummaryAndAlternatives:
    """Tests for Phase 6d dispatch summary JSON export and alternative hospital transparency."""

    def test_all_corridor_hospitals_considered_with_reasons(self, road_graph):
        """Verify all valid corridor hospitals are evaluated and carry plain decision reasons."""
        result = plan_emergency_dispatch(
            incident_lat=31.2185,
            incident_lon=75.7852,
            accident_id="loc-phagwara-banga-road",
            caller_text="Two cars collided head-on, multiple passengers injured, severe bleeding.",
            graph=road_graph,
        )

        assert len(result.alternative_hospitals) == 8
        chosen_count = 0
        for h in result.alternative_hospitals:
            assert "osm_id" in h
            assert "name" in h
            assert "driving_km" in h
            assert "transit_seconds" in h
            assert "penalty" in h
            assert "adjusted_cost_seconds" in h
            assert h["considered"] is True
            assert "note" in h
            assert len(h["note"]) > 0

            if h["chosen"]:
                chosen_count += 1
                assert "Chosen:" in h["note"]
            else:
                assert "Not chosen:" in h["note"]

        assert chosen_count == 1
        assert len(result.alternative_ambulances) == 6
        assert sum(1 for a in result.alternative_ambulances if a["chosen"]) == 1

    def test_dispatch_summary_json_matches_database(self, tmp_path):
        """Verify exported dispatch_summary.json exactly matches database records."""
        from core.routing.dispatch import export_dispatch_summary
        from db.database import DEFAULT_DB_PATH, get_route_results

        out_json = tmp_path / "dispatch_summary.json"
        summary = export_dispatch_summary(output_path=out_json)

        assert out_json.exists()
        assert "generated_at" in summary
        assert "disclaimers" in summary
        assert len(summary["disclaimers"]) == 3
        assert "scenarios" in summary

        db_rows = get_route_results()
        assert len(summary["scenarios"]) == len(db_rows)

        rows_by_id = {r["accident_id"]: r for r in db_rows}
        for sc in summary["scenarios"]:
            loc_id = sc["location_id"]
            assert loc_id in rows_by_id
            row = rows_by_id[loc_id]

            assert sc["ambulance"]["id"] == row["ambulance_id"]
            assert sc["hospital"]["osm_id"] == row["hospital_id"]
            assert sc["total_km"] == pytest.approx(row["distance_km"], rel=1e-3)
            assert sc["total_seconds"] == pytest.approx(row["estimated_time_minutes"] * 60.0, rel=1e-3)
            assert sc["ambulance"]["eta_seconds"] == pytest.approx(row["ambulance_eta_minutes"] * 60.0, rel=1e-3)
            assert sc["hospital"]["transit_seconds"] == pytest.approx(row["hospital_transit_minutes"] * 60.0, rel=1e-3)
            assert sc["leg_to_incident"] == json.loads(row["leg1_waypoints"])
            assert sc["leg_to_hospital"] == json.loads(row["leg2_waypoints"])

            # Tab 6-specific honesty and route-audit fields come from stored data.
            assert summary["simulation_notice"].startswith("SIMULATED DISPATCH")
            assert summary["gps_status"].startswith("NOT AVAILABLE")
            assert "no live traffic" in summary["traffic_note"].lower()
            assert sc["simulation_notice"] == summary["simulation_notice"]
            assert sc["traffic_note"] == summary["traffic_note"]
            assert sc["location_context"]["caller_used_for"].startswith("Nothing in Tab 6")
            assert sc["location_context"]["hospital_search_used_for"].startswith("Tab 5")
            assert sc["route_verification"] == route_geometry_audit(
                sc["leg_to_incident"], sc["leg_to_hospital"]
            )
            assert sc["route_verification"]["legs_join_at_incident"] is True
            assert sc["ambulance"]["selection_reason"]
            assert "No live GPS" in sc["ambulance"]["selection_criteria"]
            assert sc["hospital"]["emergency_capability"]
            assert "trauma-suited" not in json.dumps(sc).lower()
            assert sc["trajectory"][0]["leg"] == "leg1_ambulance_to_incident"
            assert sc["trajectory"][-1]["leg"] == "leg2_incident_to_hospital"
            assert sc["trajectory"][-1]["t_seconds"] == pytest.approx(sc["total_seconds"])

            assert len(sc["hospital_alternatives"]) == 8
            assert len(sc["ambulance_alternatives"]) == 6

            # Every non-chosen hospital must have an explanatory note
            for ha in sc["hospital_alternatives"]:
                assert ha["note"] is not None
                assert len(ha["note"]) > 0
