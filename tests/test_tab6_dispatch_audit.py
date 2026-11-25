"""Tab 6 dispatch audit regressions.

All scenario data comes from the real dispatch summary; mock graphs only supply
road geometry so selection rules can be isolated without touching the database.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import networkx as nx
import pytest
from streamlit.testing.v1 import AppTest

from app.dispatch_ui import (
    DEFAULT_DISPATCH_SUMMARY_PATH,
    DISPATCH_SUMMARY_ENV,
    LIVE_TRAFFIC_NOTICE,
    SIMULATION_NOTICE,
    build_dispatch_map,
    emergency_status_from_capability,
    hospital_alternative_rows,
    hospital_eligibility,
    iter_map_markers,
    location_audit,
    marker_tooltip,
    navigation_ready,
    number_text,
    replay_follows_route,
    replay_labels,
    replay_phase,
    routing_check_lines,
    scenarios,
    selection_consistency,
)
from core.routing.dispatch import (
    HOSPITAL_RANKING_PRIORITY,
    get_candidate_hospitals,
    plan_emergency_dispatch,
)
from core.schemas import Hospital

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_FILE = REPO_ROOT / "app" / "main.py"


@pytest.fixture(scope="module")
def summary() -> dict:
    if not DEFAULT_DISPATCH_SUMMARY_PATH.is_file():
        pytest.skip("Real dispatch summary is unavailable")
    payload = json.loads(DEFAULT_DISPATCH_SUMMARY_PATH.read_text(encoding="utf-8"))
    if not scenarios(payload):
        pytest.skip("Real dispatch summary has no scenarios")
    return payload


@pytest.fixture(scope="module")
def scenario(summary: dict) -> dict:
    return scenarios(summary)[0]


def _hospital(identifier: str, name: str, lat: float, lon: float, **kwargs) -> Hospital:
    return Hospital(
        hospital_id=identifier,
        name=name,
        latitude=lat,
        longitude=lon,
        data_source="REAL",
        **kwargs,
    )


def _graph() -> nx.MultiDiGraph:
    """Two-minute ambulance approach and two hospital legs from one incident."""
    graph = nx.MultiDiGraph()
    graph.add_node(1, y=31.01, x=75.0)      # ambulance base (nearer to hospital A)
    graph.add_node(2, y=31.005, x=75.0)     # second ambulance base
    graph.add_node(100, y=31.0, x=75.0)     # incident
    graph.add_node(200, y=31.02, x=75.0)    # hospital A
    graph.add_node(201, y=31.04, x=75.0)    # hospital B
    graph.add_edge(1, 100, length=1000.0, speed_kph=60.0, travel_time=60.0)
    graph.add_edge(2, 100, length=500.0, speed_kph=10.0, travel_time=180.0)
    graph.add_edge(100, 200, length=1000.0, speed_kph=60.0, travel_time=60.0)
    graph.add_edge(100, 201, length=4000.0, speed_kph=60.0, travel_time=240.0)
    return graph


def _ambulances() -> list[dict]:
    return [
        {
            "ambulance_id": "amb-01",
            "call_sign": "AMB-1",
            "latitude": 31.01,
            "longitude": 75.0,
            "status": "available",
            "data_source": "SIMULATED",
        }
    ]


def _plan(candidates: list[Hospital]) -> object:
    return plan_emergency_dispatch(
        incident_lat=31.0,
        incident_lon=75.0,
        accident_id="tab6-audit",
        graph=_graph(),
        ambulances=_ambulances(),
        candidate_hospitals=candidates,
    )


# ---------------------------------------------------------------------------
# 1. Eligibility — no unrelated facility can be an emergency destination
# ---------------------------------------------------------------------------
def test_unrelated_facility_cannot_be_selected_even_if_supplied_as_candidate() -> None:
    mis_tagged = _hospital("osm-way-832187216", "LPU Architecture Deisgn", 31.02, 75.0)
    clinic = _hospital("osm-node-1", "Dr Aman Clinic", 31.04, 75.0, emergency="yes")
    valid = _hospital("osm-node-2", "Civil Hospital Phagwara", 31.04, 75.0)

    result = _plan([mis_tagged, clinic, valid])

    assert result.hospital_id == "osm-node-2"
    assert result.chosen_hospital["facility_category"] == "hospital"
    assert len(result.alternative_hospitals) == 1
    assert result.alternative_hospitals[0]["hospital_id"] == "osm-node-2"
    assert "Architecture" not in result.chosen_hospital["name"]
    assert "Clinic" not in result.chosen_hospital["name"]


def test_no_eligible_candidate_raises_clear_error() -> None:
    from pytest import raises

    with raises(ValueError, match="No dispatch-eligible hospital"):
        _plan([_hospital("osm-way-832187216", "LPU Architecture Deisgn", 31.02, 75.0)])


def test_cached_corridor_candidates_all_pass_tab5_classifier() -> None:
    hospitals = get_candidate_hospitals()
    assert len(hospitals) >= 3
    for hospital in hospitals:
        eligibility = hospital_eligibility({"hospital": {"name": hospital.name, "osm_id": hospital.hospital_id}})
        assert eligibility["dispatch_eligible"] is True
        assert eligibility["category"] in {"hospital", "trauma centre"}


# ---------------------------------------------------------------------------
# 2-3. Ranking priority consistent with Tab 5
# ---------------------------------------------------------------------------
def test_source_reported_emergency_beats_nearer_unspecified_hospital() -> None:
    near = _hospital("osm-node-a", "Civil Hospital Phagwara", 31.02, 75.0)
    far_emergency = _hospital(
        "osm-node-b", "Modern Hospital, Jalandhar", 31.04, 75.0, emergency="yes"
    )
    result = _plan([near, far_emergency])

    assert result.hospital_id == "osm-node-b"
    assert result.chosen_hospital["emergency_status"] == "SOURCE_CONFIRMED"
    assert "not independently verified" in result.chosen_hospital["emergency_suitability"]
    assert "24/7 not established" in result.chosen_hospital["emergency_suitability"]
    assert result.chosen_hospital["ranking_priority"] == HOSPITAL_RANKING_PRIORITY


def test_trauma_centre_beats_plain_hospital_at_same_emergency_status() -> None:
    hospital = _hospital(
        "osm-node-a", "Civil Hospital Phagwara", 31.02, 75.0, emergency="yes"
    )
    trauma = _hospital("osm-node-b", "Civil Trauma Centre", 31.04, 75.0, emergency="yes")
    result = _plan([hospital, trauma])
    assert result.hospital_id == "osm-node-b"
    assert result.chosen_hospital["facility_category"] == "trauma centre"


def test_three_tab_locations_stay_separate_with_tab4_location_field(scenario: dict) -> None:
    """Tab 4 stores its caller text in `location`; Tab 6 must not substitute it."""
    selected = {
        "name": "NH-44 GT Road near Jagatjit Sugar Mill",
        "latitude": 31.2142,
        "longitude": 75.7785,
        "data_source": "SIMULATED",
    }
    report = {"location": "MG Road near Trinity Metro station"}
    audit = location_audit(report, selected, scenario)

    assert [row["Source"] for row in audit["rows"]][0].startswith("SIMULATED sample call (Tab 4)")
    assert audit["rows"][0]["Reported location"] == report["location"]
    assert audit["rows"][1]["Used for in Tab 6"].startswith("Nothing")
    assert audit["rows"][2]["Used for in Tab 6"].startswith("Both route legs")
    assert "no substitution" in " ".join(audit["discrepancies"]).lower()
    assert str(scenario["incident"]["lat"]) in audit["dispatch_coordinates_used"]
    assert str(scenario["incident"]["lon"]) in audit["dispatch_coordinates_used"]
    assert audit["gps_status"].startswith("NOT AVAILABLE")


# ---------------------------------------------------------------------------
# 4-5. Recorded routes, totals and selection consistency
# ---------------------------------------------------------------------------
def test_all_real_scenarios_match_recorded_routes_and_rankings(summary: dict) -> None:
    assert len(scenarios(summary)) == 4
    for item in scenarios(summary):
        consistency = selection_consistency(item)
        assert consistency["ambulance_lowest_eta"] is True, item["location_id"]
        assert consistency["hospital_lowest_adjusted_cost"] is True, item["location_id"]
        assert consistency["hospital_priority_ok"] is True, item["location_id"]

        lines = " ".join(routing_check_lines(item))
        assert "Distance totals" in lines and "matches" in lines
        assert "Time totals" in lines and "matches" in lines
        assert "no live traffic" in lines.lower()
        assert "no device gps verification" in lines.lower()

        verification = item["route_verification"]
        assert verification["legs_join_at_incident"] is True
        assert verification["leg1_polyline_km"] > 0.0
        assert verification["leg2_polyline_km"] > 0.0


def test_selection_consistency_flags_stale_summary_data(scenario: dict) -> None:
    changed = copy.deepcopy(scenario)
    changed["ambulance"]["eta_seconds"] = 1.0
    other = next(
        entry for entry in changed["hospital_alternatives"]
        if entry["osm_id"] != changed["hospital"]["osm_id"]
    )
    other["adjusted_cost_seconds"] = 1.0
    result = selection_consistency(changed)
    assert result["ambulance_lowest_eta"] is False
    assert result["hospital_lowest_adjusted_cost"] is False
    assert result["hospital_priority_ok"] is False
    assert any("DIFFERS" in line for line in result["details"])


def test_replay_step_is_checked_against_its_own_leg_only(scenario: dict) -> None:
    changed = copy.deepcopy(scenario)
    label = replay_labels(changed)[0]
    assert replay_phase(changed, label) == "ambulance base → incident"
    assert replay_follows_route(changed, label) is True

    wrong_point = changed["leg_to_hospital"][len(changed["leg_to_hospital"]) // 2]
    for step in changed["trajectory"]:
        if f"t = {number_text(step['t_seconds'])} s" == label:
            step["lat"], step["lon"] = wrong_point
            break
    assert replay_follows_route(changed, label) is False


# ---------------------------------------------------------------------------
# 6-8. Labelling, UI readability, missing data and navigation
# ---------------------------------------------------------------------------
def test_hospital_alternatives_are_chosen_marked_and_tab5_labelled(scenario: dict) -> None:
    rows = hospital_alternative_rows(scenario)
    assert rows
    chosen = [row for row in rows if row["Chosen"] == "yes"]
    assert len(chosen) == 1
    assert chosen[0]["OSM ID"] == scenario["hospital"]["osm_id"]
    assert chosen[0]["Tab 5 category"] == "hospital"
    assert "unverified" in chosen[0]["Emergency status"]
    assert all(row["Source"] == "REAL" for row in rows)
    assert all(row["Tab 5 category"] in {"hospital", "trauma centre"} for row in rows)


def test_emergency_status_never_claims_verified_capability() -> None:
    assert "not independently verified" in emergency_status_from_capability(
        "OSM tag emergency=yes (tagged emergency department; not a clinical verification)"
    )
    assert "emergency=no" in emergency_status_from_capability("OSM tag emergency=no")
    assert "unverified" in emergency_status_from_capability(None)


def test_navigation_links_are_not_built_from_placeholder_coordinates() -> None:
    assert navigation_ready((31.2185, 75.7852)) is True
    assert navigation_ready((0.0, 0.0)) is False
    assert navigation_ready(None) is False


def test_ineligible_summary_destination_is_flagged_in_ui_and_on_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: dict
) -> None:
    changed = copy.deepcopy(scenario)
    changed["hospital"]["name"] = "LPU Architecture Deisgn"
    changed["hospital"]["osm_id"] = "osm-way-832187216"

    eligibility = hospital_eligibility(changed)
    assert eligibility["dispatch_eligible"] is False
    assert "NOT ELIGIBLE" in eligibility["status"]
    tooltips = [marker_tooltip(marker) for marker in iter_map_markers(build_dispatch_map(changed))]
    assert any("INELIGIBLE FACILITY" in tip for tip in tooltips)
    assert not any(tip == "REAL OSM: chosen hospital" for tip in tooltips)

    payload = {"disclaimers": ["© OpenStreetMap contributors"], "scenarios": [changed]}
    injected = tmp_path / "dispatch_summary_ineligible.json"
    injected.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(DISPATCH_SUMMARY_ENV, str(injected))

    at = AppTest.from_file(str(APP_FILE), default_timeout=300).run()
    assert not at.exception, [item.value for item in at.exception]
    dispatch = next(tab for tab in at.tabs if tab.label.startswith("6."))
    text = " ".join(
        str(getattr(node, "value", ""))
        for kind in ("markdown", "text", "caption", "warning", "error")
        for node in dispatch.get(kind)
    )
    assert "NOT ELIGIBLE — do not use as emergency destination" in text
    assert any("NOT a dispatch-eligible" in item.value for item in dispatch.error)


def test_tab6_makes_no_live_dispatch_or_gps_claims(scenario: dict) -> None:
    """Only source labels are asserted; the scenario argument keeps this data-backed."""
    assert scenario["ambulance"]["data_source"] == "SIMULATED"
    assert scenario["incident"]["data_source"] == "SIMULATED"
    assert scenario["hospital"]["data_source"] == "REAL"
    text = " ".join(
        [
            SIMULATION_NOTICE,
            "No live GPS, vehicle availability, operational status or real ambulance contact is used.",
            "GPS verification: NOT AVAILABLE — no device GPS verification was performed",
            LIVE_TRAFFIC_NOTICE,
        ]
    )
    assert "no real ambulance has been contacted" in text.lower()
    assert "no live gps" in text.lower()
    assert "no device gps verification" in text.lower()
    assert "no live traffic" in text.lower()
    assert "gps tracking" not in text.lower()
