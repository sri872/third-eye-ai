"""Phase 8b dispatch tests.

Data tests read the REAL data/outputs/dispatch_summary.json written by the
Phase 6d agent. They skip with a clear message while that file is absent; no
scenario data is hand-written here. The "not available yet" behaviour is tested
separately in tests/test_app_ui.py against an injected nonexistent path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import folium
import pytest
from streamlit.testing.v1 import AppTest

from app.dispatch_ui import (
    DEFAULT_DISPATCH_SUMMARY_PATH,
    DISPATCH_SUMMARY_ENV,
    FALLBACK_TILE_NAME,
    GOOGLE_MAPS_LABEL,
    SIMULATION_NOTICE,
    alternative_rows,
    ambulance_selection_reason,
    build_dispatch_map,
    build_google_maps_url,
    coord_text,
    coordinate_pair,
    display,
    emergency_plain_text,
    emergency_status,
    fallback_geometry_check,
    hospital_alternative_rows,
    hospital_capability_text,
    hospital_eligibility,
    iter_map_markers,
    location_audit,
    marker_tooltip,
    navigation_ready,
    replay_follows_route,
    replay_labels,
    replay_phase,
    replay_position,
    route_audit_lines,
    route_verification,
    routing_check_lines,
    scenarios,
    selection_consistency,
)
from core.hospitals.emergency_numbers import get_emergency_numbers
from core.routing.dispatch import AMBULANCE_SELECTION_CRITERIA, route_geometry_audit

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_FILE = REPO_ROOT / "app" / "main.py"
APP_DIR = REPO_ROOT / "app"

FORBIDDEN_WIDGETS = (
    "st.button",
    "st.text_input",
    "st.text_area",
    "st.chat_input",
    "st.form_submit_button",
    "st.download_button",
    "st.link_button",
)


@pytest.fixture(scope="module")
def summary() -> dict:
    """Load the real summary; skip rather than fabricate scenario data."""
    if not DEFAULT_DISPATCH_SUMMARY_PATH.is_file():
        pytest.skip(
            "Real dispatch summary not available yet at "
            f"{DEFAULT_DISPATCH_SUMMARY_PATH}; skipping data-backed tests."
        )
    try:
        payload = json.loads(DEFAULT_DISPATCH_SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        pytest.skip(f"Real dispatch summary is not readable yet: {exc}")
    if not isinstance(payload, dict) or not scenarios(payload):
        pytest.skip("Real dispatch summary has no scenarios yet; skipping data-backed tests.")
    return payload


@pytest.fixture
def scenario(summary: dict) -> dict:
    return scenarios(summary)[0]


def _run_app() -> AppTest:
    return AppTest.from_file(str(APP_FILE), default_timeout=300).run()


def _rendered(tab: object) -> str:
    return " ".join(
        str(getattr(node, "value", ""))
        for kind in ("info", "markdown", "text", "caption", "warning", "error")
        for node in tab.get(kind)  # type: ignore[attr-defined]
    )


def test_google_maps_url_builder_uses_official_scheme(scenario: dict) -> None:
    incident = coordinate_pair(scenario.get("incident") or {}, "lat", "lon")
    base = coordinate_pair(scenario.get("ambulance") or {}, "base_lat", "base_lon")
    hospital = coordinate_pair(scenario.get("hospital") or {}, "lat", "lon")
    assert incident is not None and base is not None and hospital is not None

    origin = coord_text(base)
    destination = coord_text(hospital)
    waypoint = coord_text(incident)
    assert origin and destination and waypoint

    via_url = build_google_maps_url(origin, destination, [waypoint])
    assert via_url == (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={origin}&destination={destination}"
        f"&waypoints={waypoint}&travelmode=driving"
    )
    parsed = parse_qs(urlparse(via_url).query)
    assert parsed["api"] == ["1"]
    assert parsed["origin"] == [origin]
    assert parsed["destination"] == [destination]
    assert parsed["waypoints"] == [waypoint]
    assert parsed["travelmode"] == ["driving"]

    direct_url = build_google_maps_url(coord_text(incident), destination)
    assert "waypoints" not in parse_qs(urlparse(direct_url).query)


def test_map_builder_returns_folium_map_with_expected_markers(scenario: dict) -> None:
    fmap = build_dispatch_map(scenario)
    assert isinstance(fmap, folium.Map)

    markers = list(iter_map_markers(fmap))
    tooltips = [marker_tooltip(marker) for marker in markers]
    assert len(markers) >= 3, tooltips
    assert any("SIMULATED" in tip and "incident" in tip for tip in tooltips), tooltips
    assert any("SIMULATED" in tip and "ambulance base" in tip for tip in tooltips), tooltips
    assert any("REAL OSM" in tip and "hospital" in tip for tip in tooltips), tooltips

    polylines = [
        child for child in fmap._children.values() if isinstance(child, folium.PolyLine)
    ]
    assert len(polylines) >= 2
    colors = {str(line.options.get("color")) for line in polylines}
    assert len(colors) >= 2, colors

    rendered = fmap.get_root().render()
    assert "openstreetmap.org" in rendered
    assert "OpenStreetMap" in rendered
    assert any(type(child).__name__ == "FitBounds" for child in fmap._children.values())


def test_replay_step_selection_changes_simulated_marker_position(scenario: dict) -> None:
    labels = replay_labels(scenario)
    if len(labels) < 2:
        pytest.skip("Real trajectory has fewer than two replay steps.")

    first, last = labels[0], labels[-1]

    def replay_location(fmap: folium.Map) -> tuple[float, float] | None:
        for marker in iter_map_markers(fmap):
            if "replay" in marker_tooltip(marker).lower():
                return tuple(marker.location)
        return None

    first_map = build_dispatch_map(scenario, first)
    last_map = build_dispatch_map(scenario, last)
    first_position = replay_location(first_map)
    last_position = replay_location(last_map)

    assert first_position is not None and last_position is not None
    assert first_position == replay_position(scenario, first)
    assert last_position == replay_position(scenario, last)
    assert first_position != last_position


def test_app_scan_finds_no_forbidden_widgets_or_phone_links() -> None:
    offenses: list[str] = []
    for path in sorted(APP_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(REPO_ROOT))
        for widget in FORBIDDEN_WIDGETS:
            name = widget.split(".", 1)[1]
            if re.search(rf"\bst(?:\.sidebar)?\.{re.escape(name)}\s*\(", source):
                offenses.append(f"{relative}: {widget}")
        if re.search(r"(?i)\btel\s*:", source):
            offenses.append(f"{relative}: phone link")
    assert not offenses, "Forbidden app constructs: " + ", ".join(offenses)


def test_dispatch_tab_renders_real_summary_fields(summary: dict, scenario: dict) -> None:
    at = _run_app()
    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]
    dispatch = next(tab for tab in at.tabs if tab.label.startswith("6."))
    text = _rendered(dispatch)

    assert "not determined (demo placement)" in text
    assert "ranking heuristic, not a clinical rating" in text
    assert "not determined" in text  # bed availability
    assert GOOGLE_MAPS_LABEL in text
    assert "This app does not place calls; dial from your phone." in text
    for disclaimer in summary.get("disclaimers", []):
        assert display(disclaimer) in text
    for phrase in ("OpenStreetMap contributors", "no live traffic", "corridor only"):
        assert phrase.lower() in text.lower()

    for value in (
        display(scenario.get("location_name")),
        display((scenario.get("hospital") or {}).get("name")),
        display((scenario.get("ambulance") or {}).get("callsign")),
        display(scenario.get("ranking_formula")),
    ):
        assert value in text

    labels = [node.value for node in dispatch.get("markdown")]
    maps_links = [item for item in labels if GOOGLE_MAPS_LABEL in item]
    assert len(maps_links) >= 2


def test_missing_and_null_fields_render_not_determined(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, summary: dict
) -> None:
    """Null out fields on a COPY of the real summary; never invent scenario data."""
    payload = json.loads(json.dumps(summary))
    first = payload["scenarios"][0]
    first["ranking_formula"] = None
    first["total_seconds"] = None
    first["total_km"] = None
    for key in ("incident", "ambulance", "hospital"):
        if isinstance(first.get(key), dict):
            first[key] = dict(first[key])
    first["incident"]["lat"] = None
    first["ambulance"]["eta_seconds"] = None
    first["hospital"]["phone"] = None
    first["hospital"]["emergency_tag"] = None
    first["hospital"]["specialties"] = None

    injected = tmp_path / "dispatch_summary_with_nulls.json"
    injected.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(DISPATCH_SUMMARY_ENV, str(injected))

    at = _run_app()
    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]
    dispatch = next(tab for tab in at.tabs if tab.label.startswith("6."))
    text = _rendered(dispatch)

    assert "not determined (demo placement)" in text
    assert "not listed" in text
    assert "not specified in OpenStreetMap" in text
    assert "not specified" in text
    assert text.count("not determined") >= 5


def test_emergency_numbers_are_plain_text_and_status_comes_from_data() -> None:
    contacts = get_emergency_numbers(urgency=None, hazards=[])
    assert contacts
    for contact in contacts:
        text = emergency_plain_text(contact)
        assert contact["number"] in text
        assert contact["service"] in text
        expected = emergency_status(contact)
        assert expected in text
        if expected == "verified":
            assert str(contact["verification_status"]).upper() == "VERIFIED"
            assert contact.get("official_source") or contact.get("source_url")

    verified = dict(contacts[0])
    assert emergency_status(verified) in {"verified", "UNVERIFIED"}
    unverified = dict(contacts[0])
    unverified["verification_status"] = "UNVERIFIED"
    assert emergency_status(unverified) == "UNVERIFIED"
    no_source = dict(contacts[0])
    no_source["official_source"] = None
    no_source["source_url"] = None
    assert emergency_status(no_source) == "UNVERIFIED"

    at = _run_app()
    assert not at.exception, [item.value for item in at.exception]
    hospitals = next(tab for tab in at.tabs if tab.label.startswith("5."))
    text = _rendered(hospitals)
    assert "This app does not place calls; dial from your phone." in text
    assert "tel:" not in text.lower()


def test_location_audit_keeps_three_sources_separate(scenario: dict) -> None:
    locations = json.loads((REPO_ROOT / "config" / "demo_incident_locations.json").read_text())
    sidebar = next(
        item for item in locations
        if item["location_id"] == "loc-phagwara-nh44-sugarmill"
    )
    caller_location = "MG Road near Trinity Metro station"
    audit = location_audit({"caller_location": caller_location}, sidebar, scenario)

    sources = [row["Source"] for row in audit["rows"]]
    assert len(sources) == 3
    assert "Tab 4" in sources[0] and "Tab 5" in sources[1] and "Tab 6" in sources[2]
    assert audit["rows"][0]["Used for in Tab 6"].startswith("Nothing")
    assert audit["rows"][1]["Used for in Tab 6"].startswith("Nothing")
    assert audit["rows"][2]["Used for in Tab 6"] == (
        "Both route legs, distances, travel times and totals"
    )

    details = " ".join(audit["discrepancies"])
    assert caller_location in details
    assert str(scenario["location_name"]) in details
    assert str(sidebar["name"]) in details
    incident = scenario["incident"]
    assert str(incident["lat"]) in audit["dispatch_coordinates_used"]
    assert str(incident["lon"]) in audit["dispatch_coordinates_used"]
    assert audit["gps_status"].startswith("NOT AVAILABLE")


def test_ambulance_selection_uses_recorded_data_only(scenario: dict) -> None:
    ambulance = scenario["ambulance"]
    reason = ambulance_selection_reason(scenario)
    assert "ETA" in reason and "s" in reason
    assert "lowest" in reason.lower()
    assert "No live GPS" in ambulance["selection_criteria"]
    assert "no live gps" in AMBULANCE_SELECTION_CRITERIA.lower()

    alternatives = scenario["ambulance_alternatives"]
    chosen = [row for row in alternatives if row.get("chosen")]
    assert len(chosen) == 1
    etas = [float(row["eta_seconds"]) for row in alternatives]
    assert chosen[0]["eta_seconds"] == min(etas)
    assert all(row.get("selection_reason") for row in alternatives)
    assert chosen[0]["selection_reason"] == reason
    assert "status" not in ambulance
    assert display(ambulance.get("equipment_level")) != "not determined"


def test_hospital_capability_comes_from_osm_tags_only(scenario: dict) -> None:
    hospital = scenario["hospital"]
    capability = hospital_capability_text(hospital)
    assert "capability unverified" in capability.lower() or "emergency=yes" in capability.lower()
    assert "trauma" not in capability.lower()
    assert hospital["data_source"] == "REAL"
    assert str(hospital["osm_id"]).startswith("osm-")

    for row in scenario["hospital_alternatives"]:
        assert "trauma-suited" not in str(row.get("note", "")).lower()
        assert row["emergency_capability"]
    rows = alternative_rows(scenario["hospital_alternatives"], "REAL")
    assert rows[0]["Emergency capability"]

    summary_text = DEFAULT_DISPATCH_SUMMARY_PATH.read_text(encoding="utf-8")
    assert "trauma-suited" not in summary_text.lower()


def test_eligibility_and_selection_consistency_use_recorded_fields(scenario: dict) -> None:
    eligibility = hospital_eligibility(scenario)
    assert eligibility["dispatch_eligible"] is True
    assert "hospital" in eligibility["category"].lower() or "trauma" in eligibility["category"].lower()

    selection = selection_consistency(scenario)
    assert selection["ambulance_lowest_eta"] is True
    assert selection["hospital_lowest_adjusted_cost"] is True
    assert all("matches" in line for line in selection["details"][:2])

    rows = hospital_alternative_rows(scenario)
    assert sum(row["Chosen"] == "yes" for row in rows) == 1
    assert all(row["Tab 5 category"] and row["Emergency status"] for row in rows)
    assert "unverified" in rows[0]["Emergency status"].lower()

    assert navigation_ready((31.2, 75.7)) is True
    assert navigation_ready(None) is False
    assert navigation_ready((0.0, 0.0)) is False


def test_route_geometry_and_totals_use_both_recorded_legs(scenario: dict) -> None:
    verification = route_verification(scenario)
    assert verification == route_geometry_audit(
        scenario["leg_to_incident"], scenario["leg_to_hospital"]
    )
    assert verification["legs_join_at_incident"] is True
    assert verification["repeated_segments_leg1"] == 0
    assert verification["repeated_segments_leg2"] == 0
    assert verification["shared_segments_between_legs"] == 0
    assert verification["leg1_polyline_km"] > 0.0
    assert verification["leg2_polyline_km"] > 0.0

    lines = routing_check_lines(scenario)
    joined = " ".join(lines).lower()
    assert any("Distance totals" in line and "matches" in line for line in lines)
    assert any("Time totals" in line and "matches" in line for line in lines)
    assert any("Replay clock" in line and "matches" in line for line in lines)
    assert "leg1_ambulance_to_incident" in joined
    assert "leg2_incident_to_hospital" in joined
    assert "no live traffic" in joined
    assert "no device gps verification" in joined

    audit_text = " ".join(route_audit_lines(scenario))
    assert "km +" in audit_text and "s" in audit_text


def test_replay_covers_both_legs_and_stays_on_stored_route(scenario: dict) -> None:
    labels = replay_labels(scenario)
    assert len(labels) >= 3
    assert replay_phase(scenario, labels[0]) == "ambulance base → incident"
    assert replay_phase(scenario, labels[-1]) == "incident → hospital"

    last_point = scenario["leg_to_hospital"][-1]
    last_position = replay_position(scenario, labels[-1])
    assert last_position is not None
    assert last_position[0] == pytest.approx(last_point[0])
    assert last_position[1] == pytest.approx(last_point[1])

    for label in (labels[0], labels[len(labels) // 2], labels[-1]):
        assert replay_follows_route(scenario, label) is True

    final_map = build_dispatch_map(scenario, labels[-1])
    marker = next(
        marker for marker in iter_map_markers(final_map)
        if "replay" in marker_tooltip(marker).lower()
    )
    assert "incident → hospital" in marker_tooltip(marker)
    assert tuple(marker.location) == last_position


def test_static_fallback_uses_same_two_leg_geometry(scenario: dict) -> None:
    check = fallback_geometry_check(scenario)
    assert check["png_available"] is True
    assert check["geojson_available"] is True
    assert check["geojson_matches_legs"] is True

    payload = json.loads((REPO_ROOT / scenario["geojson"]).read_text(encoding="utf-8"))
    lines = [
        feature["geometry"]["coordinates"]
        for feature in payload["features"]
        if feature["geometry"]["type"] == "LineString"
    ]
    expected_first = [[lon, lat] for lat, lon in scenario["leg_to_incident"]]
    expected_second = [[lon, lat] for lat, lon in scenario["leg_to_hospital"]]
    assert lines[0] == expected_first
    assert lines[1] == expected_second


def test_map_has_switchable_fallback_tiles_and_osm_attribution(scenario: dict) -> None:
    rendered = build_dispatch_map(scenario).get_root().render()
    assert "openstreetmap.org" in rendered
    assert "OpenStreetMap" in rendered
    assert FALLBACK_TILE_NAME in rendered
    assert "tile.openstreetmap.fr/hot" in rendered
    assert "L.control.layers" in rendered

    polylines = [
        child for child in build_dispatch_map(scenario)._children.values()
        if isinstance(child, folium.PolyLine)
    ]
    assert len(polylines) >= 2
    assert len({str(line.options.get("color")) for line in polylines}) >= 2


def test_tab6_layout_and_honesty_statements(scenario: dict) -> None:
    at = _run_app()
    assert not at.exception, [item.value for item in at.exception]
    dispatch = next(tab for tab in at.tabs if tab.label.startswith("6."))
    text = _rendered(dispatch)
    headings = [node.value for node in dispatch.get("markdown")]

    for heading in (
        "### 1. Incident / dispatch location",
        "### 2. Ambulance and alternatives (SIMULATED)",
        "### 3. Hospital and alternatives (REAL OSM)",
        "### 4. Map, ETA and emergency contacts",
    ):
        assert heading in headings

    def table_as_text(node: object) -> str:
        value = getattr(node, "value", "")
        return value.to_string() if hasattr(value, "to_string") else str(value)

    table_text = " ".join(table_as_text(node) for node in dispatch.get("table"))
    assert "SIMULATED sample call (Tab 4)" in table_text
    assert "SIMULATED hospital-search selection (sidebar / Tab 5)" in table_text
    assert "SIMULATED dispatch demo location (Tab 6 summary)" in table_text

    expander_labels = [item.label for item in dispatch.get("expander")]
    assert "Location source details and provenance" in expander_labels
    assert "Hospital source details and limitations" in expander_labels
    assert any(
        label.startswith("Technical diagnostics") and "route legs" in label
        for label in expander_labels
    )

    assert SIMULATION_NOTICE in text
    assert "No live GPS" in text
    assert "no real ambulance has been contacted" in text.lower()
    assert "Selection reason" in text
    assert "Selection criteria" in text
    assert "Tab 5 eligibility check" in text
    assert "Emergency status (Tab 5 rule)" in text
    assert "Hospital ranking priority (same as Tab 5)" in text
    assert hospital_capability_text(scenario["hospital"]) in text
    assert FALLBACK_TILE_NAME in text
    assert "This app does not place calls; dial from your phone." in text
    assert "trauma-suited" not in text.lower()
    assert "gps verified" not in text.lower()
