"""Phase 8b dispatch helpers: summary loading, map building, and plain URLs.

This module is deliberately free of Streamlit calls so the data-facing parts can
be tested directly. All values come from data/outputs/dispatch_summary.json;
nothing is invented, and unknown values are rendered as "not determined".
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlencode

import folium

from core.hospitals.classification import CATEGORY_PRIORITY, classify_facility
from core.hospitals.location_consistency import compare_locations
from core.routing.dispatch import (
    AMBULANCE_SELECTION_CRITERIA,
    HOSPITAL_RANKING_PRIORITY,
    NO_LIVE_TRAFFIC_NOTE,
    route_geometry_audit,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DISPATCH_SUMMARY_PATH = REPO_ROOT / "data" / "outputs" / "dispatch_summary.json"
DISPATCH_SUMMARY_ENV = "THIRD_EYE_DISPATCH_SUMMARY"

NOT_DETERMINED = "not determined"
OSM_ATTRIBUTION = (
    '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)
GOOGLE_MAPS_LABEL = (
    "Opens Google Maps with Google's own routing and live traffic; "
    "this is not the project's route."
)
SIMULATION_NOTICE = "SIMULATED DISPATCH — NO REAL AMBULANCE HAS BEEN CONTACTED"
GPS_NOTICE = "NOT AVAILABLE — no device GPS verification was performed"
LIVE_TRAFFIC_NOTICE = "Estimated travel time from OSM speed data; no live traffic information is available."
NO_CALL_NOTICE = "This app does not place calls; dial from your phone."
UNVERIFIED_CAPABILITY = "Emergency department and specialties not specified in OpenStreetMap; capability unverified"
FALLBACK_TILE_NAME = "OSM HOT fallback tiles"

_REASON_KEYS = (
    "reason",
    "why",
    "why_suggested",
    "why_chosen",
    "selection_reason",
    "ranking_reason",
    "rationale",
    "suitability_rule",
    "rejection_reason",
    "explanation",
    "note",
)


# ---------------------------------------------------------------------------
# Summary loading (path is injectable for tests)
# ---------------------------------------------------------------------------
def configured_summary_path(path: str | Path | None = None) -> Path:
    """Resolve the summary path; an explicit argument wins over the environment."""
    if path is not None:
        return Path(path)
    injected = os.environ.get(DISPATCH_SUMMARY_ENV)
    if injected:
        return Path(injected)
    return DEFAULT_DISPATCH_SUMMARY_PATH


def load_dispatch_summary(path: str | Path | None = None) -> dict[str, Any]:
    """Read the real Phase 6d summary; return an empty dict if it is unavailable."""
    target = configured_summary_path(path)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def scenarios(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only well-formed scenario records from the real payload."""
    raw = payload.get("scenarios")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Value formatting: never guess a missing value
# ---------------------------------------------------------------------------
def display(value: Any) -> str:
    """Render a value or the explicit unknown marker."""
    if value is None:
        return NOT_DETERMINED
    if isinstance(value, str) and not value.strip():
        return NOT_DETERMINED
    return str(value)


def with_unit(value: Any, unit: str) -> str:
    """Attach a unit only when the real value exists."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return NOT_DETERMINED
    return f"{display(value)} {unit}"


def optional_value(value: Any, missing_label: str) -> str:
    """Render an explicitly documented fallback for a nullable OSM field."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return missing_label
    return str(value)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def number_text(value: Any) -> str:
    """Compact number formatting used for replay labels and coordinates."""
    number = _as_float(value)
    if number is None:
        return NOT_DETERMINED
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def coordinate_pair(
    mapping: Any, lat_key: str = "lat", lon_key: str = "lon"
) -> tuple[float, float] | None:
    """Return a coordinate pair only when both real components are present."""
    if not isinstance(mapping, dict):
        return None
    lat = _as_float(mapping.get(lat_key))
    lon = _as_float(mapping.get(lon_key))
    if lat is None or lon is None:
        return None
    return lat, lon


def coord_text(pair: tuple[float, float] | None) -> str | None:
    """Format a pair for the official Google Maps URL scheme."""
    if pair is None:
        return None
    return f"{number_text(pair[0])},{number_text(pair[1])}"


def route_coordinates(points: Any) -> list[tuple[float, float]]:
    """Convert a [[lat, lon], ...] route leg into folium-friendly pairs."""
    result: list[tuple[float, float]] = []
    if not isinstance(points, list):
        return result
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            pair = coordinate_pair({"lat": point[0], "lon": point[1]})
        else:
            pair = coordinate_pair(point)
        if pair is not None:
            result.append(pair)
    return result


# ---------------------------------------------------------------------------
# Location provenance: three separate sources, no silent substitution
# ---------------------------------------------------------------------------
def location_audit(
    report: dict[str, Any] | None,
    selected_location: dict[str, Any] | None,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Keep caller, hospital-search and dispatch locations separate.

    The caller location is never used for a Tab 6 calculation. The dispatch demo
    location from the summary is the only location used for its two route legs.
    """
    report = report or {}
    selected_location = selected_location or {}
    incident = scenario.get("incident") or {}

    def coordinates(entry: dict[str, Any], lat_key: str = "latitude", lon_key: str = "longitude") -> str:
        pair = coordinate_pair(entry, lat_key, lon_key)
        if pair is None:
            return "not determined"
        return f"{number_text(pair[0])}, {number_text(pair[1])}"

    caller = display(report.get("caller_location") or report.get("location"))
    caller_value = NOT_DETERMINED if caller == NOT_DETERMINED else caller
    dispatch_location = {
        "name": scenario.get("location_name"),
        "latitude": incident.get("lat"),
        "longitude": incident.get("lon"),
    }
    caller_vs_dispatch = compare_locations(dispatch_location, None if caller_value == NOT_DETERMINED else caller_value)
    caller_vs_sidebar = compare_locations(selected_location, None if caller_value == NOT_DETERMINED else caller_value)

    rows = [
        {
            "Source": "SIMULATED sample call (Tab 4)",
            "Reported location": caller_value,
            "Used for in Tab 6": "Nothing — preserved separately, never substituted",
        },
        {
            "Source": "SIMULATED hospital-search selection (sidebar / Tab 5)",
            "Reported location": (
                f"{display(selected_location.get('name'))} ({coordinates(selected_location)})"
            ),
            "Used for in Tab 6": "Nothing — Tab 5 straight-line lookup only",
        },
        {
            "Source": "SIMULATED dispatch demo location (Tab 6 summary)",
            "Reported location": (
                f"{display(scenario.get('location_name'))} ({coordinates(dispatch_location)})"
            ),
            "Used for in Tab 6": "Both route legs, distances, travel times and totals",
        },
    ]

    discrepancies: list[str] = []
    if caller_vs_dispatch.status == "LOCATION MISMATCH":
        discrepancies.append(
            f"Caller and dispatch locations differ: caller said '{caller_value}', "
            f"dispatch uses '{display(scenario.get('location_name'))}'."
        )
    if caller_vs_sidebar.status == "LOCATION MISMATCH":
        discrepancies.append(
            f"Caller and hospital-search locations differ: caller said '{caller_value}', "
            f"hospital search used '{display(selected_location.get('name'))}'."
        )
    selected_text = str(selected_location.get("name") or "").strip()
    dispatch_text = str(scenario.get("location_name") or "").strip()
    if selected_text and dispatch_text and selected_text != dispatch_text:
        discrepancies.append(
            f"Dispatch demo location '{dispatch_text}' differs from the selected hospital-search "
            f"location '{selected_text}'. Both are retained; no substitution was made."
        )
    if not discrepancies:
        discrepancies.append("No location discrepancy was detected from the available records.")

    return {
        "rows": rows,
        "discrepancies": discrepancies,
        "dispatch_status": caller_vs_dispatch.status,
        "sidebar_status": caller_vs_sidebar.status,
        "gps_status": GPS_NOTICE,
        "dispatch_coordinates_used": coordinates(dispatch_location),
    }


def ambulance_selection_reason(scenario: dict[str, Any]) -> str:
    """Return the stored reason, or a data-derived one; never a guessed status."""
    ambulance = scenario.get("ambulance") or {}
    reason = display(ambulance.get("selection_reason"))
    if reason != NOT_DETERMINED:
        return reason
    alternatives = scenario.get("ambulance_alternatives") or []
    chosen = next((a for a in alternatives if isinstance(a, dict) and a.get("chosen")), None)
    reason = display((chosen or {}).get("selection_reason"))
    if reason != NOT_DETERMINED:
        return reason
    etas = [
        _as_float(a.get("eta_seconds"))
        for a in alternatives
        if isinstance(a, dict) and _as_float(a.get("eta_seconds")) is not None
    ]
    eta = _as_float(ambulance.get("eta_seconds"))
    if eta is not None and etas and abs(min(etas) - eta) < 0.05:
        return (
            f"Derived from recorded ETAs: selected unit has the lowest simulated road-network ETA "
            f"({number_text(eta)} s) among {len(etas)} simulated units."
        )
    return NOT_DETERMINED


def hospital_capability_text(hospital: dict[str, Any]) -> str:
    """Report emergency capability using only OSM tags present in the data."""
    stored = display(hospital.get("emergency_capability"))
    if stored != NOT_DETERMINED:
        return stored
    tag = hospital.get("emergency_tag")
    if str(tag or "").strip().lower() == "yes":
        return "OSM tag emergency=yes (tagged emergency department; not a clinical verification)"
    if str(tag or "").strip().lower() == "no":
        return "OSM tag emergency=no"
    return UNVERIFIED_CAPABILITY


def route_audit_lines(scenario: dict[str, Any]) -> list[str]:
    """Human-readable checks derived from the same two stored route legs."""
    verification = route_verification(scenario)
    ambulance = scenario.get("ambulance") or {}
    hospital = scenario.get("hospital") or {}
    lines: list[str] = []

    for label, key, suffix in (
        ("Leg 1 ambulance base → incident", "leg1_start", "leg1_end"),
        ("Leg 2 incident → hospital", "leg2_start", "leg2_end"),
    ):
        start, end = verification.get(key), verification.get(suffix)
        if isinstance(start, list) and isinstance(end, list) and len(start) == 2 and len(end) == 2:
            lines.append(
                f"{label}: ({number_text(start[0])}, {number_text(start[1])}) → "
                f"({number_text(end[0])}, {number_text(end[1])})."
            )
        else:
            lines.append(f"{label}: not determined.")

    join = verification.get("legs_join_at_incident")
    lines.append(f"Legs join at the incident node: {'yes' if join is True else 'no / not determined'}.")
    lines.append(
        "Repeated route segments: "
        f"{display(verification.get('repeated_segments_leg1'))} in leg 1, "
        f"{display(verification.get('repeated_segments_leg2'))} in leg 2; "
        f"{display(verification.get('shared_segments_between_legs'))} shared between legs."
    )

    leg1_km = _as_float(verification.get("leg1_polyline_km"))
    leg2_km = _as_float(verification.get("leg2_polyline_km"))
    if leg1_km is not None and leg2_km is not None:
        lines.append(
            f"Polyline lengths: {leg1_km:.2f} km + {leg2_km:.2f} km = {leg1_km + leg2_km:.2f} km "
            "(straight segments between stored graph nodes)."
        )
    lines.append(
        "Recorded leg distances/times: "
        f"ambulance {with_unit(ambulance.get('driving_distance_km'), 'km')} / "
        f"{with_unit(ambulance.get('eta_seconds'), 's')}; "
        f"hospital {with_unit(hospital.get('driving_km'), 'km')} / "
        f"{with_unit(hospital.get('transit_seconds'), 's')}."
    )
    lines.append(f"Time basis: {LIVE_TRAFFIC_NOTICE}")
    lines.append(f"GPS: {GPS_NOTICE}")
    return lines


def route_verification(scenario: dict[str, Any]) -> dict[str, Any]:
    """Use stored verification, or derive it from the stored legs when absent."""
    stored = scenario.get("route_verification")
    if isinstance(stored, dict) and stored:
        return stored
    return route_geometry_audit(
        scenario.get("leg_to_incident"), scenario.get("leg_to_hospital")
    )


def routing_check_lines(scenario: dict[str, Any]) -> list[str]:
    """Check recorded totals against the two recorded legs, with units shown."""
    ambulance = scenario.get("ambulance") or {}
    hospital = scenario.get("hospital") or {}
    leg1_km = _as_float(ambulance.get("driving_distance_km"))
    leg2_km = _as_float(hospital.get("driving_km"))
    total_km = _as_float(scenario.get("total_km"))
    eta_sec = _as_float(ambulance.get("eta_seconds"))
    transit_sec = _as_float(hospital.get("transit_seconds"))
    total_sec = _as_float(scenario.get("total_seconds"))

    lines: list[str] = []
    if leg1_km is not None and leg2_km is not None and total_km is not None:
        expected = round(leg1_km + leg2_km, 2)
        status = "matches" if abs(expected - total_km) <= 0.02 else "DIFFERS"
        lines.append(
            f"Distance totals: {leg1_km:.2f} km + {leg2_km:.2f} km = {expected:.2f} km; "
            f"recorded total {total_km:.2f} km — {status}."
        )
    else:
        lines.append("Distance totals: at least one leg value is not determined.")

    if eta_sec is not None and transit_sec is not None and total_sec is not None:
        expected = round(eta_sec + transit_sec, 1)
        status = "matches" if abs(expected - total_sec) <= 0.1 else "DIFFERS"
        lines.append(
            f"Time totals: {eta_sec:.1f} s + {transit_sec:.1f} s = {expected:.1f} s; "
            f"recorded total {total_sec:.1f} s — {status}."
        )
    else:
        lines.append("Time totals: at least one leg value is not determined.")

    trajectory = scenario.get("trajectory")
    if isinstance(trajectory, list) and trajectory:
        last = trajectory[-1]
        last_t = _as_float(last.get("t_seconds")) if isinstance(last, dict) else None
        if last_t is not None and total_sec is not None:
            status = "matches" if abs(last_t - total_sec) <= 0.1 else "DIFFERS"
            lines.append(
                f"Replay clock: final step {last_t:.1f} s; recorded total {total_sec:.1f} s — {status}."
            )
        legs = sorted(
            {
                str(step.get("leg"))
                for step in trajectory
                if isinstance(step, dict) and step.get("leg")
            }
        )
        lines.append(f"Replay covers: {', '.join(legs) if legs else NOT_DETERMINED}.")
    else:
        lines.append("Replay clock: not determined.")
    lines.append(f"Traffic basis: {NO_LIVE_TRAFFIC_NOTE}")
    lines.append(f"GPS: {GPS_NOTICE}")
    return lines


def replay_phase(scenario: dict[str, Any], label: str | None) -> str:
    """Describe which simulated leg a replay label belongs to."""
    if label is None:
        return NOT_DETERMINED
    trajectory = scenario.get("trajectory")
    if isinstance(trajectory, list):
        for step in trajectory:
            if not isinstance(step, dict):
                continue
            if f"t = {number_text(step.get('t_seconds'))} s" == label:
                leg = str(step.get("leg") or "").strip()
                if leg == "leg1_ambulance_to_incident":
                    return "ambulance base → incident"
                if leg == "leg2_incident_to_hospital":
                    return "incident → hospital"
                return display(step.get("leg"))
    return NOT_DETERMINED


def _point_to_segment_m(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """Approximate perpendicular distance in metres using a local flat-earth scale."""
    lat_scale = 111_320.0
    lon_scale = 111_320.0 * math.cos(math.radians(point[0]))
    px, py = point[1] * lon_scale, point[0] * lat_scale
    ax, ay = a[1] * lon_scale, a[0] * lat_scale
    bx, by = b[1] * lon_scale, b[0] * lat_scale
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return math.hypot(px - ax, py - ay)
    fraction = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + fraction * dx), py - (ay + fraction * dy))


def replay_follows_route(scenario: dict[str, Any], label: str | None) -> bool | None:
    """Check a replay marker against the stored geometry of ITS OWN leg only.

    Combining both legs would let a marker on the hospital leg be reported as a
    valid ambulance replay step, so the trajectory's leg label is honoured.
    """
    position = replay_position(scenario, label)
    if position is None:
        return None
    phase = replay_phase(scenario, label)
    if phase == "ambulance base → incident":
        points = route_coordinates(scenario.get("leg_to_incident"))
    elif phase == "incident → hospital":
        points = route_coordinates(scenario.get("leg_to_hospital"))
    else:
        points = route_coordinates(scenario.get("leg_to_incident")) + route_coordinates(
            scenario.get("leg_to_hospital")
        )
    if len(points) < 2:
        return None
    return min(
        _point_to_segment_m(position, a, b) for a, b in zip(points, points[1:])
    ) <= 50.0


def fallback_geometry_check(
    scenario: dict[str, Any], repo_root: Path | None = None
) -> dict[str, Any]:
    """Verify that static fallback artefacts use the same legs as the map."""
    root = repo_root or REPO_ROOT
    result: dict[str, Any] = {
        "geojson_available": False,
        "geojson_matches_legs": None,
        "png_available": False,
        "detail": NOT_DETERMINED,
    }

    def resolve(value: Any) -> Path | None:
        text = display(value)
        if text == NOT_DETERMINED:
            return None
        path = Path(text)
        return path if path.is_absolute() else root / path

    geojson_path = resolve(scenario.get("geojson"))
    if geojson_path is not None and geojson_path.is_file():
        try:
            payload = json.loads(geojson_path.read_text(encoding="utf-8"))
            lines = [
                feature.get("geometry", {}).get("coordinates", [])
                for feature in payload.get("features", [])
                if isinstance(feature, dict)
                and feature.get("geometry", {}).get("type") == "LineString"
            ]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            lines = []
        if len(lines) >= 2:
            expected = [
                [[lon, lat] for lat, lon in route_coordinates(scenario.get("leg_to_incident"))],
                [[lon, lat] for lat, lon in route_coordinates(scenario.get("leg_to_hospital"))],
            ]
            result["geojson_available"] = True
            result["geojson_matches_legs"] = lines[:2] == expected
            result["detail"] = (
                "GeoJSON leg geometry matches the summary route legs."
                if result["geojson_matches_legs"]
                else "GeoJSON leg geometry differs from the summary route legs."
            )

    png_path = resolve(scenario.get("map_png"))
    result["png_available"] = bool(png_path is not None and png_path.is_file())
    return result


# ---------------------------------------------------------------------------
# Simulated replay steps (taken verbatim from the trajectory)
# ---------------------------------------------------------------------------
def replay_labels(scenario: dict[str, Any]) -> list[str]:
    """Build selectbox labels such as "t = 0 s, 15 s, ..." from the trajectory."""
    trajectory = scenario.get("trajectory")
    labels: list[str] = []
    if not isinstance(trajectory, list):
        return labels
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        seconds = number_text(step.get("t_seconds"))
        if seconds == NOT_DETERMINED:
            continue
        label = f"t = {seconds} s"
        if label not in labels:
            labels.append(label)
    return labels


def replay_position(
    scenario: dict[str, Any], label: str | None
) -> tuple[float, float] | None:
    """Find the trajectory coordinate matching one replay label."""
    if label is None:
        return None
    trajectory = scenario.get("trajectory")
    if not isinstance(trajectory, list):
        return None
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        seconds = number_text(step.get("t_seconds"))
        if seconds != NOT_DETERMINED and f"t = {seconds} s" == label:
            return coordinate_pair(step, "lat", "lon")
    return None


# ---------------------------------------------------------------------------
# Official Google Maps Directions URL (no API key, no project routing)
# ---------------------------------------------------------------------------
def build_google_maps_url(
    origin: str,
    destination: str,
    waypoints: Iterable[str] | None = None,
    travelmode: str = "driving",
) -> str:
    """Build a keyless Google Maps directions URL with the official api=1 scheme."""
    params: list[tuple[str, str]] = [
        ("api", "1"),
        ("origin", origin),
        ("destination", destination),
    ]
    points = list(waypoints) if waypoints else []
    if points:
        params.append(("waypoints", "|".join(points)))
    params.append(("travelmode", travelmode))
    return "https://www.google.com/maps/dir/?" + urlencode(params, safe=",-.|")


# ---------------------------------------------------------------------------
# Folium map (OSM tiles, real route legs, simulated replay marker)
# ---------------------------------------------------------------------------
def iter_map_markers(fmap: folium.Map) -> Iterator[Any]:
    """Yield every marker in a folium map, including nested feature groups."""
    for child in getattr(fmap, "_children", {}).values():
        if isinstance(child, folium.Marker):
            yield child
        yield from iter_map_markers(child)


def marker_tooltip(marker: Any) -> str:
    """Read a marker's tooltip text without relying on folium internals."""
    for child in getattr(marker, "_children", {}).values():
        if type(child).__name__ == "Tooltip":
            return str(getattr(child, "text", ""))
    return ""


def build_dispatch_map(
    scenario: dict[str, Any], replay_label: str | None = None
) -> folium.Map:
    """Build the dispatch map from real summary coordinates and route legs."""
    incident = coordinate_pair(scenario.get("incident") or {}, "lat", "lon")
    base = coordinate_pair(scenario.get("ambulance") or {}, "base_lat", "base_lon")
    hospital = coordinate_pair(scenario.get("hospital") or {}, "lat", "lon")

    center = base or incident or hospital or [0.0, 0.0]
    fmap = folium.Map(location=list(center), zoom_start=13, tiles=None)
    folium.TileLayer(
        tiles="openstreetmap",
        attr=OSM_ATTRIBUTION,
        name="OpenStreetMap",
        control=True,
        show=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr=(
            '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors (HOT tile style)'
        ),
        name=FALLBACK_TILE_NAME,
        overlay=False,
        control=True,
        show=False,
    ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    if incident:
        folium.Marker(
            location=list(incident),
            tooltip="SIMULATED: incident (demo placement)",
            icon=folium.Icon(color="red", icon="warning-sign", prefix="glyphicon"),
        ).add_to(fmap)
    if base:
        folium.Marker(
            location=list(base),
            tooltip="SIMULATED: ambulance base",
            icon=folium.Icon(color="orange", icon="plus-sign", prefix="glyphicon"),
        ).add_to(fmap)
    if hospital:
        eligibility = hospital_eligibility(scenario)
        tooltip = (
            "REAL OSM: chosen hospital"
            if eligibility["dispatch_eligible"]
            else "INELIGIBLE FACILITY — not a dispatch hospital (Tab 5 rule)"
        )
        folium.Marker(
            location=list(hospital),
            tooltip=tooltip,
            icon=folium.Icon(
                color="green" if eligibility["dispatch_eligible"] else "red",
                icon="plus-sign",
                prefix="glyphicon",
            ),
        ).add_to(fmap)

    replay = replay_position(scenario, replay_label)
    if replay:
        phase = replay_phase(scenario, replay_label)
        phase_text = "" if phase == NOT_DETERMINED else f" ({phase})"
        folium.Marker(
            location=list(replay),
            tooltip=f"SIMULATED ambulance replay: {replay_label}{phase_text}",
            icon=folium.Icon(color="blue", icon="ambulance", prefix="fa"),
        ).add_to(fmap)

    leg_to_incident = route_coordinates(scenario.get("leg_to_incident"))
    leg_to_hospital = route_coordinates(scenario.get("leg_to_hospital"))
    if len(leg_to_incident) > 1:
        folium.PolyLine(
            leg_to_incident,
            color="#d93025",
            weight=5,
            tooltip="SIMULATED: ambulance base to incident (no live traffic)",
        ).add_to(fmap)
    if len(leg_to_hospital) > 1:
        folium.PolyLine(
            leg_to_hospital,
            color="#1a73e8",
            weight=5,
            tooltip="SIMULATED: incident to hospital (no live traffic)",
        ).add_to(fmap)

    bounds = [point for point in (base, incident, hospital) if point]
    bounds.extend(leg_to_incident)
    bounds.extend(leg_to_hospital)
    if len(bounds) >= 2:
        fmap.fit_bounds(bounds, padding=(25, 25))
    elif bounds:
        fmap.location = list(bounds[0])
    return fmap


# ---------------------------------------------------------------------------
# Alternatives and emergency numbers
# ---------------------------------------------------------------------------
def _first_present(entry: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = entry.get(key)
        if value is not None and value != "":
            return value
    return None


def _seconds_or_minutes(entry: dict[str, Any], seconds_key: str, minutes_key: str) -> Any:
    seconds = _as_float(entry.get(seconds_key))
    if seconds is not None:
        return seconds
    minutes = _as_float(entry.get(minutes_key))
    return None if minutes is None else minutes * 60.0


def alternative_rows(entries: Any, default_badge: str) -> list[dict[str, str]]:
    """Render concise alternative rows, keeping the recorded selection reason."""
    rows: list[dict[str, str]] = []
    if not isinstance(entries, list):
        return rows
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        reason = NOT_DETERMINED
        for key in _REASON_KEYS:
            value = entry.get(key)
            if value is not None and value != "":
                reason = display(value)
                break
        badge = display(entry.get("data_source")) if entry.get("data_source") else default_badge

        if _first_present(entry, ("callsign", "call_sign")) is not None:
            lat = _first_present(entry, ("base_lat", "latitude"))
            lon = _first_present(entry, ("base_lon", "longitude"))
            coords = (
                f"{number_text(lat)}, {number_text(lon)}"
                if lat is not None and lon is not None
                else NOT_DETERMINED
            )
            row = {
                "ID": display(_first_present(entry, ("id", "ambulance_id"))),
                "Callsign": display(_first_present(entry, ("callsign", "call_sign"))),
                "Base (SIMULATED)": display(_first_present(entry, ("base_name", "base_hospital"))),
                "Base coordinates (SIMULATED)": coords,
                "ETA": with_unit(
                    _seconds_or_minutes(entry, "eta_seconds", "travel_time_min"), "s"
                ),
                "Road distance": with_unit(entry.get("driving_distance_km"), "km"),
                "Selection reason": reason,
                "Source": badge,
            }
        elif _first_present(entry, ("osm_id", "hospital_id")) is not None:
            row = {
                "Facility (REAL OSM)": display(entry.get("name")),
                "OSM ID": display(_first_present(entry, ("osm_id", "hospital_id"))),
                "Road distance": with_unit(
                    _first_present(entry, ("driving_km", "driving_distance_km")), "km"
                ),
                "Transit": with_unit(
                    _seconds_or_minutes(entry, "transit_seconds", "transit_minutes"), "s"
                ),
                "Penalty": display(_first_present(entry, ("penalty", "suitability_penalty"))),
                "Adjusted cost": with_unit(
                    _first_present(entry, ("adjusted_cost_seconds", "adjusted_cost")), "s"
                ),
                "Emergency capability": display(
                    entry.get("emergency_capability") or UNVERIFIED_CAPABILITY
                ),
                "Ranking reason": reason,
                "Source": badge,
            }
        else:
            row = {str(key): display(value) for key, value in entry.items()}
            row["Reason"] = reason
            row["Source"] = badge
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Tab 5-consistent eligibility, ranking and navigation checks
# ---------------------------------------------------------------------------
def hospital_eligibility(scenario: dict[str, Any]) -> dict[str, Any]:
    """Apply the Tab 5 classifier to the facility stored as the Tab 6 destination."""
    hospital = scenario.get("hospital") or {}
    name = display(hospital.get("name"))
    classification = classify_facility(name)
    eligible = classification.dispatch_eligible
    status = (
        "DISPATCH-ELIGIBLE HOSPITAL / TRAUMA CENTRE"
        if eligible
        else "NOT ELIGIBLE — do not use as emergency destination"
    )
    return {
        "name": name,
        "osm_id": display(hospital.get("osm_id")),
        "category": classification.category,
        "dispatch_eligible": eligible,
        "status": status,
        "detail": classification.reason or (
            "Name/tag evidence establishes a hospital or trauma centre; "
            "emergency department capability remains source-reported only."
        ),
    }


def emergency_status_from_capability(capability: Any) -> str:
    """Render the stored OSM emergency tag without upgrading it to verification."""
    text = str(capability or "").lower()
    if "emergency=yes" in text:
        return "OSM emergency=yes (source-reported; not independently verified)"
    if "emergency=no" in text:
        return "OSM emergency=no (no emergency department reported in source)"
    return "not specified in OpenStreetMap — emergency department unverified"


def hospital_alternative_rows(scenario: dict[str, Any]) -> list[dict[str, str]]:
    """Tab 5-labelled hospital alternatives, including which one was chosen."""
    rows = alternative_rows(scenario.get("hospital_alternatives"), "REAL")
    chosen_id = display((scenario.get("hospital") or {}).get("osm_id"))
    for row in rows:
        osm_id = display(row.get("OSM ID"))
        row["Chosen"] = "yes" if chosen_id != NOT_DETERMINED and osm_id == chosen_id else "no"
        name = display(row.get("Facility (REAL OSM)"))
        row["Tab 5 category"] = classify_facility(name).category
        row["Emergency status"] = emergency_status_from_capability(
            row.get("Emergency capability")
        )
    return rows


def selection_consistency(scenario: dict[str, Any]) -> dict[str, Any]:
    """Check the recorded choice against the recorded alternative scores."""
    ambulance = scenario.get("ambulance") or {}
    hospital = scenario.get("hospital") or {}
    ambulance_etas = [
        value
        for entry in (scenario.get("ambulance_alternatives") or [])
        if isinstance(entry, dict)
        and (value := _as_float(entry.get("eta_seconds"))) is not None
    ]
    chosen_eta = _as_float(ambulance.get("eta_seconds"))
    ambulance_lowest = bool(
        chosen_eta is not None and ambulance_etas and abs(min(ambulance_etas) - chosen_eta) <= 0.05
    )

    hospital_costs = [
        value
        for entry in (scenario.get("hospital_alternatives") or [])
        if isinstance(entry, dict)
        and (value := _as_float(entry.get("adjusted_cost_seconds"))) is not None
    ]
    chosen_id = display(hospital.get("osm_id"))
    chosen_cost = next(
        (
            _as_float(entry.get("adjusted_cost_seconds"))
            for entry in (scenario.get("hospital_alternatives") or [])
            if isinstance(entry, dict)
            and display(entry.get("osm_id") or entry.get("hospital_id")) == chosen_id
        ),
        None,
    )
    hospital_lowest = bool(
        chosen_cost is not None and hospital_costs and abs(min(hospital_costs) - chosen_cost) <= 0.05
    )

    def priority(entry: dict[str, Any]) -> tuple[int, int]:
        capability = str(entry.get("emergency_capability") or "").lower()
        if "emergency=yes" in capability:
            emergency_tier = 0
        elif "emergency=no" in capability:
            emergency_tier = 2
        else:
            emergency_tier = 1
        category = classify_facility(str(entry.get("name") or "")).category
        return emergency_tier, CATEGORY_PRIORITY.get(category, 4)

    hospital_entries = [
        entry for entry in (scenario.get("hospital_alternatives") or [])
        if isinstance(entry, dict)
    ]
    chosen_priority = priority(
        {
            "name": hospital.get("name"),
            "emergency_capability": hospital.get("emergency_capability"),
        }
    )
    hospital_priority_ok = bool(
        hospital_entries
        and chosen_priority == min(priority(entry) for entry in hospital_entries)
        and hospital_lowest
    )
    details = [
        f"Ambulance: chosen ETA {with_unit(chosen_eta, 's')}; "
        f"lowest recorded ETA {with_unit(min(ambulance_etas) if ambulance_etas else None, 's')} "
        f"— {'matches' if ambulance_lowest else 'DIFFERS / not determined'}.",
        f"Hospital: chosen adjusted cost {with_unit(chosen_cost, 's')}; "
        f"lowest recorded cost {with_unit(min(hospital_costs) if hospital_costs else None, 's')} "
        f"— {'matches' if hospital_lowest else 'DIFFERS / not determined'}.",
        f"Hospital Tab 5 priority tier: {chosen_priority} "
        f"— {'matches highest available tier' if hospital_priority_ok else 'DIFFERS / not determined'}.",
        f"Priority rule: {HOSPITAL_RANKING_PRIORITY}",
    ]
    return {
        "ambulance_lowest_eta": ambulance_lowest,
        "hospital_lowest_adjusted_cost": hospital_lowest,
        "hospital_priority_ok": hospital_priority_ok,
        "details": details,
    }


def navigation_ready(pair: tuple[float, float] | None) -> bool:
    """Never build an external navigation URL from a missing/placeholder coordinate."""
    return pair is not None and pair != (0.0, 0.0)


def emergency_status(contact: dict[str, Any]) -> str:
    """Only call a number verified when the source data says so and names a source."""
    status = str(contact.get("verification_status") or "").strip().upper()
    has_source = bool(contact.get("official_source") or contact.get("source_url"))
    return "verified" if status == "VERIFIED" and has_source else "UNVERIFIED"


def emergency_plain_text(contact: dict[str, Any]) -> str:
    """Plain-text emergency contact rendering (no phone links of any kind)."""
    return (
        f"{display(contact.get('number'))} — {display(contact.get('service'))} — "
        f"{emergency_status(contact)} — source: {display(contact.get('official_source'))} "
        f"({display(contact.get('source_url'))})"
    )
