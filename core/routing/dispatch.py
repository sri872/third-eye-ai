"""Emergency ambulance dispatch and hospital routing engine for Project Third-Eye.

Coordinates:
1. Nearest-ambulance dispatch based on true road network A* travel times.
2. Conservative hospital ranking from OSM tags and travel time; clinical capability is never assumed.
3. Full calculation of driving distance vs straight-line distance ratios.
4. Transparent labeling of SIMULATED fleet entities and estimated travel times.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import networkx as nx

from core.rag.facility_classification import CATEGORY_PRIORITY, classify_facility
from core.rag.hospital_data import fetch_or_load_hospitals
from core.rag.retriever import (
    emergency_suitability,
    evaluate_facility_suitability,
    formulate_retrieval_query,
    haversine_distance_km,
)
from core.routing.astar import astar_shortest_path, extract_route_coordinates
from core.routing.road_graph import (
    MAP_ATTRIBUTION,
    SPEED_DISCLAIMER,
    get_nearest_node,
    haversine_distance_m,
    prepare_network_graph,
)
from core.schemas import Hospital, IncidentReport, RouteResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AMBULANCES_PATH = PROJECT_ROOT / "config" / "simulated_ambulances.json"
DEFAULT_HOSPITALS_PATH = PROJECT_ROOT / "data" / "hospitals_osm.json"

CORRIDOR_BBOX = {
    "min_lon": 75.69,
    "min_lat": 31.20,
    "max_lon": 75.80,
    "max_lat": 31.27,
}
COVERAGE_NOTE = "Routing covers this corridor only (Phagwara corridor: 75.69-75.80°E, 31.20-31.27°N)"

AMBULANCE_SELECTION_CRITERIA = (
    "Lowest simulated road-network travel time from A* over the cached OSM graph; "
    "ties broken by shorter straight-line distance. No live GPS, vehicle location, "
    "operational status or real ambulance contact is used."
)

NOT_SPECIFIED_OSM = "not specified in OpenStreetMap"

NO_LIVE_TRAFFIC_NOTE = "Estimated from OSM speed data; no live traffic information is available."

# Same priority order as Tab 5: source-confirmed emergency first, then facility
# category, then the transparent travel-time heuristic. This is a ranking rule,
# not a verified medical-suitability claim.
HOSPITAL_RANKING_PRIORITY = (
    "source-reported emergency department (OSM emergency=yes) first; then facility "
    "category (trauma centre before hospital); then adjusted road travel time. "
    "This is a transparent heuristic, not verified medical suitability."
)


def describe_emergency_capability(hospital: Hospital) -> str:
    """Report only what the OSM record actually says about emergency capability."""
    tag = str(getattr(hospital, "emergency", "") or "").strip().lower()
    if tag == "yes":
        return "OSM tag emergency=yes (tagged emergency department; not a clinical verification)"
    if tag == "no":
        return "OSM tag emergency=no"
    return f"Emergency department and specialties {NOT_SPECIFIED_OSM}; capability unverified"


def load_simulated_ambulances(config_path: Path | str = DEFAULT_AMBULANCES_PATH) -> List[Dict[str, Any]]:
    """Load simulated ambulance fleet configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Ambulance config file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_candidate_hospitals(
    incident_text: str = "",
    hospitals_json_path: Path | str = DEFAULT_HOSPITALS_PATH,
    corridor_only: bool = True,
) -> List[Hospital]:
    """Load real OSM hospitals in the region filtered by medical validity and corridor bounds.

    Strictly enforces:
    1. Medical validity rule: excludes non-medical buildings (e.g. LPU Architecture),
       clinics, dentists, pharmacies, and stores.
    2. Geographic corridor rule: ensures candidates lie inside the drivable road graph.
    """
    all_hospitals, _, _ = fetch_or_load_hospitals(cache_path=hospitals_json_path, save_to_db=False)

    # Filter 1: Medical destination validity. Apply the same conservative classifier
    # used by Tab 5, not only the stored exclusion reason, so a stale or missing
    # reason cannot promote a mis-tagged/non-medical building to a destination.
    valid_hospitals = [
        h for h in all_hospitals
        if h.excluded_from_dispatch_reason is None
        and classify_facility(h.name).dispatch_eligible
    ]

    if not corridor_only:
        return valid_hospitals

    # Filter 2: Graph corridor coverage
    corridor_hospitals = [
        h for h in valid_hospitals
        if (CORRIDOR_BBOX["min_lat"] <= h.latitude <= CORRIDOR_BBOX["max_lat"])
        and (CORRIDOR_BBOX["min_lon"] <= h.longitude <= CORRIDOR_BBOX["max_lon"])
    ]

    if len(corridor_hospitals) < 3:
        raise ValueError(
            f"Fewer than 3 valid hospitals lie inside routing graph corridor (found {len(corridor_hospitals)})."
        )

    return corridor_hospitals


def plan_emergency_dispatch(
    incident_lat: float,
    incident_lon: float,
    accident_id: str,
    incident: Optional[IncidentReport] = None,
    caller_text: Optional[str] = None,
    graph: Optional[nx.MultiDiGraph] = None,
    ambulances: Optional[List[Dict[str, Any]]] = None,
    candidate_hospitals: Optional[List[Hospital]] = None,
) -> RouteResult:
    """Plan end-to-end emergency routing: ambulance -> incident -> optimal hospital.

    Args:
        incident_lat: Latitude of accident scene.
        incident_lon: Longitude of accident scene.
        accident_id: Unique incident/accident identifier.
        incident: Optional IncidentReport with caller keywords and triage verdict.
        caller_text: Optional caller text override.
        graph: NetworkX MultiDiGraph of driving network. Loaded if None.
        ambulances: List of ambulance dicts. Loaded from config if None.
        candidate_hospitals: List of candidate Hospital objects. Loaded if None.

    Returns:
        Populated RouteResult schema with both legs, alternative rankings, and metrics.
    """
    if graph is None:
        graph = prepare_network_graph()
    if ambulances is None:
        ambulances = load_simulated_ambulances()
    if candidate_hospitals is None:
        text_context = caller_text or (incident.summary_text if incident else "")
        candidate_hospitals = get_candidate_hospitals(text_context)
    else:
        # Caller-supplied candidates must satisfy the SAME Tab 5 eligibility rule.
        # Proximity, a hospital-shaped OSM tag or an emergency=yes tag never makes
        # a clinic, university, office or shop an emergency destination.
        eligible_candidates = [
            h for h in candidate_hospitals
            if h.excluded_from_dispatch_reason is None
            and classify_facility(h.name).dispatch_eligible
        ]
        if not eligible_candidates:
            raise ValueError(
                "No dispatch-eligible hospital among supplied candidates; "
                "unrelated and outpatient facilities are never emergency destinations."
            )
        candidate_hospitals = eligible_candidates

    # 1. Snap incident to road network
    incident_node, inc_snap_dist = get_nearest_node(graph, incident_lat, incident_lon)

    # 2. Compute driving ETA from each available ambulance to incident
    ambulance_evaluations: List[Dict[str, Any]] = []
    for amb in ambulances:
        if amb.get("status", "available") != "available":
            continue

        amb_lat = float(amb["latitude"])
        amb_lon = float(amb["longitude"])
        amb_node, _ = get_nearest_node(graph, amb_lat, amb_lon)

        path, time_sec, dist_m = astar_shortest_path(graph, amb_node, incident_node)
        straight_dist_m = haversine_distance_m(amb_lat, amb_lon, incident_lat, incident_lon)

        ambulance_evaluations.append({
            "ambulance_id": amb["ambulance_id"],
            "call_sign": amb.get("call_sign", amb["ambulance_id"]),
            "base_hospital": amb.get("base_hospital", "Unknown Base"),
            "equipment_level": amb.get("equipment_level", "Advanced"),
            "latitude": amb_lat,
            "longitude": amb_lon,
            "path_nodes": path,
            "travel_time_sec": time_sec,
            "travel_time_min": round(time_sec / 60.0, 2),
            "driving_distance_km": round(dist_m / 1000.0, 2),
            "straight_line_km": round(straight_dist_m / 1000.0, 2),
            "data_source": "SIMULATED",
        })

    if not ambulance_evaluations:
        raise ValueError("No available ambulances found for dispatch.")

    # Sort ambulances by travel_time_sec ascending, breaking ties by straight_line_km
    ambulance_evaluations.sort(key=lambda a: (a["travel_time_sec"], a["straight_line_km"]))
    best_ambulance = ambulance_evaluations[0]

    # 3. Filter and rank candidate hospitals
    incident_context = caller_text or ""
    if incident:
        incident_context += f" {' '.join(incident.injuries)} {' '.join(incident.hazards)} {incident.urgency or ''}"

    hospital_evaluations: List[Dict[str, Any]] = []
    for hosp in candidate_hospitals:
        classification = classify_facility(hosp.name)
        emergency_info = emergency_suitability(hosp)
        # Check suitability rule
        penalty, suit_rule = evaluate_facility_suitability(hosp, incident_context)

        hosp_node, _ = get_nearest_node(graph, hosp.latitude, hosp.longitude)
        path, time_sec, dist_m = astar_shortest_path(graph, incident_node, hosp_node)

        sl_dist_m = haversine_distance_m(incident_lat, incident_lon, hosp.latitude, hosp.longitude)
        sl_km = round(sl_dist_m / 1000.0, 2)
        driving_km = round(dist_m / 1000.0, 2)
        transit_min = round(time_sec / 60.0, 2)

        # Driving vs straight-line ratio
        ratio = round(dist_m / max(sl_dist_m, 1.0), 2)

        # Cost metric for selection: travel time with penalty weighting for unsuitable single-specialties
        adjusted_cost = time_sec * (1.0 + (penalty * 2.0))

        hospital_evaluations.append({
            "hospital_id": hosp.hospital_id,
            "osm_id": hosp.hospital_id,
            "name": hosp.name,
            "latitude": hosp.latitude,
            "longitude": hosp.longitude,
            "phone": hosp.contact_number,
            "emergency": hosp.emergency or "not specified",
            "emergency_tag": hosp.emergency if hosp.emergency in ("yes", "no") else None,
            "specialties": hosp.specialties,
            "specialties_osm": hosp.specialties or "not specified in OSM",
            "suitability_rule": suit_rule,
            "suitability_penalty": penalty,
            "emergency_capability": describe_emergency_capability(hosp),
            "emergency_tag_verified": str(hosp.emergency or "").strip().lower() == "yes",
            "facility_category": classification.category,
            "category_tier": CATEGORY_PRIORITY.get(classification.category, 4),
            "emergency_status": emergency_info["status"],
            "emergency_suitability": emergency_info["label"],
            "emergency_tier": emergency_info["tier"],
            "recommended_for_emergency": emergency_info["recommended_for_emergency"],
            "path_nodes": path,
            "travel_time_sec": time_sec,
            "transit_minutes": transit_min,
            "driving_distance_km": driving_km,
            "straight_line_km": sl_km,
            "driving_vs_straight_ratio": ratio,
            "adjusted_cost": adjusted_cost,
            "attribution": MAP_ATTRIBUTION,
            "data_source": "REAL",
        })

    # Sort ALL dispatch-eligible candidates by the SAME priority used in Tab 5:
    # source-reported emergency status, then facility category, then the
    # transparent adjusted travel time. No arbitrary slicing: every eligible
    # facility is evaluated and retained. This is a heuristic, not clinical rating.
    hospital_evaluations.sort(
        key=lambda h: (
            h["emergency_tier"],
            h["category_tier"],
            h["adjusted_cost"],
            h["travel_time_sec"],
        )
    )
    best_hospital = hospital_evaluations[0]

    # 4. Extract waypoints
    leg1_coords = extract_route_coordinates(graph, best_ambulance["path_nodes"])
    leg2_coords = extract_route_coordinates(graph, best_hospital["path_nodes"])
    combined_waypoints = leg1_coords + leg2_coords

    total_dist_km = round(best_ambulance["driving_distance_km"] + best_hospital["driving_distance_km"], 2)
    # Persist full-precision minutes derived from the authoritative seconds. Rounding
    # to two decimals here makes export_dispatch_summary()'s minutes * 60 disagree
    # with the seconds already stored in chosen_ambulance / chosen_hospital.
    ambulance_eta_sec = best_ambulance["travel_time_sec"]
    hospital_transit_sec = best_hospital["travel_time_sec"]
    total_time_min = (round(ambulance_eta_sec, 1) + round(hospital_transit_sec, 1)) / 60.0

    # Clean alternative dictionaries for serialization
    serializable_alt_ambs = []
    for a in ambulance_evaluations:
        is_chosen = (a["ambulance_id"] == best_ambulance["ambulance_id"])
        if is_chosen:
            reason = (
                f"Chosen: lowest simulated road-network ETA ({a['travel_time_min']:.2f} min / "
                f"{a['travel_time_sec']:.1f} s) among {len(ambulance_evaluations)} simulated units; "
                "tie-break is shorter straight-line distance."
            )
        else:
            reason = (
                f"Not chosen: simulated road-network ETA ({a['travel_time_min']:.2f} min / "
                f"{a['travel_time_sec']:.1f} s) is longer than the selected unit "
                f"({best_ambulance['travel_time_min']:.2f} min / "
                f"{best_ambulance['travel_time_sec']:.1f} s)."
            )
        serializable_alt_ambs.append({
            "id": a["ambulance_id"],
            "ambulance_id": a["ambulance_id"],
            "callsign": a["call_sign"],
            "call_sign": a["call_sign"],
            "base_name": a["base_hospital"],
            "base_hospital": a["base_hospital"],
            "base_lat": a["latitude"],
            "base_lon": a["longitude"],
            "equipment_level": a["equipment_level"],
            "eta_seconds": round(a["travel_time_sec"], 1),
            "travel_time_min": a["travel_time_min"],
            "driving_distance_km": a["driving_distance_km"],
            "selection_reason": reason,
            "chosen": is_chosen,
        })

    serializable_alt_hosps = []
    for h in hospital_evaluations:
        is_chosen = (h["hospital_id"] == best_hospital["hospital_id"])
        if is_chosen:
            note = (
                f"Chosen: highest Tab 5 priority among dispatch-eligible corridor facilities "
                f"({h['emergency_status']}, {h['facility_category']}); lowest adjusted travel "
                f"time within that tier ({h['travel_time_sec']:.1f}s). Emergency capability "
                "unverified."
            )
        elif h["suitability_penalty"] > 0:
            note = f"Not chosen: down-ranked by suitability rule ({h['suitability_rule']})"
        else:
            note = f"Not chosen: driving time ({h['travel_time_sec']:.1f}s) is longer than chosen hospital ({best_hospital['travel_time_sec']:.1f}s)"

        serializable_alt_hosps.append({
            "osm_id": h["hospital_id"],
            "hospital_id": h["hospital_id"],
            "name": h["name"],
            "driving_km": h["driving_distance_km"],
            "transit_seconds": round(h["travel_time_sec"], 1),
            "transit_minutes": h["transit_minutes"],
            "straight_line_km": h["straight_line_km"],
            "penalty": h["suitability_penalty"],
            "adjusted_cost_seconds": round(h["adjusted_cost"], 1),
            "considered": True,
            "chosen": is_chosen,
            "note": note,
            "suitability_rule": h["suitability_rule"],
            "emergency_capability": h["emergency_capability"],
            "emergency_tag_verified": h["emergency_tag_verified"],
            "facility_category": h["facility_category"],
            "emergency_status": h["emergency_status"],
            "emergency_suitability": h["emergency_suitability"],
            "recommended_for_emergency": h["recommended_for_emergency"],
            "specialties_osm": h["specialties_osm"],
            "data_source": "REAL",
        })

    route_id = f"route-{accident_id}"

    return RouteResult(
        route_id=route_id,
        ambulance_id=best_ambulance["ambulance_id"],
        hospital_id=best_hospital["hospital_id"],
        accident_id=accident_id,
        distance_km=total_dist_km,
        estimated_time_minutes=total_time_min,
        route_waypoints=combined_waypoints,
        leg1_waypoints=leg1_coords,
        leg2_waypoints=leg2_coords,
        ambulance_eta_minutes=ambulance_eta_sec / 60.0,
        hospital_transit_minutes=hospital_transit_sec / 60.0,
        straight_line_km=best_hospital["straight_line_km"],
        driving_vs_straight_ratio=best_hospital["driving_vs_straight_ratio"],
        chosen_ambulance={
            "ambulance_id": best_ambulance["ambulance_id"],
            "id": best_ambulance["ambulance_id"],
            "call_sign": best_ambulance["call_sign"],
            "callsign": best_ambulance["call_sign"],
            "base_hospital": best_ambulance["base_hospital"],
            "base_name": best_ambulance["base_hospital"],
            "base_lat": best_ambulance["latitude"],
            "base_lon": best_ambulance["longitude"],
            "equipment_level": best_ambulance["equipment_level"],
            "eta_minutes": best_ambulance["travel_time_min"],
            "eta_seconds": round(best_ambulance["travel_time_sec"], 1),
            "selection_reason": next(
                a["selection_reason"] for a in serializable_alt_ambs if a["chosen"]
            ),
            "selection_criteria": AMBULANCE_SELECTION_CRITERIA,
            "data_source": "SIMULATED",
        },
        chosen_hospital={
            "hospital_id": best_hospital["hospital_id"],
            "osm_id": best_hospital["hospital_id"],
            "name": best_hospital["name"],
            "lat": best_hospital["latitude"],
            "lon": best_hospital["longitude"],
            "phone": best_hospital.get("phone"),
            "emergency": best_hospital["emergency"],
            "emergency_tag": best_hospital.get("emergency_tag"),
            "specialties": best_hospital.get("specialties"),
            "emergency_tag_osm": best_hospital["emergency"],
            "specialties_osm": best_hospital["specialties_osm"],
            "facility_type": "hospital",
            "straight_line_km": best_hospital["straight_line_km"],
            "driving_km": best_hospital["driving_distance_km"],
            "driving_distance_km": best_hospital["driving_distance_km"],
            "transit_seconds": round(best_hospital["travel_time_sec"], 1),
            "transit_minutes": best_hospital["transit_minutes"],
            "ranking_heuristic": "travel_time_with_specialty_penalty",
            "ranking_heuristic_formula": "adjusted_cost_sec = travel_time_sec * (1.0 + (suitability_penalty * 2.0))",
            "ranking_adjusted_cost_sec": round(best_hospital["adjusted_cost"], 1),
            "suitability_penalty": best_hospital["suitability_penalty"],
            "suitability_rule": best_hospital["suitability_rule"],
            "emergency_capability": best_hospital["emergency_capability"],
            "emergency_tag_verified": best_hospital["emergency_tag_verified"],
            "facility_category": best_hospital["facility_category"],
            "emergency_status": best_hospital["emergency_status"],
            "emergency_suitability": best_hospital["emergency_suitability"],
            "recommended_for_emergency": best_hospital["recommended_for_emergency"],
            "ranking_priority": HOSPITAL_RANKING_PRIORITY,
            "data_source": "REAL",
        },
        alternative_ambulances=serializable_alt_ambs,
        alternative_hospitals=serializable_alt_hosps,
        traffic_condition="no live traffic (OSM speed estimates)",
        confidence=None,
        attribution=MAP_ATTRIBUTION,
        disclaimer=SPEED_DISCLAIMER,
        coverage_note=COVERAGE_NOTE,
        data_source="SIMULATED",
    )


NOT_DETERMINED_TEXT = "not determined"


def normalize_hospital_note(note: Any) -> str:
    """Remove unverified trauma claims from notes stored by earlier runs."""
    text = str(note or "").strip()
    text = text.replace(
        "among trauma-suited corridor facilities",
        "among dispatch-eligible corridor facilities; emergency capability unverified",
    )
    return text or NOT_DETERMINED_TEXT


def route_geometry_audit(leg1: Any, leg2: Any) -> Dict[str, Any]:
    """Derive route-geometry facts from the two stored legs, without new routing."""
    first = route_coordinates(leg1)
    second = route_coordinates(leg2)

    def segments(points: List[List[float]]) -> set:
        return {tuple(tuple(point) for point in pair) for pair in zip(points, points[1:])}

    def repeated_segments(points: List[List[float]]) -> int:
        seen: set = set()
        repeats = 0
        for pair in zip(points, points[1:]):
            key = tuple(tuple(point) for point in pair)
            if key in seen:
                repeats += 1
            seen.add(key)
        return repeats

    def length_km(points: List[List[float]]) -> float:
        return round(
            sum(
                haversine_distance_m(a[0], a[1], b[0], b[1])
                for a, b in zip(points, points[1:])
            ) / 1000.0,
            2,
        )

    return {
        "leg1_start": first[0] if first else None,
        "leg1_end": first[-1] if first else None,
        "leg2_start": second[0] if second else None,
        "leg2_end": second[-1] if second else None,
        "legs_join_at_incident": bool(first and second and first[-1] == second[0]),
        "repeated_segments_leg1": repeated_segments(first),
        "repeated_segments_leg2": repeated_segments(second),
        "shared_segments_between_legs": len(segments(first) & segments(second)),
        "leg1_polyline_km": length_km(first),
        "leg2_polyline_km": length_km(second),
        "leg1_points": len(first),
        "leg2_points": len(second),
    }


def route_coordinates(points: Any) -> List[List[float]]:
    """Return genuine [lat, lon] pairs from a stored route leg."""
    result: List[List[float]] = []
    if not isinstance(points, list):
        return result
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                lat, lon = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                continue
            result.append([lat, lon])
    return result


def export_dispatch_summary(
    db_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    incidents_path: Optional[Path | str] = None,
    routes_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Export complete dispatch summary JSON derived directly from SQLite and configuration."""
    from datetime import datetime, timezone
    from core.routing.simulate import simulate_dispatch_trajectory
    from db.database import DEFAULT_DB_PATH, db_session

    t_db = Path(db_path) if db_path else DEFAULT_DB_PATH
    out_file = Path(output_path) if output_path else PROJECT_ROOT / "data" / "outputs" / "dispatch_summary.json"
    inc_file = Path(incidents_path) if incidents_path else PROJECT_ROOT / "config" / "demo_incident_locations.json"
    r_dir = Path(routes_dir) if routes_dir else PROJECT_ROOT / "data" / "outputs" / "routes"

    with open(inc_file, "r", encoding="utf-8") as f:
        incidents_list = json.load(f)
    incidents_by_id = {inc["location_id"]: inc for inc in incidents_list}

    with db_session(t_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM route_results ORDER BY accident_id ASC;")
        route_rows = cursor.fetchall()
        cursor.execute("SELECT * FROM hospitals;")
        hosp_rows = {h["hospital_id"]: h for h in cursor.fetchall()}
        cursor.execute("SELECT * FROM ambulances;")
        amb_rows = {a["ambulance_id"]: a for a in cursor.fetchall()}

    scenarios = []
    for row in route_rows:
        loc_id = row["accident_id"]
        inc = incidents_by_id.get(loc_id, {})
        loc_name = inc.get("name", loc_id)
        inc_lat = float(inc.get("latitude", 0.0))
        inc_lon = float(inc.get("longitude", 0.0))

        chosen_amb = json.loads(row["chosen_ambulance"]) if row["chosen_ambulance"] else {}
        chosen_hosp = json.loads(row["chosen_hospital"]) if row["chosen_hospital"] else {}
        alt_ambs_raw = json.loads(row["alternative_ambulances"]) if row["alternative_ambulances"] else []
        alt_hosps_raw = json.loads(row["alternative_hospitals"]) if row["alternative_hospitals"] else []

        leg1 = json.loads(row["leg1_waypoints"]) if row["leg1_waypoints"] else []
        leg2 = json.loads(row["leg2_waypoints"]) if row["leg2_waypoints"] else []

        amb_db = amb_rows.get(row["ambulance_id"])
        hosp_db = hosp_rows.get(row["hospital_id"])

        # Hospital details from chosen_hospital or db
        hosp_phone = chosen_hosp.get("phone") or (hosp_db["contact_number"] if hosp_db and "contact_number" in hosp_db.keys() else None)
        hosp_emerg = chosen_hosp.get("emergency_tag") or (hosp_db["emergency"] if hosp_db and "emergency" in hosp_db.keys() else None)
        if hosp_emerg == "not specified":
            hosp_emerg = None
        hosp_specs = chosen_hosp.get("specialties") or (hosp_db["specialties"] if hosp_db and "specialties" in hosp_db.keys() else None)
        if hosp_specs == "not specified in OSM":
            hosp_specs = None

        # Build trajectory and timing fields from the same full-precision seconds used
        # by plan_emergency_dispatch(), so total_seconds is exactly the leg sum and the
        # chosen hospital's transit_seconds matches its alternatives-table entry.
        eta_sec = round(row["ambulance_eta_minutes"] * 60.0, 1) if row["ambulance_eta_minutes"] is not None else 60.0
        transit_sec = round(row["hospital_transit_minutes"] * 60.0, 1) if row["hospital_transit_minutes"] is not None else 0.0
        total_sec = round(eta_sec + transit_sec, 1)
        traj_raw = simulate_dispatch_trajectory(
            leg1, eta_sec, leg2, transit_sec, step_seconds=15.0
        )
        trajectory = [
            {
                "t_seconds": round(s["elapsed_sec"], 1),
                "lat": s["latitude"],
                "lon": s["longitude"],
                "leg": s["leg"],
            }
            for s in traj_raw
        ]

        # Ambulance reasons come from the stored evaluation when available. For older
        # records with no reason field, derive one from the recorded ETA values only.
        chosen_eta = eta_sec
        for old in alt_ambs_raw:
            if (old.get("id") or old.get("ambulance_id")) == row["ambulance_id"]:
                chosen_eta = float(old.get("eta_seconds", chosen_eta))
                break

        def ambulance_reason(entry: Dict[str, Any], seconds: float, is_chosen: bool) -> str:
            stored = entry.get("selection_reason") or entry.get("reason")
            if stored:
                return str(stored)
            if is_chosen:
                return (
                    f"Chosen: lowest recorded simulated road-network ETA ({seconds:.1f} s) "
                    f"among {len(alt_ambs_raw)} simulated units (derived from stored ETAs)."
                )
            return (
                f"Not chosen: recorded simulated road-network ETA ({seconds:.1f} s) is "
                f"longer than the selected unit ({chosen_eta:.1f} s) "
                "(derived from stored ETAs)."
            )

        ambulance_alternatives = []
        for a in alt_ambs_raw:
            unit_id = a.get("id") or a.get("ambulance_id")
            seconds = float(a.get("eta_seconds", round(a.get("travel_time_min", 0.0) * 60.0, 1)))
            is_chosen = unit_id == row["ambulance_id"]
            ambulance_alternatives.append({
                "id": unit_id,
                "callsign": a.get("callsign") or a.get("call_sign"),
                "base_name": a.get("base_name") or a.get("base_hospital"),
                "base_lat": a.get("base_lat"),
                "base_lon": a.get("base_lon"),
                "equipment_level": a.get("equipment_level"),
                "eta_seconds": seconds,
                "driving_distance_km": a.get("driving_distance_km"),
                "selection_reason": ambulance_reason(a, seconds, is_chosen),
                "chosen": is_chosen,
            })

        # Format hospital_alternatives: [{"osm_id","name","driving_km","transit_seconds","penalty","adjusted_cost_seconds","considered","note"}]
        def capability_from_db(osm_id: Any) -> str:
            record = hosp_rows.get(str(osm_id))
            tag = ""
            if record is not None and "emergency" in record.keys():
                tag = str(record["emergency"] or "").strip().lower()
            if tag == "yes":
                return "OSM tag emergency=yes (tagged emergency department; not a clinical verification)"
            if tag == "no":
                return "OSM tag emergency=no"
            return f"Emergency department and specialties {NOT_SPECIFIED_OSM}; capability unverified"

        hospital_alternatives = [
            {
                "osm_id": h.get("osm_id") or h.get("hospital_id"),
                "name": h.get("name"),
                "driving_km": h.get("driving_km") or h.get("driving_distance_km"),
                "transit_seconds": h.get("transit_seconds", round(h.get("transit_minutes", 0.0) * 60.0, 1)),
                "penalty": h.get("penalty", h.get("suitability_penalty", 0.0)),
                "adjusted_cost_seconds": h.get("adjusted_cost_seconds", h.get("adjusted_cost", 0.0)),
                "considered": bool(h.get("considered", True)),
                "note": normalize_hospital_note(h.get("note")),
                "emergency_capability": h.get("emergency_capability") or capability_from_db(
                    h.get("osm_id") or h.get("hospital_id")
                ),
                "facility_category": h.get("facility_category"),
                "emergency_status": h.get("emergency_status"),
                "recommended_for_emergency": h.get("recommended_for_emergency"),
                "suitability_rule": h.get("suitability_rule"),
            }
            for h in alt_hosps_raw
        ]

        png_rel = f"data/outputs/routes/{loc_id}_route.png"
        geojson_rel = f"data/outputs/routes/{loc_id}_route.geojson"

        scenario_dict = {
            "location_id": loc_id,
            "location_name": loc_name,
            "incident": {
                "lat": inc_lat,
                "lon": inc_lon,
                "data_source": "SIMULATED",
            },
            "severity": "not determined (demo placement)",
            "ambulance": {
                "id": row["ambulance_id"],
                "callsign": chosen_amb.get("callsign") or chosen_amb.get("call_sign") or row["ambulance_id"],
                "base_name": chosen_amb.get("base_name") or chosen_amb.get("base_hospital") or (amb_db["base_hospital"] if amb_db and "base_hospital" in amb_db.keys() else "Unknown Base"),
                "base_lat": float(chosen_amb.get("base_lat", amb_db["latitude"] if amb_db else 0.0)),
                "base_lon": float(chosen_amb.get("base_lon", amb_db["longitude"] if amb_db else 0.0)),
                "equipment_level": chosen_amb.get("equipment_level"),
                "eta_seconds": eta_sec,
                "driving_distance_km": chosen_amb.get("driving_distance_km") or next(
                    (
                        a.get("driving_distance_km")
                        for a in ambulance_alternatives
                        if a.get("chosen")
                    ),
                    None,
                ),
                "selection_reason": chosen_amb.get("selection_reason") or (
                    next(
                        (a["selection_reason"] for a in ambulance_alternatives if a["chosen"]),
                        NOT_DETERMINED_TEXT,
                    )
                ),
                "selection_criteria": chosen_amb.get("selection_criteria") or AMBULANCE_SELECTION_CRITERIA,
                "data_source": "SIMULATED",
            },
            "hospital": {
                "osm_id": row["hospital_id"],
                "name": chosen_hosp.get("name", hosp_db["name"] if hosp_db else row["hospital_id"]),
                "lat": float(chosen_hosp.get("lat", hosp_db["latitude"] if hosp_db else 0.0)),
                "lon": float(chosen_hosp.get("lon", hosp_db["longitude"] if hosp_db else 0.0)),
                "phone": hosp_phone,
                "emergency_tag": hosp_emerg,
                "specialties": hosp_specs,
                "emergency_capability": chosen_hosp.get("emergency_capability") or capability_from_db(row["hospital_id"]),
                "data_source": "REAL",
                "straight_line_km": float(row["straight_line_km"]) if row["straight_line_km"] is not None else 0.0,
                "driving_km": float(chosen_hosp.get("driving_km", chosen_hosp.get("driving_distance_km", 0.0))),
                "transit_seconds": transit_sec,
            },
            "leg_to_incident": leg1,
            "leg_to_hospital": leg2,
            "route_verification": route_geometry_audit(leg1, leg2),
            "location_context": {
                "dispatch_lat": inc_lat,
                "dispatch_lon": inc_lon,
                "used_for": "Both A* route legs, leg distances, leg travel times and totals in Tab 6",
                "caller_reported": None,
                "caller_used_for": "Nothing in Tab 6; caller location is preserved separately from Tab 4",
                "hospital_search": None,
                "hospital_search_used_for": "Tab 5 straight-line lookup only; never substituted in Tab 6",
                "gps_status": "NOT AVAILABLE — no device GPS verification was performed",
            },
            "total_km": float(row["distance_km"]),
            "total_seconds": total_sec,
            "traffic_note": NO_LIVE_TRAFFIC_NOTE,
            "simulation_notice": "SIMULATED DISPATCH — NO REAL AMBULANCE HAS BEEN CONTACTED",
            "ranking_formula": "adjusted_cost_sec = travel_time_sec * (1.0 + (suitability_penalty * 2.0))",
            "ranking_priority": HOSPITAL_RANKING_PRIORITY,
            "trajectory": trajectory,
            "ambulance_alternatives": ambulance_alternatives,
            "hospital_alternatives": hospital_alternatives,
            "map_png": png_rel,
            "geojson": geojson_rel,
        }
        scenarios.append(scenario_dict)

    summary_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "simulation_notice": "SIMULATED DISPATCH — NO REAL AMBULANCE HAS BEEN CONTACTED",
        "gps_status": "NOT AVAILABLE — no device GPS verification was performed",
        "traffic_note": NO_LIVE_TRAFFIC_NOTE,
        "disclaimers": [
            "© OpenStreetMap contributors",
            "Estimated from OSM speed data, no live traffic",
            COVERAGE_NOTE,
        ],
        "scenarios": scenarios,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data
