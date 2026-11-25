"""Zero-typing emergency dispatch runner for Project Third-Eye.

Loads simulated incident locations from config/demo_incident_locations.json,
computes optimal A* emergency routes, generates route GeoJSON and PNG maps,
simulates vehicle trajectories, and persists decisions into SQLite.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.routing.dispatch import COVERAGE_NOTE, export_dispatch_summary, plan_emergency_dispatch
from core.routing.render import render_route_map, save_route_geojson
from core.routing.road_graph import (
    MAP_ATTRIBUTION,
    SPEED_DISCLAIMER,
    get_graph_stats,
    prepare_network_graph,
)
from core.routing.simulate import simulate_leg_trajectory
from db.database import (
    DEFAULT_DB_PATH,
    get_ambulances,
    get_route_results,
    init_db,
    save_ambulances,
    save_route_result,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INCIDENTS_PATH = PROJECT_ROOT / "config" / "demo_incident_locations.json"
AMBULANCES_PATH = PROJECT_ROOT / "config" / "simulated_ambulances.json"
OUTPUT_ROUTES_DIR = PROJECT_ROOT / "data" / "outputs" / "routes"


def main():
    print("=" * 78)
    print("PROJECT THIRD-EYE: MULTIMODAL EMERGENCY DISPATCH & ROUTING ENGINE")
    print("=" * 78)

    # 1. Initialize SQLite tables
    init_db()
    OUTPUT_ROUTES_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load and persist simulated ambulance fleet
    with open(AMBULANCES_PATH, "r", encoding="utf-8") as f:
        ambulance_fleet = json.load(f)
    saved_amb_count = save_ambulances(ambulance_fleet)
    print(f"[INIT] Loaded {saved_amb_count} simulated ambulances into SQLite.")

    # 3. Load or build road network graph
    print(f"[GRAPH] Loading road network graph...")
    graph = prepare_network_graph()
    stats = get_graph_stats(graph)
    print(f"  Nodes: {stats['node_count']}, Edges: {stats['edge_count']}, Total Road Length: {stats['total_road_length_km']} km")
    print(f"  Attribution: {stats['attribution']}")
    print(f"  Disclaimer: {stats['disclaimer']}")
    print(f"  Coverage: {COVERAGE_NOTE}\n")

    # 4. Load demo incident locations
    with open(INCIDENTS_PATH, "r", encoding="utf-8") as f:
        incidents = json.load(f)

    results_summary = []

    for idx, inc in enumerate(incidents, 1):
        loc_id = inc["location_id"]
        loc_name = inc["name"]
        lat = inc["latitude"]
        lon = inc["longitude"]
        env = inc.get("environment", "Unknown")

        print("-" * 78)
        print(f"SCENARIO {idx}/{len(incidents)}: {loc_name}")
        print(f"Location ID: {loc_id} | GPS: ({lat:.4f}, {lon:.4f}) [SIMULATED]")
        print(f"Environment: {env}")
        print("Severity: not determined (demo placement)")
        print("Candidate Selection Rule: All valid corridor hospitals (non-medical places, outpatient clinics, pharmacies excluded; corridor bounds: 31.20-31.27°N, 75.69-75.80°E)")

        # Plan dispatch
        route = plan_emergency_dispatch(
            incident_lat=lat,
            incident_lon=lon,
            accident_id=loc_id,
            caller_text="Two cars collided head-on, multiple passengers injured, severe bleeding.",
            graph=graph,
        )

        # Save to SQLite
        saved_route_id = save_route_result(route)

        # Export GeoJSON and PNG map
        geojson_path = OUTPUT_ROUTES_DIR / f"{loc_id}_route.geojson"
        png_path = OUTPUT_ROUTES_DIR / f"{loc_id}_route.png"

        save_route_geojson(route, (lat, lon), geojson_path)
        render_route_map(
            graph=graph,
            route=route,
            incident_coords=(lat, lon),
            output_path=png_path,
            title=f"Third-Eye Dispatch: {loc_name}",
        )

        # Run trajectory simulation on Leg 1
        traj = simulate_leg_trajectory(
            waypoints=route.leg1_waypoints or [],
            total_time_seconds=(route.ambulance_eta_minutes or 1.0) * 60.0,
            step_seconds=15.0,
            leg_name="leg1_ambulance_to_incident",
        )

        amb_cs = route.chosen_ambulance["call_sign"]
        amb_base = route.chosen_ambulance["base_hospital"]
        hosp_name = route.chosen_hospital["name"]
        hosp_emergency = route.chosen_hospital.get("emergency_tag_osm") or "not specified"
        hosp_type = route.chosen_hospital.get("facility_type", "hospital")
        hosp_specs = route.chosen_hospital.get("specialties_osm") or "not specified in OSM"
        adj_cost = route.chosen_hospital.get("ranking_adjusted_cost_sec", 0.0)
        # Authoritative leg seconds (same values exported to dispatch_summary.json).
        amb_eta_s = float(route.chosen_ambulance.get("eta_seconds", route.ambulance_eta_minutes * 60.0))
        hosp_transit_s = float(route.chosen_hospital.get("transit_seconds", route.hospital_transit_minutes * 60.0))

        # 1. Print all available ambulances evaluation
        print("\n  Available Ambulances Evaluation (all 6 fleet units):")
        print(f"    {'Callsign':<16} | {'Base Hospital':<32} | {'ETA (min / sec)':<16} | {'Dist':<8} | {'Chosen'}")
        print("    " + "-" * 88)
        for amb_alt in route.alternative_ambulances:
            is_ch = amb_alt.get("chosen", False)
            ch_str = "YES (FASTEST)" if is_ch else "No"
            t_min = amb_alt.get("travel_time_min", 0.0)
            t_sec = amb_alt.get("eta_seconds", t_min * 60.0)
            d_km = amb_alt.get("driving_distance_km", 0.0)
            cs = amb_alt.get("callsign", amb_alt.get("call_sign", "UNKNOWN"))
            bh = amb_alt.get("base_name", amb_alt.get("base_hospital", "Unknown"))
            print(f"    {cs:<16} | {bh[:32]:<32} | {t_min:4.1f}m ({t_sec:5.0f}s)   | {d_km:5.2f}km | {ch_str}")

        # 2. Print all candidate hospitals evaluation
        print(f"\n  Candidate Hospitals Evaluation (all {len(route.alternative_hospitals)} corridor facilities):")
        print(f"    {'Hospital Name':<28} | {'Driving':<8} | {'Time (min/s)':<14} | {'Penalty':<7} | {'Adj Cost':<9} | {'Considered':<10} | {'Decision Note'}")
        print("    " + "-" * 115)
        for h_alt in route.alternative_hospitals:
            h_name = h_alt.get("name", "Unknown")
            h_d_km = h_alt.get("driving_km", 0.0)
            h_t_sec = h_alt.get("transit_seconds", 0.0)
            h_t_min = h_t_sec / 60.0
            h_pen = h_alt.get("penalty", 0.0)
            h_cost = h_alt.get("adjusted_cost_seconds", 0.0)
            h_cons = "YES" if h_alt.get("considered", True) else "NO"
            h_note = h_alt.get("note", "")
            print(f"    {h_name[:28]:<28} | {h_d_km:5.2f}km  | {h_t_min:4.1f}m ({h_t_sec:4.0f}s) | {h_pen:5.2f}   | {h_cost:6.1f}s   | {h_cons:<10} | {h_note}")

        print(f"\n  [DISPATCH] Ambulance: {route.ambulance_id} ({amb_cs}) [SIMULATED]")
        print(f"             Station Base: {amb_base}")
        print(f"             Ambulance ETA: {amb_eta_s / 60:.2f} min ({amb_eta_s:.1f} sec)")
        print(f"  [HOSPITAL] Destination: {route.hospital_id} ({hosp_name}) [REAL OSM]")
        print(f"             Facility type from name/tag: {hosp_type}")
        print(f"             Emergency tag in OpenStreetMap: {hosp_emergency}")
        print(f"             Specialties in OpenStreetMap: {hosp_specs}")
        print(f"             Ranking Heuristic (not a clinical rating): travel time with penalty weighting for unsuited single-specialties")
        print(f"             Formula: adjusted_cost_sec = travel_time_sec * (1.0 + (suitability_penalty * 2.0))")
        print(f"             Adjusted Cost: {adj_cost:.1f} s")
        print(f"             Hospital Transit: {hosp_transit_s / 60:.2f} min ({hosp_transit_s:.1f} sec)")
        print(f"             Driving Distance: {route.chosen_hospital['driving_distance_km']:.2f} km")
        print(f"             Straight-line Distance: {route.straight_line_km:.2f} km")
        print(f"             Driving / Straight-Line Ratio: {route.driving_vs_straight_ratio:.2f}x")
        print(f"  [MISSION]  Total Driving: {route.distance_km:.2f} km, Estimated Total Time: {(amb_eta_s + hosp_transit_s) / 60:.2f} min")
        print(f"  [COVERAGE] {route.coverage_note}")
        print(f"  [SIMULATION] Trajectory: {len(traj)} waypoint steps generated at 15s intervals.")
        print(f"  [ARTIFACTS] Map: {png_path.name} | GeoJSON: {geojson_path.name}")
        print(f"  [DATABASE] Stored route record: {saved_route_id}")

        results_summary.append({
            "scenario": idx,
            "location_id": loc_id,
            "name": loc_name,
            "ambulance": amb_cs,
            "eta_min": amb_eta_s / 60.0,
            "hospital": hosp_name,
            "transit_min": hosp_transit_s / 60.0,
            "driving_km": route.distance_km,
            "total_min": (amb_eta_s + hosp_transit_s) / 60.0,
        })

    print("\n" + "=" * 78)
    print("DISPATCH SUMMARY TABLE")
    print("=" * 78)
    print(f"{'No':<3} | {'Incident Location':<30} | {'Ambulance':<15} | {'ETA':<7} | {'Target Hospital':<25} | {'Total'}")
    print("-" * 105)
    for r in results_summary:
        print(f"{r['scenario']:<3} | {r['name'][:30]:<30} | {r['ambulance']:<15} | {r['eta_min']:<5.1f}m | {r['hospital'][:25]:<25} | {r['total_min']:<5.1f}m")
    print("-" * 105)
    print(f"Attribution: {MAP_ATTRIBUTION}")
    print(f"Disclaimer: {SPEED_DISCLAIMER}")
    print(f"Coverage: {COVERAGE_NOTE}")
    print(f"Database Persistence Verified: {len(get_route_results())} total records in route_results table.")

    # 5. Export comprehensive dispatch summary JSON
    summary_export = export_dispatch_summary()
    print(f"Exported dispatch summary JSON: data/outputs/dispatch_summary.json ({len(summary_export['scenarios'])} scenarios)")


if __name__ == "__main__":
    main()
