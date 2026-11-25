"""Map and GeoJSON rendering for Project Third-Eye routing.

Generates GeoJSON feature collections and high-contrast Matplotlib route maps
showing the road network, ambulance dispatch leg, and hospital transfer leg.
Strictly includes OpenStreetMap attribution and speed disclaimers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from core.routing.road_graph import MAP_ATTRIBUTION, SPEED_DISCLAIMER
from core.schemas import RouteResult


def save_route_geojson(
    route: RouteResult,
    incident_coords: Tuple[float, float],
    output_path: Path | str,
) -> Path:
    """Export complete routing solution as GeoJSON FeatureCollection."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    features: List[Dict[str, Any]] = []

    # Leg 1: Ambulance -> Incident (coordinates in GeoJSON are [lon, lat])
    if route.leg1_waypoints:
        coords_lon_lat = [[pt[1], pt[0]] for pt in route.leg1_waypoints]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords_lon_lat,
            },
            "properties": {
                "name": "Leg 1: Ambulance to Incident",
                "ambulance_id": route.ambulance_id,
                "eta_minutes": route.ambulance_eta_minutes,
                "color": "#1E88E5",
                "data_source": "SIMULATED",
            },
        })

    # Leg 2: Incident -> Hospital
    if route.leg2_waypoints:
        coords_lon_lat = [[pt[1], pt[0]] for pt in route.leg2_waypoints]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords_lon_lat,
            },
            "properties": {
                "name": "Leg 2: Incident to Hospital",
                "hospital_id": route.hospital_id,
                "transit_minutes": route.hospital_transit_minutes,
                "color": "#D81B60",
                "data_source": "SIMULATED",
            },
        })

    # Point: Ambulance Start
    if route.leg1_waypoints:
        first = route.leg1_waypoints[0]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [first[1], first[0]]},
            "properties": {
                "name": f"Ambulance Origin ({route.ambulance_id})",
                "marker-symbol": "hospital",
                "marker-color": "#1E88E5",
                "data_source": "SIMULATED",
            },
        })

    # Point: Incident Scene
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [incident_coords[1], incident_coords[0]]},
        "properties": {
            "name": f"Incident Scene ({route.accident_id})",
            "marker-symbol": "danger",
            "marker-color": "#F4511E",
            "data_source": "SIMULATED",
        },
    })

    # Point: Target Hospital
    if route.leg2_waypoints:
        last = route.leg2_waypoints[-1]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [last[1], last[0]]},
            "properties": {
                "name": f"Target Hospital ({route.hospital_id})",
                "marker-symbol": "cross",
                "marker-color": "#43A047",
                "data_source": "SIMULATED",
            },
        })

    feature_collection = {
        "type": "FeatureCollection",
        "metadata": {
            "route_id": route.route_id,
            "accident_id": route.accident_id,
            "total_distance_km": route.distance_km,
            "total_time_minutes": route.estimated_time_minutes,
            "attribution": MAP_ATTRIBUTION,
            "disclaimer": SPEED_DISCLAIMER,
            "coverage_note": route.coverage_note,
            "data_source": "SIMULATED",
        },
        "features": features,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(feature_collection, f, indent=2)

    return out


def render_route_map(
    graph: nx.MultiDiGraph,
    route: RouteResult,
    incident_coords: Tuple[float, float],
    output_path: Path | str,
    title: str = "Project Third-Eye Emergency Dispatch",
) -> Path:
    """Render high-resolution route map using Matplotlib with OSM attribution."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=200)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    # Plot road network background
    for u, v, k, data in graph.edges(keys=True, data=True):
        if "geometry" in data and data["geometry"] is not None:
            xs, ys = data["geometry"].xy
            ax.plot(xs, ys, color="#E0E0E0", linewidth=0.6, alpha=0.8, zorder=1)
        else:
            x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
            x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
            ax.plot([x1, x2], [y1, y2], color="#E0E0E0", linewidth=0.6, alpha=0.8, zorder=1)

    # Plot Leg 1 (Ambulance to Incident)
    if route.leg1_waypoints and len(route.leg1_waypoints) > 1:
        l1_lats = [pt[0] for pt in route.leg1_waypoints]
        l1_lons = [pt[1] for pt in route.leg1_waypoints]
        ax.plot(
            l1_lons,
            l1_lats,
            color="#1E88E5",
            linewidth=3.2,
            linestyle="--",
            label=f"Leg 1: Dispatched Amb ({route.ambulance_id}, ETA {route.ambulance_eta_minutes:.1f}m)",
            zorder=3,
        )

    # Plot Leg 2 (Incident to Hospital)
    if route.leg2_waypoints and len(route.leg2_waypoints) > 1:
        l2_lats = [pt[0] for pt in route.leg2_waypoints]
        l2_lons = [pt[1] for pt in route.leg2_waypoints]
        ax.plot(
            l2_lons,
            l2_lats,
            color="#D81B60",
            linewidth=3.2,
            linestyle="-",
            label=f"Leg 2: Transfer to Hosp ({route.hospital_id}, {route.hospital_transit_minutes:.1f}m)",
            zorder=4,
        )

    # Key Markers
    if route.leg1_waypoints:
        amb_start = route.leg1_waypoints[0]
        ax.scatter(
            [amb_start[1]],
            [amb_start[0]],
            color="#1E88E5",
            edgecolor="#0D47A1",
            s=120,
            marker="s",
            zorder=5,
            label=f"Ambulance Origin",
        )

    ax.scatter(
        [incident_coords[1]],
        [incident_coords[0]],
        color="#F4511E",
        edgecolor="#B71C1C",
        s=180,
        marker="*",
        zorder=6,
        label=f"Incident Scene ({route.accident_id})",
    )

    if route.leg2_waypoints:
        hosp_end = route.leg2_waypoints[-1]
        ax.scatter(
            [hosp_end[1]],
            [hosp_end[0]],
            color="#43A047",
            edgecolor="#1B5E20",
            s=140,
            marker="P",
            zorder=5,
            label=f"Destination Hospital",
        )

    # Set viewport around route bounds with margin
    all_pts = []
    if route.leg1_waypoints:
        all_pts.extend(route.leg1_waypoints)
    if route.leg2_waypoints:
        all_pts.extend(route.leg2_waypoints)
    all_pts.append([incident_coords[0], incident_coords[1]])

    all_lats = [pt[0] for pt in all_pts]
    all_lons = [pt[1] for pt in all_pts]
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    lat_pad = max((max_lat - min_lat) * 0.15, 0.008)
    lon_pad = max((max_lon - min_lon) * 0.15, 0.008)

    ax.set_ylim(min_lat - lat_pad, max_lat + lat_pad)
    ax.set_xlim(min_lon - lon_pad, max_lon + lon_pad)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude (°E)", fontsize=9, color="#555555")
    ax.set_ylabel("Latitude (°N)", fontsize=9, color="#555555")
    ax.grid(True, linestyle=":", alpha=0.4, color="#AAAAAA")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    # Attribution & Disclaimer in bottom margin
    footer_text = (
        f"{MAP_ATTRIBUTION} | {SPEED_DISCLAIMER}\n"
        f"Coverage: {route.coverage_note}\n"
        f"SIMULATED fleet & emergency model | Total Driving: {route.distance_km:.2f} km, {route.estimated_time_minutes:.1f} min"
    )
    plt.figtext(
        0.5,
        0.02,
        footer_text,
        ha="center",
        fontsize=8,
        color="#555555",
        bbox=dict(facecolor="#F0F0F0", edgecolor="#DDDDDD", boxstyle="round,pad=0.4", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out, dpi=200)
    plt.close(fig)

    return out
