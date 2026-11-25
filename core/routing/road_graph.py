"""Road network extraction and processing for Project Third-Eye.

Loads OpenStreetMap driving network for the Phagwara corridor, populates
edge lengths and travel times, and caches the network as GraphML.
Strictly attributes data to '© OpenStreetMap contributors'.
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import networkx as nx
import osmnx as ox

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OSM_PATH = PROJECT_ROOT / "data" / "phagwara_corridor.osm"
DEFAULT_GRAPHML_PATH = PROJECT_ROOT / "data" / "phagwara_drive_network.graphml"

MAP_ATTRIBUTION = "© OpenStreetMap contributors"
SPEED_DISCLAIMER = "Estimated from OSM speed data, no live traffic"

# Speed defaults by highway type (km/h) for Indian regional roads
DEFAULT_HIGHWAY_SPEEDS: Dict[str, float] = {
    "motorway": 80.0,
    "trunk": 65.0,
    "trunk_link": 40.0,
    "primary": 50.0,
    "primary_link": 35.0,
    "secondary": 40.0,
    "secondary_link": 30.0,
    "tertiary": 30.0,
    "tertiary_link": 25.0,
    "residential": 25.0,
    "unclassified": 25.0,
    "living_street": 15.0,
    "service": 15.0,
}
FALLBACK_SPEED_KPH = 35.0


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two coordinates in meters."""
    r_earth = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r_earth * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def prepare_network_graph(
    osm_path: Path | str = DEFAULT_OSM_PATH,
    graphml_path: Path | str = DEFAULT_GRAPHML_PATH,
    force_rebuild: bool = False,
) -> nx.MultiDiGraph:
    """Build or load the driving network graph with edge lengths and travel times.
    
    If graphml_path exists and not force_rebuild, loads cached GraphML.
    Otherwise parses the OSM XML, assigns speeds and travel times, and saves GraphML.
    """
    g_path = Path(graphml_path)
    o_path = Path(osm_path)

    if g_path.exists() and not force_rebuild:
        return load_graphml_network(g_path)

    if not o_path.exists():
        raise FileNotFoundError(f"Source OSM XML corridor not found at {o_path}")

    # Parse OSM XML into a MultiDiGraph
    graph = ox.graph_from_xml(str(o_path))

    # Populate edge lengths if missing
    for u, v, k, data in graph.edges(keys=True, data=True):
        if "length" not in data or data["length"] is None or float(data["length"]) <= 0:
            n1 = graph.nodes[u]
            n2 = graph.nodes[v]
            dist = haversine_distance_m(n1["y"], n1["x"], n2["y"], n2["x"])
            data["length"] = max(dist, 1.0)
        else:
            data["length"] = float(data["length"])

    # Assign speeds and travel times
    for u, v, k, data in graph.edges(keys=True, data=True):
        hwy = data.get("highway", "unclassified")
        if isinstance(hwy, list):
            hwy = hwy[0]
        speed = DEFAULT_HIGHWAY_SPEEDS.get(str(hwy), FALLBACK_SPEED_KPH)
        if "maxspeed" in data and data["maxspeed"]:
            ms_raw = str(data["maxspeed"]).split()[0]
            try:
                speed = float(ms_raw)
            except ValueError:
                pass
        data["speed_kph"] = float(speed)
        speed_mps = data["speed_kph"] * (1000.0 / 3600.0)
        data["travel_time"] = round(data["length"] / speed_mps, 2)

    # Cache as GraphML
    g_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, filepath=str(g_path))
    return graph


def load_graphml_network(graphml_path: Path | str = DEFAULT_GRAPHML_PATH) -> nx.MultiDiGraph:
    """Load cached GraphML driving network and ensure numeric typing."""
    graph = ox.load_graphml(str(graphml_path))
    for _, _, _, data in graph.edges(keys=True, data=True):
        data["length"] = float(data.get("length", 1.0))
        data["speed_kph"] = float(data.get("speed_kph", FALLBACK_SPEED_KPH))
        data["travel_time"] = float(data.get("travel_time", data["length"] / 10.0))
    for _, ndata in graph.nodes(data=True):
        ndata["y"] = float(ndata["y"])
        ndata["x"] = float(ndata["x"])
    return graph


def get_nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> Tuple[int, float]:
    """Snap a coordinate (lat, lon) to the nearest road network node.
    
    Returns (node_id, distance_meters).
    """
    if "crs" in graph.graph:
        node_id, dist = ox.distance.nearest_nodes(graph, X=lon, Y=lat, return_dist=True)
        return int(node_id), float(dist)

    # Robust fallback for graphs without CRS metadata
    best_node = None
    best_dist = float("inf")
    for n, data in graph.nodes(data=True):
        n_lat = data.get("y", data.get("lat"))
        n_lon = data.get("x", data.get("lon"))
        if n_lat is not None and n_lon is not None:
            d = haversine_distance_m(lat, lon, float(n_lat), float(n_lon))
            if d < best_dist:
                best_dist = d
                best_node = n
    if best_node is None:
        raise ValueError("Graph has no valid nodes to snap to.")
    return best_node, best_dist


def get_graph_stats(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """Return summary statistics of the driving network graph."""
    total_length_m = sum(float(data.get("length", 0.0)) for _, _, _, data in graph.edges(keys=True, data=True))
    lats = [d["y"] for _, d in graph.nodes(data=True)]
    lons = [d["x"] for _, d in graph.nodes(data=True)]
    return {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "total_road_length_km": round(total_length_m / 1000.0, 2),
        "bounds": {
            "min_lat": min(lats) if lats else 0.0,
            "max_lat": max(lats) if lats else 0.0,
            "min_lon": min(lons) if lons else 0.0,
            "max_lon": max(lons) if lons else 0.0,
        },
        "attribution": MAP_ATTRIBUTION,
        "disclaimer": SPEED_DISCLAIMER,
    }
