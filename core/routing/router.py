"""Road Network Router using NetworkX and GraphML.

Calculates deterministic road routes, distances, and estimated travel times
between SOS incident coordinates and target hospitals.
"""

import os
import math
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx


class RoadNetworkRouter:
    """Computes shortest path routes on the road graph network."""

    def __init__(self, graph_path: Optional[str] = None):
        if graph_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            graph_path = os.path.abspath(
                os.path.join(curr_dir, "..", "..", "data", "phagwara_drive_network.graphml")
            )
        self.graph_path = graph_path
        self.graph: Optional[nx.MultiDiGraph] = None
        self._load_graph()

    def _load_graph(self) -> None:
        if os.path.exists(self.graph_path):
            try:
                self.graph = nx.read_graphml(self.graph_path)
                # Ensure all edge length attributes are floats for Dijkstra routing
                for u, v, k, data in self.graph.edges(keys=True, data=True):
                    if "length" in data:
                        try:
                            data["length"] = float(data["length"])
                        except (ValueError, TypeError):
                            data["length"] = 100.0
                    else:
                        data["length"] = 100.0
            except Exception:
                self.graph = None

    def find_nearest_node(self, lat: float, lon: float) -> Optional[str]:
        """Find the nearest graph node to given coordinates using Euclidean approximation."""
        if self.graph is None:
            return None

        best_node = None
        min_dist_sq = float("inf")

        for node_id, data in self.graph.nodes(data=True):
            try:
                ny = float(data.get("y", 0.0))
                nx_coord = float(data.get("x", 0.0))
                dist_sq = (lat - ny) ** 2 + (lon - nx_coord) ** 2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_node = node_id
            except (ValueError, TypeError):
                continue

        return best_node

    def calculate_route(
        self,
        start_lat: float,
        start_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> Dict[str, Any]:
        """Calculate road network route between start and destination coordinates.

        The route ALWAYS starts at (start_lat, start_lon) representing the SOS location.
        """
        # Google Maps navigation external link
        gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={start_lat:.5f},{start_lon:.5f}&destination={dest_lat:.5f},{dest_lon:.5f}"

        # 1. Attempt Road Network Shortest Path
        if self.graph is not None:
            start_node = self.find_nearest_node(start_lat, start_lon)
            dest_node = self.find_nearest_node(dest_lat, dest_lon)

            if start_node and dest_node and start_node != dest_node:
                try:
                    # Dijkstra shortest path using edge 'length' (meters)
                    path_nodes = nx.shortest_path(
                        self.graph,
                        source=start_node,
                        target=dest_node,
                        weight="length",
                    )

                    # Accumulate road distance along edges
                    total_meters = 0.0
                    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                        edge_data = self.graph.get_edge_data(u, v)
                        if edge_data:
                            # MultiDiGraph returns dict of parallel edges
                            first_edge = list(edge_data.values())[0] if isinstance(edge_data, dict) else edge_data
                            total_meters += float(first_edge.get("length", 100.0))

                    road_km = max(0.1, round(total_meters / 1000.0, 2))
                    # Estimate travel time assuming 40 km/h average speed in urban/suburban corridor
                    travel_time_min = max(1.0, round((road_km / 40.0) * 60.0, 1))

                    # Extract path coordinates
                    route_coords: List[Tuple[float, float]] = [(start_lat, start_lon)]
                    for n in path_nodes:
                        ndata = self.graph.nodes[n]
                        route_coords.append((float(ndata["y"]), float(ndata["x"])))
                    route_coords.append((dest_lat, dest_lon))

                    return {
                        "status": "ROAD_NETWORK_ROUTE_FOUND",
                        "distance_km": road_km,
                        "estimated_time_minutes": travel_time_min,
                        "route_geometry": route_coords,
                        "node_count": len(path_nodes),
                        "google_maps_url": gmaps_url,
                        "is_road_network": True,
                    }
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

        # 2. Fallback: Straight-line Haversine Distance (with explicit disclaimer)
        haversine_km = self.haversine_distance_km(start_lat, start_lon, dest_lat, dest_lon)
        # Apply winding factor 1.35 and 40 km/h speed
        estimated_road_km = round(haversine_km * 1.35, 2)
        travel_time_min = max(1.0, round((estimated_road_km / 40.0) * 60.0, 1))

        return {
            "status": "ROUTING_FALLBACK_HAVERSINE",
            "distance_km": estimated_road_km,
            "estimated_time_minutes": travel_time_min,
            "route_geometry": [(start_lat, start_lon), (dest_lat, dest_lon)],
            "node_count": 2,
            "google_maps_url": gmaps_url,
            "is_road_network": False,
            "notes": "Direct road network path not connected; straight-line estimate with 1.35 winding factor used.",
        }

    @staticmethod
    def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great-circle distance between two points in kilometers."""
        r = 6371.0  # Earth's radius in kilometers
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2.0) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return round(r * c, 3)
