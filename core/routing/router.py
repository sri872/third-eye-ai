"""Directed road-network routing using the saved OpenStreetMap GraphML graph.

Only an actual connected path in the saved graph is returned as a road route.
There is deliberately no straight-line-to-road-distance fallback.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "phagwara_drive_network.graphml"
MAX_SNAP_DISTANCE_M = 1800.0
CONNECTOR_SPEED_KPH = 20.0
DEFAULT_ROAD_SPEED_KPH = 30.0


class RouteUnavailableError(RuntimeError):
    """A genuine road-network route cannot be calculated for these coordinates."""


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


class RoadNetworkRouter:
    """Computes minimum-travel-time routes over a directed OSM road graph."""

    def __init__(
        self,
        graph_path: Optional[str] = None,
        graph: Optional[nx.Graph] = None,
        max_snap_distance_m: float = MAX_SNAP_DISTANCE_M,
    ):
        self.graph_path = str(graph_path or DEFAULT_GRAPH_PATH)
        self.max_snap_distance_m = float(max_snap_distance_m)
        self.graph: Optional[nx.MultiDiGraph] = None
        if graph is not None:
            self.graph = self._normalize_graph(graph)
        else:
            self._load_graph()

    @staticmethod
    def _edge_metrics(attributes: dict[str, Any]) -> tuple[float, float]:
        try:
            distance = float(attributes.get("length", 0.0))
        except (TypeError, ValueError):
            distance = 0.0
        if not math.isfinite(distance) or distance <= 0:
            raise RouteUnavailableError("Road graph contains an invalid edge length.")
        try:
            seconds = float(attributes.get("travel_time", 0.0))
        except (TypeError, ValueError):
            seconds = 0.0
        if not math.isfinite(seconds) or seconds <= 0:
            try:
                speed = float(attributes.get("speed_kph", DEFAULT_ROAD_SPEED_KPH))
            except (TypeError, ValueError):
                speed = DEFAULT_ROAD_SPEED_KPH
            if not math.isfinite(speed) or speed <= 0:
                speed = DEFAULT_ROAD_SPEED_KPH
            seconds = distance / (speed * 1000.0 / 3600.0)
        return distance, seconds

    @classmethod
    def _normalize_graph(cls, graph: nx.Graph) -> nx.MultiDiGraph:
        if not graph.is_directed():
            raise RouteUnavailableError(
                "The saved road graph is not directed; one-way road restrictions cannot be respected."
            )
        if graph.is_multigraph():
            normalized = nx.MultiDiGraph(graph)
        else:
            normalized = nx.MultiDiGraph(graph)
        valid_nodes = []
        for node, attrs in normalized.nodes(data=True):
            try:
                attrs["x"] = float(attrs["x"])
                attrs["y"] = float(attrs["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(attrs["x"]) or not math.isfinite(attrs["y"]):
                continue
            valid_nodes.append(node)
        normalized.remove_nodes_from(set(normalized.nodes) - set(valid_nodes))
        if len(normalized) < 2:
            raise RouteUnavailableError("Road graph contains fewer than two coordinate-tagged nodes.")
        for _, _, _, attrs in normalized.edges(keys=True, data=True):
            cls._edge_metrics(attrs)  # validate and normalize numeric attributes
            distance, seconds = cls._edge_metrics(attrs)
            attrs["length"] = distance
            attrs["travel_time"] = seconds
        return normalized

    def _load_graph(self) -> None:
        path = Path(self.graph_path)
        if not path.is_file():
            raise RouteUnavailableError(f"Saved road graph is missing: {path}")
        try:
            graph = nx.read_graphml(
                path,
                node_type=str,
                force_multigraph=True,
            )
            self.graph = self._normalize_graph(graph)
        except RouteUnavailableError:
            raise
        except Exception as exc:
            raise RouteUnavailableError(f"Saved road graph could not be loaded: {exc}") from exc

    def find_nearest_node(self, lat: float, lon: float) -> tuple[str, float]:
        """Return nearest graph node and true great-circle snap distance in metres."""
        if self.graph is None or not self.graph:
            raise RouteUnavailableError("Road graph is not loaded.")
        best_node = None
        best_distance = float("inf")
        for node, attrs in self.graph.nodes(data=True):
            distance = haversine_distance_m(lat, lon, attrs["y"], attrs["x"])
            if distance < best_distance:
                best_node, best_distance = node, distance
        if best_node is None:
            raise RouteUnavailableError("Road graph has no coordinate-tagged nodes.")
        if best_distance > self.max_snap_distance_m:
            raise RouteUnavailableError(
                f"Coordinate is outside saved road-graph coverage: nearest road node is "
                f"{best_distance:.0f} m away (limit {self.max_snap_distance_m:.0f} m)."
            )
        return best_node, best_distance

    def calculate_route(
        self,
        start_lat: float,
        start_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> dict[str, Any]:
        """Return a road-network path, distance and graph-based travel-time estimate."""
        values = (start_lat, start_lon, dest_lat, dest_lon)
        try:
            start_lat, start_lon, dest_lat, dest_lon = (float(v) for v in values)
        except (TypeError, ValueError):
            raise ValueError("Route coordinates must be numeric.") from None
        if not all(math.isfinite(v) for v in (start_lat, start_lon, dest_lat, dest_lon)):
            raise ValueError("Route coordinates must be finite.")
        if not (-90 <= start_lat <= 90 and -90 <= dest_lat <= 90):
            raise ValueError("Route latitude is outside geographic limits.")
        if not (-180 <= start_lon <= 180 and -180 <= dest_lon <= 180):
            raise ValueError("Route longitude is outside geographic limits.")
        if self.graph is None:
            raise RouteUnavailableError("Saved road graph is not loaded.")

        start_node, start_snap_m = self.find_nearest_node(start_lat, start_lon)
        dest_node, dest_snap_m = self.find_nearest_node(dest_lat, dest_lon)

        def edge_time(_u, _v, edge_bundle):
            # NetworkX passes a key->attributes mapping for MultiDiGraph weights.
            return min(self._edge_metrics(attrs)[1] for attrs in edge_bundle.values())

        try:
            nodes = nx.shortest_path(
                self.graph,
                source=start_node,
                target=dest_node,
                weight=edge_time,
                method="dijkstra",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            raise RouteUnavailableError(
                "No directed road path connects the SOS location and this hospital in the saved graph."
            ) from exc

        graph_distance_m = 0.0
        graph_time_seconds = 0.0
        for u, v in zip(nodes, nodes[1:]):
            edge_bundle = self.graph.get_edge_data(u, v)
            if not edge_bundle:
                raise RouteUnavailableError("Road path contains a missing graph edge.")
            distance, seconds = min(
                (self._edge_metrics(attrs) for attrs in edge_bundle.values()),
                key=lambda metrics: metrics[1],
            )
            graph_distance_m += distance
            graph_time_seconds += seconds

        connector_distance_m = start_snap_m + dest_snap_m
        connector_speed_mps = CONNECTOR_SPEED_KPH * 1000.0 / 3600.0
        connector_seconds = connector_distance_m / connector_speed_mps
        coords = [(start_lat, start_lon)]
        coords.extend(
            (self.graph.nodes[node]["y"], self.graph.nodes[node]["x"])
            for node in nodes
        )
        coords.append((dest_lat, dest_lon))
        # De-duplicate repeated adjacent coordinates without changing route order.
        route_geometry = []
        for point in coords:
            if not route_geometry or point != route_geometry[-1]:
                route_geometry.append(point)

        distance_m = graph_distance_m + connector_distance_m
        total_seconds = graph_time_seconds + connector_seconds
        google_maps_url = (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={quote(f'{start_lat:.6f},{start_lon:.6f}')}"
            f"&destination={quote(f'{dest_lat:.6f},{dest_lon:.6f}')}"
            "&travelmode=driving"
        )
        return {
            "status": "ROAD_NETWORK_ROUTE_FOUND",
            "distance_km": round(distance_m / 1000.0, 2),
            "estimated_time_minutes": round(total_seconds / 60.0, 1),
            "road_graph_distance_km": round(graph_distance_m / 1000.0, 2),
            "off_network_connector_m": round(connector_distance_m, 0),
            "estimated_connector_minutes": round(connector_seconds / 60.0, 1),
            "route_geometry": route_geometry,
            "node_count": len(nodes),
            "start_snap_m": round(start_snap_m, 0),
            "destination_snap_m": round(dest_snap_m, 0),
            "google_maps_url": google_maps_url,
            "is_road_network": True,
            "route_source": "SAVED_OPENSTREETMAP_DIRECTED_GRAPH",
            "travel_time_note": (
                "Estimated from static travel_time values on the saved OSM graph plus "
                f"straight-line endpoint connectors assumed at {CONNECTOR_SPEED_KPH:.0f} km/h. "
                "No live traffic or ambulance ETA is available."
            ),
        }

    @staticmethod
    def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance helper; never returned as a driving route."""
        return haversine_distance_m(lat1, lon1, lat2, lon2) / 1000.0
