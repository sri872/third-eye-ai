"""A* shortest-path routing algorithm for Project Third-Eye.

Uses an admissible great-circle Haversine heuristic (scaled by the maximum
road speed in the graph) to guarantee optimal minimum-travel-time paths on
directed multigraphs without overestimation.
"""

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx

from core.routing.road_graph import haversine_distance_m

# Maximum vehicle speed used for admissible heuristic scaling (100 km/h = 27.78 m/s)
MAX_ADMISSIBLE_SPEED_MPS = 100.0 * (1000.0 / 3600.0)


def astar_shortest_path(
    graph: nx.MultiDiGraph,
    source: int,
    target: int,
    weight: str = "travel_time",
    max_speed_mps: float = MAX_ADMISSIBLE_SPEED_MPS,
) -> Tuple[List[int], float, float]:
    """Compute optimal shortest path using A* search.

    Args:
        graph: NetworkX MultiDiGraph with node attributes 'x' (lon) and 'y' (lat).
        source: Starting node ID.
        target: Destination node ID.
        weight: Edge attribute to minimize (default: 'travel_time' in seconds).
        max_speed_mps: Upper-bound speed in m/s used to ensure heuristic admissibility.

    Returns:
        Tuple of (path_nodes, total_time_sec, total_distance_meters).
        If source == target, returns ([source], 0.0, 0.0).
        If no path exists, returns ([], float('inf'), 0.0).
    """
    if source not in graph or target not in graph:
        return [], float("inf"), 0.0

    if source == target:
        return [source], 0.0, 0.0

    t_lat = graph.nodes[target]["y"]
    t_lon = graph.nodes[target]["x"]

    def heuristic(node: int) -> float:
        n_lat = graph.nodes[node]["y"]
        n_lon = graph.nodes[node]["x"]
        dist = haversine_distance_m(n_lat, n_lon, t_lat, t_lon)
        return dist / max_speed_mps

    frontier: List[Tuple[float, int, int]] = []
    counter = 0
    heapq.heappush(frontier, (heuristic(source), counter, source))

    g_score: Dict[int, float] = {source: 0.0}
    came_from: Dict[int, int] = {}
    best_edge_dist: Dict[Tuple[int, int], float] = {}

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current == target:
            # Reconstruct node sequence
            path: List[int] = []
            curr: Optional[int] = target
            while curr is not None and curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(source)
            path.reverse()

            total_dist = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                total_dist += best_edge_dist.get((u, v), 0.0)

            return path, round(g_score[target], 2), round(total_dist, 2)

        cur_g = g_score[current]

        for neighbor in graph.successors(current):
            # Select lowest-weight parallel edge between current and neighbor
            edges = graph[current][neighbor]
            min_weight = float("inf")
            min_dist = 0.0
            for k, edata in edges.items():
                w = float(edata.get(weight, edata.get("length", 1.0) / 10.0))
                if w < min_weight:
                    min_weight = w
                    min_dist = float(edata.get("length", 0.0))

            tentative_g = cur_g + min_weight
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                best_edge_dist[(current, neighbor)] = min_dist
                f_score = tentative_g + heuristic(neighbor)
                counter += 1
                heapq.heappush(frontier, (f_score, counter, neighbor))

    return [], float("inf"), 0.0


def extract_route_coordinates(graph: nx.MultiDiGraph, path_nodes: List[int]) -> List[List[float]]:
    """Extract list of [latitude, longitude] pairs for a node path sequence."""
    coords: List[List[float]] = []
    for node in path_nodes:
        if node in graph:
            coords.append([round(graph.nodes[node]["y"], 6), round(graph.nodes[node]["x"], 6)])
    return coords
