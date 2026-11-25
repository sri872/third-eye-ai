"""Deterministic ambulance movement simulator for Project Third-Eye.

Simulates vehicle movement along discrete route waypoints at constant or
edge-proportional time intervals. Explicitly marked as SIMULATED.
"""

import math
from typing import Any, Dict, List, Tuple

from core.routing.road_graph import haversine_distance_m


def interpolate_coordinate(
    p1: Tuple[float, float], p2: Tuple[float, float], fraction: float
) -> Tuple[float, float]:
    """Linearly interpolate between two (lat, lon) coordinates."""
    lat = p1[0] + (p2[0] - p1[0]) * fraction
    lon = p1[1] + (p2[1] - p1[1]) * fraction
    return lat, lon


def simulate_leg_trajectory(
    waypoints: List[List[float]],
    total_time_seconds: float,
    step_seconds: float = 10.0,
    leg_name: str = "leg1_ambulance_to_incident",
) -> List[Dict[str, Any]]:
    """Simulate vehicle progress along a sequence of [lat, lon] waypoints.

    Args:
        waypoints: List of [latitude, longitude] pairs.
        total_time_seconds: Total duration of the leg in seconds.
        step_seconds: Sampling time delta in seconds.
        leg_name: Identifier for the route leg.

    Returns:
        List of simulation snapshots, each labeled with data_source: SIMULATED.
    """
    if not waypoints or len(waypoints) < 2 or total_time_seconds <= 0:
        if waypoints:
            pt = waypoints[0]
            return [{
                "step": 0,
                "elapsed_sec": 0.0,
                "latitude": round(pt[0], 6),
                "longitude": round(pt[1], 6),
                "progress_pct": 100.0,
                "leg": leg_name,
                "data_source": "SIMULATED",
            }]
        return []

    # Calculate cumulative distance along polyline segments
    segment_lengths: List[float] = []
    cumulative_dists: List[float] = [0.0]
    total_dist_m = 0.0

    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        dist = haversine_distance_m(p1[0], p1[1], p2[0], p2[1])
        segment_lengths.append(dist)
        total_dist_m += dist
        cumulative_dists.append(total_dist_m)

    if total_dist_m <= 0:
        total_dist_m = 1.0

    # Generate discrete time snapshots
    snapshots: List[Dict[str, Any]] = []
    current_time = 0.0
    step_idx = 0

    while current_time < total_time_seconds:
        progress_frac = current_time / total_time_seconds
        target_dist = progress_frac * total_dist_m

        # Locate segment containing target_dist
        seg_idx = 0
        for i in range(len(segment_lengths)):
            if cumulative_dists[i + 1] >= target_dist:
                seg_idx = i
                break
            seg_idx = i

        seg_start_dist = cumulative_dists[seg_idx]
        seg_len = max(segment_lengths[seg_idx], 0.001)
        seg_frac = min(max((target_dist - seg_start_dist) / seg_len, 0.0), 1.0)

        p1 = (waypoints[seg_idx][0], waypoints[seg_idx][1])
        p2 = (waypoints[seg_idx + 1][0], waypoints[seg_idx + 1][1])
        lat, lon = interpolate_coordinate(p1, p2, seg_frac)

        snapshots.append({
            "step": step_idx,
            "elapsed_sec": round(current_time, 1),
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "progress_pct": round(progress_frac * 100.0, 1),
            "leg": leg_name,
            "data_source": "SIMULATED",
        })

        step_idx += 1
        current_time += step_seconds

    # Final destination point
    last_pt = waypoints[-1]
    snapshots.append({
        "step": step_idx,
        "elapsed_sec": round(total_time_seconds, 1),
        "latitude": round(last_pt[0], 6),
        "longitude": round(last_pt[1], 6),
        "progress_pct": 100.0,
        "leg": leg_name,
        "data_source": "SIMULATED",
    })

    return snapshots


def simulate_dispatch_trajectory(
    leg1: List[List[float]],
    leg1_time_seconds: float,
    leg2: List[List[float]],
    leg2_time_seconds: float,
    step_seconds: float = 15.0,
) -> List[Dict[str, Any]]:
    """Simulate ambulance-to-incident and incident-to-hospital movement together.

    Both legs come from the stored A* route geometry and the stored full-precision
    travel times. The second leg's clock starts when the first leg ends; the shared
    incident/junction snapshot is kept only once. No live GPS or traffic is used.
    """
    first = simulate_leg_trajectory(
        leg1,
        leg1_time_seconds,
        step_seconds=step_seconds,
        leg_name="leg1_ambulance_to_incident",
    )
    second = simulate_leg_trajectory(
        leg2,
        leg2_time_seconds,
        step_seconds=step_seconds,
        leg_name="leg2_incident_to_hospital",
    )
    if not first and not second:
        return []

    offset = float(first[-1]["elapsed_sec"]) if first else 0.0
    total_seconds = float(leg1_time_seconds) + float(leg2_time_seconds)
    combined: List[Dict[str, Any]] = []
    combined.extend(dict(snapshot) for snapshot in first)
    for snapshot in second[1:]:
        entry = dict(snapshot)
        entry["elapsed_sec"] = round(offset + float(entry["elapsed_sec"]), 1)
        combined.append(entry)

    for step, entry in enumerate(combined):
        entry["step"] = step
        elapsed = float(entry["elapsed_sec"])
        entry["progress_pct"] = (
            round(100.0 * elapsed / total_seconds, 1) if total_seconds > 0 else 100.0
        )
    return combined
