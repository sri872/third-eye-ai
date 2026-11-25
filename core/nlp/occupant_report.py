"""Occupant and pedestrian reporting module for Project Third-Eye (Phase 4).

Generates OccupantReport records from Phase 1 person detections.
Rules:
- Visible people only: occupants inside vehicles cannot be reliably counted from a single image.
- Every record explicitly contains the limitation note.
- Never guesses age, gender, injury, or health state: injury_severity is "not determined", consciousness is "unknown".
- Computes persons_visible_count and persons_near_vehicles_count.
- Assigns each person to the vehicle it overlaps most (by vehicle_id), or None if disjoint.
- Confidence comes straight from the object detector; never fabricated.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import uuid

from core.schemas import Detection, OccupantReport, VehicleReport

logger = logging.getLogger("third_eye.nlp.occupant_report")

VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle"}
DEFAULT_LIMITATION_NOTE = (
    "Visible people only. Occupants inside vehicles cannot be reliably counted from a single image."
)


def compute_box_intersection_ratio(box_a: List[float], box_b: List[float]) -> float:
    """Compute intersection area divided by area of box_a (share of person inside/touching vehicle)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    return float(inter_area / area_a)


def build_occupant_reports(
    detections: List[Detection],
    vehicle_reports: Optional[List[VehicleReport]] = None,
    timestamp: Optional[datetime] = None,
) -> Tuple[List[OccupantReport], Dict[str, int]]:
    """Build OccupantReport schemas from scene detections.

    Args:
        detections: All Detection objects from Phase 1 Faster R-CNN.
        vehicle_reports: Optional list of VehicleReports corresponding to vehicle detections.
        timestamp: Optional capture timestamp.

    Returns:
        Tuple of (list of OccupantReport schemas, summary count dict).
    """
    ts = timestamp or datetime.now(timezone.utc)

    # Separate person and vehicle detections
    person_dets = [d for d in detections if d.label == "person"]
    vehicle_dets = [d for d in detections if d.label in VEHICLE_LABELS]

    # Map each vehicle detection to a vehicle_id
    vehicle_id_map: List[Tuple[str, List[float]]] = []
    for idx, v_det in enumerate(vehicle_dets):
        if vehicle_reports and idx < len(vehicle_reports):
            vid = vehicle_reports[idx].vehicle_id
        else:
            vid = f"veh-{idx+1:03d}"
        vehicle_id_map.append((vid, v_det.bbox))

    persons_visible_count = len(person_dets)
    persons_near_vehicles_count = 0
    occupant_reports: List[OccupantReport] = []

    for idx, p_det in enumerate(person_dets, 1):
        best_vehicle_id: Optional[str] = None
        best_overlap: float = 0.0

        for vid, v_box in vehicle_id_map:
            overlap = compute_box_intersection_ratio(p_det.bbox, v_box)
            if overlap > best_overlap:
                best_overlap = overlap
                best_vehicle_id = vid

        # Intersection exists if overlap > 0
        if best_overlap > 0.0:
            persons_near_vehicles_count += 1
        else:
            best_vehicle_id = None
            best_overlap = 0.0

        occ_id = f"occ-{idx:03d}-{uuid.uuid4().hex[:6]}"
        occ_report = OccupantReport(
            occupant_id=occ_id,
            vehicle_id=best_vehicle_id,
            bbox=[round(float(c), 2) for c in p_det.bbox],
            estimated_count=1,
            injury_severity="not determined",  # Strictly never guessed from image
            consciousness="unknown",          # Strictly never guessed from image
            confidence=p_det.confidence,      # Real detector confidence straight from model
            overlap_ratio=round(best_overlap, 4) if best_overlap > 0 else 0.0,
            persons_visible_count=persons_visible_count,
            persons_near_vehicles_count=0,  # Will update after full loop
            limitation_note=DEFAULT_LIMITATION_NOTE,
            timestamp=ts,
            data_source=p_det.data_source,
        )
        occupant_reports.append(occ_report)

    # Update persons_near_vehicles_count on all records
    for rep in occupant_reports:
        rep.persons_near_vehicles_count = persons_near_vehicles_count

    counts_summary = {
        "persons_visible_count": persons_visible_count,
        "persons_near_vehicles_count": persons_near_vehicles_count,
    }

    return occupant_reports, counts_summary
