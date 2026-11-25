"""Geometric feature extraction from object detections for Project Third-Eye.

Extracts real physical and spatial indicators from object bounding boxes:
- Vehicle count and person count
- Maximum and mean pairwise vehicle box IoU
- Overlap ratio between vehicle pairs
- Count of heavily overlapping pairs
- Vehicle-person bounding box intersection count
- Aspect-ratio outliers (indicating overturned or deformed vehicles)
- Scene crowd density
"""

import itertools
from typing import Any, Dict, List, Tuple

from core.schemas import Detection

VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle"}


def compute_iou(box_a: List[float], box_b: List[float]) -> Tuple[float, float]:
    """Compute Intersection over Union (IoU) and intersection over min-area ratio.

    Args:
        box_a: [xmin, ymin, xmax, ymax]
        box_b: [xmin, ymin, xmax, ymax]

    Returns:
        (iou, overlap_ratio_min): IoU in [0, 1], and intersection / min(area_a, area_b) in [0, 1].
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0, 0.0

    area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    union_area = area_a + area_b - inter_area
    iou = inter_area / union_area if union_area > 0.0 else 0.0
    overlap_min = inter_area / min(area_a, area_b)

    return round(float(iou), 4), round(float(overlap_min), 4)


def extract_accident_features(
    detections: List[Detection],
    image_width: int = 1920,
    image_height: int = 1080,
    heavy_iou_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Extract geometric accident features from object detections.

    Args:
        detections: List of Detection schema objects.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        heavy_iou_threshold: IoU threshold for heavily overlapping vehicles.

    Returns:
        Dict containing real computed geometric features.
    """
    vehicles = [d for d in detections if d.label in VEHICLE_LABELS]
    persons = [d for d in detections if d.label == "person"]

    v_count = len(vehicles)
    p_count = len(persons)

    max_vehicle_iou = 0.0
    mean_vehicle_iou = 0.0
    max_overlap_min = 0.0
    heavily_overlapping_pairs = 0
    overlapping_pair_count = 0
    total_vehicle_pairs = 0

    # 1. Pairwise vehicle-vehicle IoU analysis
    if v_count >= 2:
        pairs = list(itertools.combinations(vehicles, 2))
        total_vehicle_pairs = len(pairs)
        ious = []
        overlap_mins = []

        for v1, v2 in pairs:
            iou, o_min = compute_iou(v1.bbox, v2.bbox)
            ious.append(iou)
            overlap_mins.append(o_min)

            if iou > 0.0:
                overlapping_pair_count += 1
            if iou >= heavy_iou_threshold or o_min >= 0.25:
                heavily_overlapping_pairs += 1

        if ious:
            max_vehicle_iou = max(ious)
            mean_vehicle_iou = round(sum(ious) / len(ious), 4)
            max_overlap_min = max(overlap_mins)

    overlap_ratio = round(overlapping_pair_count / total_vehicle_pairs, 4) if total_vehicle_pairs > 0 else 0.0

    # 2. Vehicle-person collision/overlap analysis
    vehicle_person_overlap_count = 0
    max_person_vehicle_iou = 0.0
    for p in persons:
        for v in vehicles:
            iou, o_min = compute_iou(p.bbox, v.bbox)
            if iou > 0.0 or o_min > 0.10:
                vehicle_person_overlap_count += 1
                if iou > max_person_vehicle_iou:
                    max_person_vehicle_iou = iou

    # 3. Vehicle bounding box aspect ratio outliers
    # Normal upright vehicle w/h typically ranges between 0.8 and 2.5.
    # Outliers (< 0.5 or > 3.0) often indicate rollover, jackknifed trucks, or severe impact deformation.
    aspect_ratio_outliers = 0
    for v in vehicles:
        w = max(1.0, v.bbox[2] - v.bbox[0])
        h = max(1.0, v.bbox[3] - v.bbox[1])
        ratio = w / h
        if ratio > 2.8 or ratio < 0.6:
            aspect_ratio_outliers += 1

    # 4. Crowd density relative to image frame
    img_area = max(1.0, float(image_width * image_height))
    # Normalized persons per megapixel
    crowd_density = round((p_count / img_area) * 1_000_000.0, 4)

    return {
        "vehicle_count": v_count,
        "person_count": p_count,
        "total_vehicle_pairs": total_vehicle_pairs,
        "max_vehicle_iou": max_vehicle_iou,
        "mean_vehicle_iou": mean_vehicle_iou,
        "max_overlap_min": max_overlap_min,
        "overlap_ratio": overlap_ratio,
        "heavily_overlapping_pairs": heavily_overlapping_pairs,
        "vehicle_person_overlap_count": vehicle_person_overlap_count,
        "max_person_vehicle_iou": max_person_vehicle_iou,
        "aspect_ratio_outliers": aspect_ratio_outliers,
        "crowd_density": crowd_density,
    }
