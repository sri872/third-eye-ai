"""Classical Vehicle Candidate Region Localization.

Uses deterministic morphological operations, edge density, contour analysis,
and geometric heuristics to detect vehicle candidate regions without any
neural networks, ML classifiers, or learned detection models.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


def localize_vehicle_candidates(
    image: np.ndarray,
    min_area_ratio: float = 0.015,
    max_area_ratio: float = 0.85,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 3.8,
) -> List[Dict[str, Any]]:
    """Localize candidate vehicle regions using deterministic classical image processing.

    Notice: This produces geometric candidate proposals based on visual boundaries and
    edge concentration. It is NOT a learned classifier and does NOT guarantee that a
    candidate region represents a verified vehicle.

    Args:
        image: Input BGR image.
        min_area_ratio: Minimum bounding-box area relative to total image area.
        max_area_ratio: Maximum bounding-box area relative to total image area.
        min_aspect_ratio: Minimum width/height ratio.
        max_aspect_ratio: Maximum width/height ratio.

    Returns:
        List of candidate region dictionaries.
    """
    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]
    img_area = float(h * w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. Bilateral filter to smooth texture while preserving structural contours
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2. Canny edge detection
    edges = cv2.Canny(blurred, 40, 130)

    # 3. Morphological closing with horizontal structuring element
    # Vehicles have prominent horizontal boundaries (bumpers, rooflines, shadows)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
    closed_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)

    # 4. Morphological dilation to bridge small gaps between components
    kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(closed_h, kernel_sq, iterations=2)

    # 5. Find external contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Dict[str, Any]] = []

    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        box_area = float(bw * bh)
        area_ratio = box_area / img_area

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        # Evaluate edge density within candidate box
        crop_edges = edges[by : by + bh, bx : bx + bw]
        edge_density = float(cv2.countNonZero(crop_edges)) / box_area if box_area > 0 else 0.0

        if edge_density < 0.02:
            continue

        # Geometric type hint based on aspect ratio and vertical position
        if aspect_ratio > 1.8:
            type_hint = "SEDAN_OR_COMMERCIAL"
        elif aspect_ratio > 1.1:
            type_hint = "CAR_OR_SUV"
        else:
            type_hint = "TWO_WHEELER_OR_UPRIGHT"

        # Heuristic confidence score (0.0 to 1.0) derived strictly from geometry
        # Vehicles typically occupy ~5% to 50% area and have aspect ratio ~1.0 to 2.2
        area_score = 1.0 - min(1.0, abs(area_ratio - 0.20) / 0.35)
        aspect_score = 1.0 - min(1.0, abs(aspect_ratio - 1.6) / 1.5)
        edge_score = min(1.0, edge_density * 8.0)
        heuristic_conf = round(0.4 * area_score + 0.3 * aspect_score + 0.3 * edge_score, 3)

        candidates.append(
            {
                "bbox": [int(bx), int(by), int(bw), int(bh)],
                "aspect_ratio": round(aspect_ratio, 2),
                "area_ratio": round(area_ratio, 3),
                "edge_density": round(edge_density, 3),
                "heuristic_confidence": heuristic_conf,
                "type_hint": type_hint,
                "label": "GEOMETRIC CANDIDATE (CLASSICAL CV HEURISTIC)",
                "method": "CLASSICAL_CONTOUR_MORPHOLOGY",
            }
        )

    # 6. Non-maximum suppression / overlap deduplication
    deduped = _suppress_overlapping_boxes(candidates, iou_thresh=0.45)
    # Sort candidates by heuristic confidence descending
    deduped.sort(key=lambda c: c["heuristic_confidence"], reverse=True)

    return deduped


def _suppress_overlapping_boxes(
    boxes: List[Dict[str, Any]], iou_thresh: float = 0.45
) -> List[Dict[str, Any]]:
    """Remove redundant overlapping candidate boxes."""
    if not boxes:
        return []

    # Sort by heuristic confidence descending
    sorted_boxes = sorted(boxes, key=lambda b: b["heuristic_confidence"], reverse=True)
    kept: List[Dict[str, Any]] = []

    for b in sorted_boxes:
        bx1, by1, bw1, bh1 = b["bbox"]
        keep = True
        for k in kept:
            kx1, ky1, kw1, kh1 = k["bbox"]
            # Intersection
            ix1 = max(bx1, kx1)
            iy1 = max(by1, ky1)
            ix2 = min(bx1 + bw1, kx1 + kw1)
            iy2 = min(by1 + bh1, ky1 + kh1)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = float(iw * ih)
            union = float(bw1 * bh1 + kw1 * kh1 - inter)
            iou = inter / union if union > 0 else 0.0
            if iou > iou_thresh:
                keep = False
                break
        if keep:
            kept.append(b)

    return kept
