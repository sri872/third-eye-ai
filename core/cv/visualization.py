"""Classical Computer Vision Visualization Module.

Annotates candidate vehicle regions and license plate regions on images
using standard OpenCV drawing primitives with clear diagnostic labels.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


def draw_vehicle_candidates(
    image: np.ndarray,
    candidates: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (255, 120, 0),  # Cyan-blue in BGR
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw vehicle candidate bounding boxes and geometric labels on an image."""
    annotated = image.copy()
    h_img, w_img = annotated.shape[:2]

    for idx, cand in enumerate(candidates, start=1):
        x, y, w, h = cand["bbox"]
        label = f"Region #{idx}: {cand.get('type_hint', 'CANDIDATE')} (AR: {cand.get('aspect_ratio', 0.0)})"

        # Draw bounding rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, thickness)

        # Label background banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

        banner_y1 = max(0, y - th - 6)
        banner_y2 = y
        cv2.rectangle(annotated, (x, banner_y1), (x + tw + 6, banner_y2), box_color, -1)
        cv2.putText(
            annotated,
            label,
            (x + 3, y - 4),
            font,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_plate_candidates(
    image: np.ndarray,
    candidates: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (0, 220, 0),  # Bright green in BGR
    text_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw license plate candidate bounding boxes on an image."""
    annotated = image.copy()

    for idx, cand in enumerate(candidates, start=1):
        x, y, w, h = cand["bbox"]
        label = f"Plate #{idx} (Score: {cand.get('score', 0.0)})"

        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

        banner_y1 = max(0, y - th - 6)
        banner_y2 = y
        cv2.rectangle(annotated, (x, banner_y1), (x + tw + 6, banner_y2), box_color, -1)
        cv2.putText(
            annotated,
            label,
            (x + 3, y - 4),
            font,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def create_processing_overview(
    image: np.ndarray,
    candidates: List[Dict[str, Any]],
    plates: List[Dict[str, Any]],
) -> np.ndarray:
    """Create a unified annotated visualization with both vehicle and plate candidates."""
    out = draw_vehicle_candidates(image, candidates)
    out = draw_plate_candidates(out, plates)
    return out
