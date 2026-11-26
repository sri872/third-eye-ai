"""Classical Character Segmentation.

Segments individual character glyphs from a binarized license plate image
using connected component and contour analysis with spatial aspect ratio bounds.
"""

from typing import List, Tuple
import cv2
import numpy as np


def segment_characters(
    binary_plate: np.ndarray,
    target_char_w: int = 24,
    target_char_h: int = 32,
    min_char_h_ratio: float = 0.25,
    max_char_h_ratio: float = 0.95,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Segment individual characters from a binarized license plate image.

    Args:
        binary_plate: Binarized plate (characters white=255, background black=0).
        target_char_w: Normalized character width for template matching.
        target_char_h: Normalized character height for template matching.
        min_char_h_ratio: Minimum character height relative to plate height.
        max_char_h_ratio: Maximum character height relative to plate height.

    Returns:
        List of tuples: (normalized_binary_char, (x, y, w, h)).
        Sorted left-to-right by x coordinate.
    """
    if binary_plate is None or binary_plate.size == 0:
        return []

    plate_h, plate_w = binary_plate.shape[:2]
    min_h = int(plate_h * min_char_h_ratio)
    max_h = int(plate_h * max_char_h_ratio)

    # 1. Clean small salt-and-pepper noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary_plate, cv2.MORPH_OPEN, kernel)

    # 2. Find contours of potential character components using RETR_CCOMP to handle outer border frames
    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Tuple[int, int, int, int]] = []

    if contours is not None:
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)

            # Reject full plate borders or giant outer frames
            if w > int(plate_w * 0.70) or h > int(plate_h * 0.95):
                continue

            # Height check: character must be at least min_h
            if h < min_h:
                continue

            # Width check (character shouldn't span more than 35% of plate width)
            if w < 4 or w > int(plate_w * 0.35):
                continue

            # Aspect ratio check: characters are usually taller than wide
            # except characters like 'M' or 'W' which can be roughly square (h/w >= 0.75)
            hw_ratio = float(h) / float(w) if w > 0 else 0.0
            if hw_ratio < 0.70 or hw_ratio > 5.5:
                continue

            candidates.append((x, y, w, h))

    # 3. Deduplicate / filter nested boxes
    filtered_boxes = _filter_nested_boxes(candidates)

    # 4. Sort left-to-right by x coordinate
    filtered_boxes.sort(key=lambda b: b[0])

    # 5. Extract and normalize each character to (target_char_w, target_char_h)
    segmented: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []

    for bx, by, bw, bh in filtered_boxes:
        char_roi = binary_plate[by : by + bh, bx : bx + bw]

        # Add small proportional border before resizing
        pad_y = max(1, int(bh * 0.10))
        pad_x = max(1, int(bw * 0.10))
        padded = cv2.copyMakeBorder(
            char_roi, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0
        )

        # Resize to fixed template dimension
        resized = cv2.resize(
            padded, (target_char_w, target_char_h), interpolation=cv2.INTER_AREA
        )
        _, norm_char = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        segmented.append((norm_char, (bx, by, bw, bh)))

    return segmented


def _filter_nested_boxes(
    boxes: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    """Remove candidate boxes that are completely inside larger character boxes."""
    if not boxes:
        return []

    kept = []
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        is_nested = False
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i != j:
                # If box 1 is inside box 2
                if (
                    x1 >= x2
                    and y1 >= y2
                    and (x1 + w1) <= (x2 + w2)
                    and (y1 + h1) <= (y2 + h2)
                ):
                    is_nested = True
                    break
        if not is_nested:
            kept.append((x1, y1, w1, h1))

    return kept
