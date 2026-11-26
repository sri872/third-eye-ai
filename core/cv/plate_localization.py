"""Classical Number Plate Region Localization.

Uses deterministic morphological filtering, vertical gradient analysis (Sobel),
and geometric aspect-ratio constraints to locate license plate candidate regions
without any neural-network models.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


def localize_plate_candidates(
    image: np.ndarray,
    search_regions: List[Tuple[int, int, int, int]] = None,
    min_aspect_ratio: float = 1.8,
    max_aspect_ratio: float = 6.0,
    min_area: int = 400,
    max_area: int = 150000,
) -> List[Dict[str, Any]]:
    """Localize candidate license plate regions.

    Args:
        image: Input BGR image.
        search_regions: Optional list of (x, y, w, h) bounding boxes to search inside.
            If None, searches across the entire image.
        min_aspect_ratio: Minimum width/height ratio (Indian plates ~2.2 - 5.0).
        max_aspect_ratio: Maximum width/height ratio.
        min_area: Minimum bounding box pixel area.
        max_area: Maximum bounding box pixel area.

    Returns:
        List of plate candidate dictionaries.
    """
    if image is None or image.size == 0:
        return []

    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # If search regions are provided, search each region; else search full image
    regions = search_regions if search_regions else [(0, 0, w_img, h_img)]
    candidates: List[Dict[str, Any]] = []

    for rx, ry, rw, rh in regions:
        # Clamp to image boundaries
        rx = max(0, min(rx, w_img - 1))
        ry = max(0, min(ry, h_img - 1))
        rw = max(1, min(rw, w_img - rx))
        rh = max(1, min(rh, h_img - ry))

        roi_gray = gray[ry : ry + rh, rx : rx + rw]
        if roi_gray.shape[0] < 20 or roi_gray.shape[1] < 40:
            continue

        # 1. Morphological Top-Hat and Black-Hat to emphasize high contrast characters
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        top_hat = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, rect_kernel)
        black_hat = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, rect_kernel)
        enhanced = cv2.add(roi_gray, top_hat)
        enhanced = cv2.subtract(enhanced, black_hat)

        # 2. Compute horizontal gradient (vertical edges of characters)
        grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)

        # 3. Gaussian blur & Otsu threshold
        blurred = cv2.GaussianBlur(grad_x, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Morphological closing to connect adjacent character vertical edges into a horizontal plate blob
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        # 5. Find contours of candidate plate blobs
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            px, py, pw, ph = cv2.boundingRect(cnt)
            area = pw * ph

            if area < min_area or area > max_area:
                continue

            ar = float(pw) / float(ph) if ph > 0 else 0.0
            if ar < min_aspect_ratio or ar > max_aspect_ratio:
                continue

            # Absolute coordinates in original image
            abs_x = rx + px
            abs_y = ry + py

            # Crop candidate plate from original BGR image
            plate_crop = image[abs_y : abs_y + ph, abs_x : abs_x + pw]
            if plate_crop.size == 0:
                continue

            # Score plate candidate based on typical Indian HSRP ratio (~3.8 - 4.5)
            # and edge density
            edge_roi = thresh[py : py + ph, px : px + pw]
            density = float(cv2.countNonZero(edge_roi)) / float(area) if area > 0 else 0.0

            ratio_fit = 1.0 - min(1.0, abs(ar - 4.0) / 2.5)
            density_fit = min(1.0, density * 3.0)
            score = round(0.6 * ratio_fit + 0.4 * density_fit, 3)

            candidates.append(
                {
                    "bbox": [int(abs_x), int(abs_y), int(pw), int(ph)],
                    "aspect_ratio": round(ar, 2),
                    "area": int(area),
                    "edge_density": round(density, 3),
                    "score": score,
                    "crop": plate_crop,
                    "method": "MORPHOLOGY_SOBEL_ASPECT_RATIO",
                }
            )

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates
