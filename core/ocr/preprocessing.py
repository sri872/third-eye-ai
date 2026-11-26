"""Classical Number Plate Preprocessing.

Enhances license plate crops using contrast adjustment, bilateral filtering,
deskewing, and adaptive binarization to prepare for character segmentation.
"""

from typing import Tuple
import cv2
import numpy as np


def preprocess_plate(
    plate_bgr: np.ndarray,
    target_height: int = 64,
    deskew: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess license plate crop.

    Args:
        plate_bgr: BGR crop of candidate license plate.
        target_height: Standardized height for character segmentation.
        deskew: Whether to estimate and correct rotational skew.

    Returns:
        Tuple of (enhanced_gray, binary_plate).
    """
    if plate_bgr is None or plate_bgr.size == 0:
        raise ValueError("Empty plate crop received.")

    # 1. Resize to target height maintaining aspect ratio
    h, w = plate_bgr.shape[:2]
    scale = float(target_height) / float(h)
    target_width = max(30, int(w * scale))
    resized = cv2.resize(plate_bgr, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale conversion
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. Deskew if requested
    if deskew:
        gray = _deskew_plate(gray)

    # 4. Bilateral filtering for edge-preserving noise reduction
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # 5. Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    # 6. Adaptive binarization (Otsu + Gaussian adaptive threshold)
    # Number plates can be dark-on-light (white/yellow plate) or light-on-dark (black plate)
    # We want characters to be white (255) on dark (0) background for contour extraction
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Check if border is white; if so, invert to keep characters white
    # Count pixels on border:
    border_pixels = np.concatenate([otsu[0, :], otsu[-1, :], otsu[:, 0], otsu[:, -1]])
    if np.mean(border_pixels) > 127:
        otsu = cv2.bitwise_not(otsu)

    return enhanced, otsu


def _deskew_plate(gray: np.ndarray) -> np.ndarray:
    """Estimate and correct rotational skew of plate crop."""
    # Compute horizontal edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return gray

    angles = []
    for line in lines:
        pts = line[0] if len(line.shape) > 1 else line
        if len(pts) >= 4:
            x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
            dx = x2 - x1
            dy = y2 - y1
            if dx != 0:
                angle = np.degrees(np.arctan2(dy, dx))
                if abs(angle) < 25.0:
                    angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
