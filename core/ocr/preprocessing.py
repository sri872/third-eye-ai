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



def preprocess_plate_variants(
    plate_bgr: np.ndarray,
    target_height: int = 80,
    deskew: bool = True,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return several polarity/threshold variants for robust classical segmentation.

    Each result is (method_name, enhanced_gray, binary_with_white_glyphs).
    This deliberately exposes multiple deterministic threshold choices instead
    of assuming one lighting/polarity model works for every plate.
    """
    enhanced, otsu_inverse = preprocess_plate(
        plate_bgr, target_height=target_height, deskew=deskew
    )
    h, w = enhanced.shape[:2]
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []

    def normalize_polarity(binary: np.ndarray) -> np.ndarray:
        # Characters should be foreground. For a crop whose outer border is
        # mostly white, invert so plate background becomes black.
        border = np.concatenate((binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]))
        return cv2.bitwise_not(binary) if float(np.mean(border)) > 127.0 else binary

    variants.append(("OTSU", enhanced, normalize_polarity(otsu_inverse)))

    block = min(31, min(h, w) if min(h, w) % 2 else min(h, w) - 1)
    block = max(3, block)
    if block % 2 == 0:
        block -= 1

    for method, polarity in (
        ("ADAPTIVE_GAUSSIAN", cv2.THRESH_BINARY_INV),
        ("ADAPTIVE_MEAN", cv2.THRESH_BINARY_INV),
    ):
        adaptive_method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if method == "ADAPTIVE_GAUSSIAN"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        binary = cv2.adaptiveThreshold(
            enhanced, 255, adaptive_method, polarity, block, 7
        )
        variants.append((method, enhanced, normalize_polarity(binary)))

    # A light median-cleaned Otsu branch helps noisy/compressed plate crops.
    median = cv2.medianBlur(enhanced, 3)
    _, cleaned = cv2.threshold(
        median, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    variants.append(("MEDIAN_OTSU", enhanced, normalize_polarity(cleaned)))

    # Remove byte-identical variants; retain the first method label.
    unique: list[tuple[str, np.ndarray, np.ndarray]] = []
    for item in variants:
        if not any(np.array_equal(item[2], prior[2]) for prior in unique):
            unique.append(item)
    return unique

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
