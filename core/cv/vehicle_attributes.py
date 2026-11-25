"""Vehicle attributes extraction module for Project Third-Eye.

Provides:
- Safe vehicle cropping with bounds clamping.
- Dominant color extraction via OpenCV k-means mapped to a standard palette.
- Vehicle type classification from detection classes.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from core.schemas import Detection

# Standard color palette
NAMED_COLORS = ["white", "black", "grey", "red", "blue", "green", "yellow", "other"]

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}


def crop_vehicle(
    image: Union[str, Path, np.ndarray],
    detection: Union[Detection, List[float], Tuple[float, float, float, float]],
) -> np.ndarray:
    """Safely crop vehicle from image with bounds clamping.

    Args:
        image: File path or NumPy image array.
        detection: Detection schema instance or bounding box [xmin, ymin, xmax, ymax].

    Returns:
        NumPy array of the cropped vehicle region.
    """
    if isinstance(image, (str, Path)):
        img_path = Path(image)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image from path: {img_path}")
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if isinstance(detection, Detection):
        bbox = detection.bbox
    else:
        bbox = list(detection)

    if len(bbox) != 4:
        raise ValueError(f"Expected 4 bounding box coordinates, got {len(bbox)}")

    xmin, ymin, xmax, ymax = bbox
    height, width = img.shape[:2]

    # Check if box is completely outside boundaries or invalid
    if xmax <= 0.0 or ymax <= 0.0 or xmin >= float(width) or ymin >= float(height):
        raise ValueError(
            f"Bounding box [{xmin}, {ymin}, {xmax}, {ymax}] is completely outside image dimensions ({width}, {height})"
        )
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Degenerate bounding box: xmin ({xmin}) >= xmax ({xmax}) or ymin ({ymin}) >= ymax ({ymax})")

    # Safe bounds clamping
    x1 = max(0, min(int(round(xmin)), width - 1))
    y1 = max(0, min(int(round(ymin)), height - 1))
    x2 = max(0, min(int(round(xmax)), width))
    y2 = max(0, min(int(round(ymax)), height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop dimensions after clamping: [{x1}, {y1}, {x2}, {y2}] for image of size ({width}, {height})")

    return img[y1:y2, x1:x2].copy()


def classify_bgr_color(b: float, g: float, r: float) -> str:
    """Classify a single BGR color into the named palette using HSV space."""
    pixel_bgr = np.uint8([[[int(b), int(g), int(r)]]])
    pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(pixel_hsv[0]), int(pixel_hsv[1]), int(pixel_hsv[2])

    # Low value = Black
    if v < 45:
        return "black"

    # Low saturation = White or Grey
    if s < 45:
        if v > 185:
            return "white"
        return "grey"

    # Chromatic classification based on Hue
    if h <= 10 or h >= 165:
        return "red"
    elif 11 <= h <= 35:
        return "yellow"
    elif 36 <= h <= 85:
        return "green"
    elif 86 <= h <= 135:
        return "blue"

    return "other"


def dominant_color(crop: np.ndarray, k: int = 4) -> Tuple[str, float]:
    """Extract dominant vehicle color using OpenCV k-means clustering.

    Maps cluster centers to named palette:
    ['white', 'black', 'grey', 'red', 'blue', 'green', 'yellow', 'other']

    Returns:
        (color_name, pixel_share): The most dominant color name and its pixel proportion [0.0 - 1.0].
    """
    if crop is None or crop.size == 0:
        return "other", 0.0

    # Resize to standard thumbnail size for fast, stable k-means
    resized = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = resized.reshape(-1, 3).astype(np.float32)

    total_pixels = len(pixels)
    if total_pixels == 0:
        return "other", 0.0

    actual_k = min(k, max(1, len(pixels)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)

    try:
        _, labels, centers = cv2.kmeans(
            pixels, actual_k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
    except cv2.error:
        # Fallback to mean color if kmeans fails
        mean_bgr = pixels.mean(axis=0)
        color = classify_bgr_color(mean_bgr[0], mean_bgr[1], mean_bgr[2])
        return color, 1.0

    labels_flat = labels.flatten()
    color_counts = {name: 0 for name in NAMED_COLORS}

    for cluster_idx, center in enumerate(centers):
        count = int(np.count_nonzero(labels_flat == cluster_idx))
        color_name = classify_bgr_color(center[0], center[1], center[2])
        color_counts[color_name] += count

    # Determine dominant color with highest pixel share
    best_color = max(color_counts, key=color_counts.get)
    best_share = round(color_counts[best_color] / total_pixels, 4)

    return best_color, best_share


def get_vehicle_type(label: str) -> Optional[str]:
    """Map object detection label to vehicle type, or None if not a vehicle."""
    normalized = label.lower().strip()
    if normalized in VEHICLE_CLASSES:
        return normalized
    return None
