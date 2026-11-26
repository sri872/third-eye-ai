"""Tests for Classical Computer Vision Module.

Verifies image loading, enhancement, edge detection, diagnostics,
vehicle candidate localization, and plate region localization.
"""

import os
import pytest
import numpy as np
import cv2

from core.cv.image_processing import (
    load_image,
    enhance_image,
    to_grayscale,
    detect_edges,
    compute_image_diagnostics,
)
from core.cv.vehicle_localization import localize_vehicle_candidates
from core.cv.plate_localization import localize_plate_candidates
from core.cv.visualization import (
    draw_vehicle_candidates,
    draw_plate_candidates,
    create_processing_overview,
)


@pytest.fixture
def synthetic_image():
    """Create a clean 400x300 BGR synthetic test image with geometric rectangles."""
    img = np.full((300, 400, 3), 180, dtype=np.uint8)
    # Add dark road at bottom
    cv2.rectangle(img, (0, 200), (400, 300), (40, 40, 40), -1)
    # Add vehicle-like rectangle (width 120, height 70) -> aspect ratio 1.71
    cv2.rectangle(img, (140, 150), (260, 220), (20, 20, 200), -1)
    # Add plate-like rectangle (width 40, height 12) -> aspect ratio 3.33
    cv2.rectangle(img, (180, 195), (220, 207), (255, 255, 255), -1)
    return img


def test_load_image_numpy(synthetic_image):
    """Test loading from existing numpy array."""
    bgr, rgb = load_image(synthetic_image)
    assert bgr.shape == synthetic_image.shape
    assert rgb.shape == synthetic_image.shape
    # Check RGB vs BGR channel ordering
    assert bgr[160, 200, 2] == rgb[160, 200, 0]  # Red channel matches


def test_load_image_preserves_portrait_orientation():
    """Portrait inputs are not automatically rotated based on aspect ratio."""
    portrait = np.zeros((400, 200, 3), dtype=np.uint8)
    bgr, rgb = load_image(portrait)
    assert bgr.shape == portrait.shape
    assert rgb.shape == portrait.shape


def test_load_image_explicit_counterclockwise_rotation():
    """Explicit rotation is applied only when the caller requests it."""
    portrait = np.zeros((400, 200, 3), dtype=np.uint8)
    portrait[20:80, 10:40] = (0, 0, 255)
    bgr, _ = load_image(portrait, rotation_degrees=270)
    expected = cv2.rotate(portrait, cv2.ROTATE_90_COUNTERCLOCKWISE)
    assert np.array_equal(bgr, expected)


def test_load_image_invalid():
    """Test invalid input raises ValueError."""
    with pytest.raises(ValueError):
        load_image(np.zeros((0, 0, 3), dtype=np.uint8))

    with pytest.raises(ValueError):
        load_image(12345)


def test_enhance_image(synthetic_image):
    """Test enhancement operations return valid shapes and datatypes."""
    enhanced = enhance_image(
        synthetic_image,
        brightness=20,
        contrast=1.2,
        apply_clahe=True,
        blur_type="gaussian",
        blur_kernel=3,
    )
    assert enhanced.shape == synthetic_image.shape
    assert enhanced.dtype == np.uint8


def test_to_grayscale_and_edges(synthetic_image):
    """Test grayscale conversion and Canny edge detection."""
    gray = to_grayscale(synthetic_image)
    assert len(gray.shape) == 2
    assert gray.shape == (300, 400)

    edges = detect_edges(gray, low_thresh=50, high_thresh=150)
    assert edges.shape == (300, 400)
    assert edges.dtype == np.uint8
    # Edge detector must detect boundaries
    assert np.count_nonzero(edges) > 50


def test_compute_image_diagnostics(synthetic_image):
    """Test image diagnostics calculation."""
    diag = compute_image_diagnostics(synthetic_image)
    assert diag["valid"] is True
    assert diag["width"] == 400
    assert diag["height"] == 300
    assert diag["channels"] == 3
    assert diag["mean_brightness"] > 0
    assert diag["sharpness_laplacian_var"] > 0
    assert diag["exposure_assessment"] in ("MID_RANGE_EXPOSURE", "LOW_EXPOSURE", "HIGH_EXPOSURE_OR_GLARE")


def test_localize_vehicle_candidates(synthetic_image):
    """Test candidate vehicle localization produces valid bounding boxes."""
    candidates = localize_vehicle_candidates(synthetic_image)
    assert isinstance(candidates, list)
    for c in candidates:
        bx, by, bw, bh = c["bbox"]
        assert 0 <= bx < 400
        assert 0 <= by < 300
        assert bw > 0 and bh > 0
        assert 0.0 <= c["heuristic_confidence"] <= 1.0
        assert "GEOMETRIC CANDIDATE" in c["label"]


def test_localize_plate_candidates(synthetic_image):
    """Test plate candidate localization."""
    plates = localize_plate_candidates(synthetic_image)
    assert isinstance(plates, list)
    for p in plates:
        px, py, pw, ph = p["bbox"]
        assert 0 <= px < 400
        assert 0 <= py < 300
        assert pw > 0 and ph > 0
        assert p["aspect_ratio"] >= 1.5
        assert p["crop"].size > 0


def test_visualization_drawing(synthetic_image):
    """Test drawing bounding boxes on images."""
    cand = [{
        "bbox": [50, 50, 100, 60],
        "aspect_ratio": 1.67,
        "type_hint": "CAR_OR_SUV",
        "heuristic_confidence": 0.85
    }]
    annotated = draw_vehicle_candidates(synthetic_image, cand)
    assert annotated.shape == synthetic_image.shape
    # Ensure drawing modified pixels
    assert not np.array_equal(annotated, synthetic_image)

    plate = [{
        "bbox": [60, 80, 40, 12],
        "score": 0.75
    }]
    annotated_plate = draw_plate_candidates(synthetic_image, plate)
    assert annotated_plate.shape == synthetic_image.shape

    combined = create_processing_overview(synthetic_image, cand, plate)
    assert combined.shape == synthetic_image.shape


def test_plate_localizer_includes_full_frame_when_vehicle_regions_are_supplied(synthetic_image):
    # The explicit ROI is deliberately on the left. The full frame must still
    # be included in the search rather than being silently replaced by ROIs.
    candidates = localize_plate_candidates(
        synthetic_image,
        search_regions=[(0, 0, 100, 100)],
        min_area=20,
    )
    assert isinstance(candidates, list)
    for candidate in candidates:
        assert candidate["method"] == "MULTI_PATH_CLASSICAL_MORPHOLOGY"
        assert len(candidate["bbox"]) == 4
