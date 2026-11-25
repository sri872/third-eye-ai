"""Unit tests for vehicle attributes module (core/cv/vehicle_attributes.py).

Tests:
- Vehicle cropping with safe bounds clamping and invalid dimension handling.
- Dominant color extraction on real crops from data/samples/ and pure color patches.
- Vehicle class to vehicle type mapping.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from core.cv.vehicle_attributes import (
    NAMED_COLORS,
    classify_bgr_color,
    crop_vehicle,
    dominant_color,
    get_vehicle_type,
)
from core.schemas import Detection

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


class TestVehicleCropping:
    """Test safe cropping and coordinate clamping."""

    def test_crop_vehicle_clamping(self):
        # 100x100 white canvas
        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Detection coordinates exceeding canvas boundaries
        det = Detection(
            detection_id="det-crop-test",
            label="car",
            bbox=[-20.0, -10.0, 120.0, 110.0],
            data_source="REAL"
        )
        crop = crop_vehicle(canvas, det)
        # Should be clamped to full canvas (100, 100)
        assert crop.shape == (100, 100, 3)

    def test_crop_vehicle_invalid_box_raises(self):
        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Box completely outside canvas
        with pytest.raises(ValueError):
            crop_vehicle(canvas, [120.0, 120.0, 150.0, 150.0])


class TestVehicleTypeMapping:
    """Test detection label to vehicle type conversion."""

    def test_valid_vehicle_types(self):
        assert get_vehicle_type("car") == "car"
        assert get_vehicle_type("bus") == "bus"
        assert get_vehicle_type("truck") == "truck"
        assert get_vehicle_type("motorcycle") == "motorcycle"
        assert get_vehicle_type("bicycle") == "bicycle"

    def test_non_vehicle_labels_rejected(self):
        assert get_vehicle_type("person") is None
        assert get_vehicle_type("dog") is None
        assert get_vehicle_type("traffic light") is None


class TestDominantColorExtraction:
    """Test color classification and k-means clustering."""

    def test_classify_pure_bgr_colors(self):
        # Blue in BGR: (255, 0, 0)
        assert classify_bgr_color(255, 0, 0) == "blue"
        # Green in BGR: (0, 255, 0)
        assert classify_bgr_color(0, 255, 0) == "green"
        # Red in BGR: (0, 0, 255)
        assert classify_bgr_color(0, 0, 255) == "red"
        # White: (255, 255, 255)
        assert classify_bgr_color(255, 255, 255) == "white"
        # Black: (0, 0, 0)
        assert classify_bgr_color(0, 0, 0) == "black"

    def test_dominant_color_on_real_sample_crop(self):
        """Test dominant_color on real image crops extracted from data/samples/."""
        sample_files = list(SAMPLES_DIR.glob("*.jpg"))
        if not sample_files:
            pytest.skip("No real sample images found in data/samples/ for color testing.")

        img = cv2.imread(str(sample_files[0]))
        assert img is not None

        # Take a real crop from the image (center region)
        h, w = img.shape[:2]
        crop = img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

        color_name, share = dominant_color(crop)
        assert color_name in NAMED_COLORS
        assert isinstance(share, float)
        assert 0.0 <= share <= 1.0
