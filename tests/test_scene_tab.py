"""Unit tests for conservative, pixel-derived scene context.

These tests cover only ``core/cv/scene_analysis.py``; no Streamlit AppTest and
no application wiring is involved. Synthetic pixel buffers are used for exact
luminance/contrast bands; the smoke test uses a real sample photo when present.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from core.cv.scene_analysis import (
    SceneAnalysisResult,
    analyze_scene_environment,
    build_scene_summary,
    suppress_duplicate_detections,
)
from core.schemas import Detection


def _detection(detection_id: str, label: str, bbox, confidence: float = 0.9) -> Detection:
    return Detection(
        detection_id=detection_id,
        label=label,
        bbox=list(bbox),
        confidence=confidence,
        data_source="REAL",
    )


class TestInvalidInput:
    @pytest.mark.parametrize("bad_input", [None, "not-an-image", 42, [1, 2, 3]])
    def test_rejects_non_image_input(self, bad_input):
        with pytest.raises(ValueError, match="Invalid image input"):
            analyze_scene_environment(bad_input)

    def test_rejects_empty_arrays(self):
        for shape in [(0, 0), (0, 10, 3), (10, 0)]:
            with pytest.raises(ValueError, match="Empty image"):
                analyze_scene_environment(np.empty(shape, dtype=np.uint8))

    def test_rejects_unsupported_array_shape(self):
        with pytest.raises(ValueError, match="Invalid image shape"):
            analyze_scene_environment(np.zeros((10, 10, 2), dtype=np.uint8))


class TestPixelDerivedMetrics:
    def test_dark_gray_array_is_night_and_low_contrast(self):
        image = np.full((80, 60), 20, dtype=np.uint8)
        result = analyze_scene_environment(image)

        assert result.mean_luminance == pytest.approx(20.0)
        assert result.contrast_std == pytest.approx(0.0)
        assert result.lighting_condition == "Night / Very low light"
        assert result.contrast_condition == "Low contrast / Diffuse"

    def test_bright_pil_image_is_bright_with_low_contrast(self):
        image = Image.fromarray(np.full((60, 80, 3), 230, dtype=np.uint8), "RGB")
        result = analyze_scene_environment(image)

        assert result.mean_luminance == pytest.approx(230.0)
        assert result.contrast_std == pytest.approx(0.0)
        assert result.lighting_condition == "Bright daylight / High glare"

    def test_luminance_uses_real_pixel_values_not_detections(self):
        # OpenCV gradient pattern: dark left half, light right half.
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, :50, :] = 10
        rgb[:, 50:, :] = 245
        image = Image.fromarray(rgb, "RGB")

        dark = analyze_scene_environment(image)
        bright = analyze_scene_environment(Image.fromarray(np.full((100, 100, 3), 245, dtype=np.uint8), "RGB"))

        assert dark.contrast_std > 65.0
        assert dark.contrast_condition == "High contrast"
        assert dark.mean_luminance < bright.mean_luminance
        assert dark.road_surface_luminance is not None
        assert bright.road_surface_luminance is not None
        assert dark.road_surface_assessment == (
            "NOT DETERMINED (foreground brightness only; material unverified)"
        )


class TestDetectionDerivedCounts:
    def test_counts_vehicle_and_person_labels_only(self):
        image = np.full((120, 160, 3), 130, dtype=np.uint8)
        detections = [
            _detection("car-1", "car", [0.0, 0.0, 50.0, 50.0]),
            _detection("truck-1", "truck", [60.0, 0.0, 110.0, 50.0]),
            _detection("bicycle-1", "bicycle", [120.0, 0.0, 150.0, 50.0]),
            _detection("person-1", "person", [0.0, 60.0, 40.0, 110.0]),
            _detection("dog-1", "dog", [60.0, 60.0, 100.0, 110.0], confidence=0.6),
        ]

        result = analyze_scene_environment(image, detections)

        assert result.vehicles_detected_count == 3
        assert result.persons_visible_count == 1
        assert "3 vehicle(s) and 1 person(s)" in result.visibility_assessment
        assert result.scene_type == "Vehicle boxes detected"
        assert result.environment_classification == "Detected-vehicle scene (environment not determined)"

    def test_empty_detection_counts_and_scene_classification(self):
        result = analyze_scene_environment(np.full((50, 50), 100, dtype=np.uint8))

        assert result.vehicles_detected_count == 0
        assert result.persons_visible_count == 0
        assert result.visibility_assessment == "No vehicles or pedestrians localized in foreground"
        assert result.scene_type == "Scene type not determined from pixels"


class TestNotDeterminedFields:
    def test_weather_and_gps_are_explicitly_not_determined(self):
        result = analyze_scene_environment(
            np.full((60, 60, 3), 150, dtype=np.uint8),
            [_detection("car-1", "car", [0, 0, 30, 30])],
            media_type="video",
        )

        assert result.weather_condition == "not determined (requires external atmospheric sensor)"
        assert result.gps_coordinates == "not determined (no GPS sensor in media pixels)"
        assert "not determined" in result.weather_condition.lower()
        assert "not determined" in result.gps_coordinates.lower()

        with pytest.raises(FrozenInstanceError):
            result.weather_condition = "sunny"  # type: ignore[misc]

    def test_result_is_frozen_dataclass_with_required_fields(self):
        expected_fields = {
            "scene_type",
            "environment_classification",
            "lighting_condition",
            "mean_luminance",
            "contrast_condition",
            "contrast_std",
            "road_surface_luminance",
            "road_surface_assessment",
            "vehicles_detected_count",
            "persons_visible_count",
            "visibility_assessment",
            "scene_description",
            "weather_condition",
            "gps_coordinates",
        }
        assert set(SceneAnalysisResult.__dataclass_fields__) == expected_fields


class TestConservativeSuppression:
    def test_near_identical_same_class_is_suppressed(self):
        detections = [
            _detection("dupe-low", "car", [10, 10, 110, 110], confidence=0.7),
            _detection("dupe-high", "car", [10, 10, 110, 110], confidence=0.95),
        ]
        result = suppress_duplicate_detections(detections)

        assert result.raw_count == 2
        assert result.kept_count == 1
        assert result.suppressed_count == 1
        assert [item.detection_id for item in result.detections] == ["dupe-high"]

    def test_adjacent_same_class_boxes_are_preserved(self):
        # Nearly identical same-class boxes with IoU 0.81 and containment 0.8182
        # remain separate because both are below the conservative thresholds.
        adjacent = [
            _detection("car-a", "car", [0, 0, 100, 100], confidence=0.9),
            _detection("car-b", "car", [19, 0, 119, 100], confidence=0.8),
        ]
        result = suppress_duplicate_detections(adjacent)

        assert result.kept_count == 2
        assert result.suppressed_count == 0
        assert {item.detection_id for item in result.detections} == {"car-a", "car-b"}
        assert suppress_duplicate_detections.__defaults__ == (0.85, 0.95)

    def test_cross_class_containment_is_preserved(self):
        result = suppress_duplicate_detections(
            [
                _detection("car-outer", "car", [0, 0, 100, 100]),
                _detection("person-inside", "person", [10, 10, 20, 90]),
            ]
        )
        assert result.kept_count == 2


class TestBuildSceneSummaryLimitations:
    def test_summary_keeps_limitations_text(self):
        detections = [_detection("car-1", "car", [0, 0, 20, 20])]
        summary = build_scene_summary(detections, 640, 480, media_type="image")

        assert summary["vehicle_count"] == 1
        assert summary["person_count"] == 0
        assert summary["postprocessed_detection_count"] == 1
        assert "Weather and road-surface condition: NOT DETERMINED from the supplied pixels." in summary["limitations"]
        assert "GPS location: NOT DETERMINED from pixels; any location is a separate simulated selection." in summary["limitations"]
        assert "Traffic speed and direction of travel: NOT DETERMINED from a single frame." in summary["limitations"]
        assert "Injuries and occupants inside vehicles: NOT DETERMINED; only visible exterior people are counted." in summary["limitations"]

    def test_video_limitations_retain_sampled_frame_constraints(self):
        summary = build_scene_summary([], 320, 240, media_type="video")
        assert any("Video evidence is limited to evenly sampled frames" in line for line in summary["limitations"])


class TestRealSampleSmoke:
    def test_real_sample_scene_metrics(self):
        sample = Path(__file__).resolve().parent.parent / "data" / "samples" / "road_1.jpg"
        if not sample.exists():
            pytest.skip("data/samples/road_1.jpg is not present; skipping real-sample smoke test")

        rgb = cv2.cvtColor(cv2.imread(str(sample)), cv2.COLOR_BGR2RGB)
        result = analyze_scene_environment(Image.fromarray(rgb))

        assert isinstance(result, SceneAnalysisResult)
        assert result.mean_luminance >= 0.0
        assert 0.0 <= result.contrast_std <= 255.0
        assert result.lighting_condition in {
            "Night / Very low light",
            "Dim / Overcast / Low daylight",
            "Normal daylight",
            "Bright daylight / High glare",
        }
        assert result.scene_description
        assert "not determined" in result.weather_condition
        assert "not determined" in result.gps_coordinates
