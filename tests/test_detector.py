"""Unit and integration tests for the computer vision detector (core/cv/detector.py).

Includes:
- Schema output validity and class filtering with explicitly MOCKED model output.
- Confidence threshold and bounding box boundary checks.
- One real-model integration test on real photos in data/samples/.
  If no images exist in data/samples/, skips with a clear message requesting real photos.
  (Strictly avoids creating fake/synthetic images).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

from core.cv.detector import (
    ALLOWED_CLASSES,
    detect,
    filter_detections,
    get_target_device,
    preprocess_image,
)
from core.cv.draw import draw_detections
from core.schemas import Detection


class TestDetectorLogicMocked:
    """Tests using explicitly MOCKED model outputs to verify filtering and schema compliance."""

    @pytest.fixture
    def mock_categories(self):
        # Sample COCO category names
        return [
            "__background__",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "couch",
            "dog",
        ]

    def test_filter_detections_class_filtering(self, mock_categories):
        """MOCKED test: Verify only traffic and triage classes are retained."""
        # Setup mocked model predictions
        # 1: person (keep), 3: car (keep), 5: airplane (reject), 9: couch (reject), 8: truck (keep)
        labels = torch.tensor([1, 3, 5, 9, 8], dtype=torch.int64)
        scores = torch.tensor([0.95, 0.88, 0.99, 0.91, 0.72], dtype=torch.float32)
        boxes = torch.tensor([
            [10.0, 10.0, 50.0, 100.0],   # person
            [60.0, 60.0, 200.0, 150.0],  # car
            [0.0, 0.0, 300.0, 200.0],    # airplane (rejected)
            [10.0, 20.0, 80.0, 90.0],    # couch (rejected)
            [50.0, 50.0, 250.0, 250.0],  # truck
        ], dtype=torch.float32)

        detections = filter_detections(
            boxes=boxes,
            labels=labels,
            scores=scores,
            categories=mock_categories,
            img_width=640,
            img_height=480,
            conf_threshold=0.5,
        )

        labels_kept = [d.label for d in detections]
        assert "person" in labels_kept
        assert "car" in labels_kept
        assert "truck" in labels_kept
        assert "airplane" not in labels_kept
        assert "couch" not in labels_kept
        assert len(detections) == 3

        # Verify Detection schema compliance
        for det in detections:
            assert isinstance(det, Detection)
            assert det.data_source == "REAL"
            assert det.confidence is not None
            assert 0.0 <= det.confidence <= 1.0
            assert len(det.bbox) == 4

    def test_filter_detections_confidence_threshold(self, mock_categories):
        """MOCKED test: Verify filtering by confidence threshold."""
        labels = torch.tensor([3, 3, 3], dtype=torch.int64)  # all cars
        scores = torch.tensor([0.85, 0.49, 0.12], dtype=torch.float32)
        boxes = torch.tensor([
            [10.0, 10.0, 100.0, 100.0],
            [10.0, 10.0, 100.0, 100.0],
            [10.0, 10.0, 100.0, 100.0],
        ], dtype=torch.float32)

        # Default threshold 0.5 should only keep score 0.85
        detections = filter_detections(
            boxes=boxes,
            labels=labels,
            scores=scores,
            categories=mock_categories,
            img_width=500,
            img_height=500,
            conf_threshold=0.5,
        )
        assert len(detections) == 1
        assert detections[0].confidence == 0.85

        # Lower threshold 0.4 should keep 0.85 and 0.49
        detections_low = filter_detections(
            boxes=boxes,
            labels=labels,
            scores=scores,
            categories=mock_categories,
            img_width=500,
            img_height=500,
            conf_threshold=0.4,
        )
        assert len(detections_low) == 2

    def test_filter_detections_reject_out_of_bounds_and_degenerate_boxes(self, mock_categories):
        """MOCKED test: Verify rejection of boxes outside image or with inverted coordinates."""
        labels = torch.tensor([3, 3, 3, 3, 3], dtype=torch.int64)  # cars
        scores = torch.tensor([0.90, 0.90, 0.90, 0.90, 0.90], dtype=torch.float32)
        img_w, img_h = 400, 300

        boxes = torch.tensor([
            [-5.0, 10.0, 100.0, 100.0],   # xmin < 0 (out of bounds)
            [10.0, -2.0, 100.0, 100.0],   # ymin < 0 (out of bounds)
            [10.0, 10.0, 450.0, 100.0],   # xmax > img_w (out of bounds)
            [10.0, 10.0, 100.0, 350.0],   # ymax > img_h (out of bounds)
            [100.0, 10.0, 50.0, 100.0],   # xmin >= xmax (degenerate)
        ], dtype=torch.float32)

        detections = filter_detections(
            boxes=boxes,
            labels=labels,
            scores=scores,
            categories=mock_categories,
            img_width=img_w,
            img_height=img_h,
            conf_threshold=0.5,
        )
        assert len(detections) == 0, f"Expected 0 detections, got {len(detections)}"

    def test_device_selection_mps_or_cpu(self):
        """Verify get_target_device returns a valid torch.device (MPS or CPU)."""
        device = get_target_device()
        assert isinstance(device, torch.device)
        assert device.type in {"mps", "cpu"}

    def test_draw_detections_renders_and_saves(self, tmp_path):
        """Verify draw_detections renders bounding boxes and writes output file."""
        import numpy as np
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        sample_det = Detection(
            detection_id="det-test-1",
            label="car",
            bbox=[10.0, 10.0, 80.0, 80.0],
            confidence=0.92,
            data_source="REAL"
        )
        out_file = tmp_path / "test_out.jpg"
        annotated_img, saved_path = draw_detections(canvas, [sample_det], output_path=out_file)
        assert saved_path.exists()
        assert annotated_img.shape == (200, 200, 3)


class TestRealModelIntegration:
    """Integration test with the live Faster R-CNN model on real sample images."""

    def test_live_detector_on_sample_image(self):
        """Runs the live Faster R-CNN model if a real photo exists in data/samples/.

        If no image is found, skips cleanly and prompts the user to add one.
        Strictly obeys the rule: do NOT generate or fake an image.
        """
        samples_dir = Path(__file__).resolve().parent.parent / "data" / "samples"
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        sample_images = [
            f for f in samples_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_exts
        ] if samples_dir.exists() else []

        if not sample_images:
            pytest.skip(
                "No real test photo found in data/samples/. "
                "Please add a real traffic or vehicle photo (e.g. data/samples/test.jpg) "
                "to run live model verification. Faked/synthetic images are strictly prohibited."
            )

        test_image_path = sample_images[0]
        detections = detect(test_image_path, conf_threshold=0.5)

        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)
            assert det.data_source == "REAL"
            assert det.label in ALLOWED_CLASSES
            assert det.confidence is not None
            assert det.confidence >= 0.5
