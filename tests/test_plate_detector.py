"""Phase 12 tests for deterministic plate-region localisation.

These tests use synthetic plate images with known coordinates. They verify only
the localisation contract; OCR text is tested separately in test_plate_reader.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.cv.plate_detector import detect_plate_regions


def _plate_canvas(
    plates: list[tuple[int, int, int, int, str]], size: tuple[int, int] = (360, 180)
) -> np.ndarray:
    """Draw bright rectangular plates with dark character-like text."""
    width, height = size
    image = np.full((height, width, 3), 205, dtype=np.uint8)
    for x0, y0, x1, y1, text in plates:
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), -1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (25, 25, 25), 2)
        cv2.putText(
            image,
            text,
            (x0 + 8, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return image


def test_detects_one_valid_plate_and_maps_to_original_coordinates() -> None:
    image = _plate_canvas([(20, 50, 300, 110, "MH12AB1234")])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    regions = detect_plate_regions(image, (100, 200, 460, 380), use_model=False)

    assert regions, "Synthetic plate must be localised before OCR"
    region = regions[0]
    assert region.bbox_local[0] >= 0 and region.bbox_local[2] <= image.shape[1]
    assert region.bbox_local[1] >= 0 and region.bbox_local[3] <= image.shape[0]
    assert region.bbox_original == (
        100 + region.bbox_local[0],
        200 + region.bbox_local[1],
        100 + region.bbox_local[2],
        200 + region.bbox_local[3],
    )
    assert region.character_candidates >= 3
    assert 0.0 < region.heuristic_score <= 1.0
    assert not gray.size == 0


def test_detects_multiple_plates_in_one_crop() -> None:
    image = _plate_canvas(
        [
            (15, 20, 165, 65, "KA01AA1111"),
            (185, 105, 340, 155, "DL02BB2222"),
        ]
    )
    regions = detect_plate_regions(image, max_candidates=4, use_model=False)
    assert len(regions) >= 2, [r.bbox_local for r in regions]
    boxes = [r.bbox_local for r in regions[:2]]
    left_plate = any(box[0] < 170 and box[2] <= 185 for box in boxes)
    right_plate = any(box[0] >= 170 for box in boxes)
    assert left_plate and right_plate, boxes


def test_invalid_vehicle_box_is_rejected() -> None:
    image = _plate_canvas([(20, 50, 300, 110, "MH12AB1234")])
    assert detect_plate_regions(image, (300, 200, 100, 380), use_model=False) == ()
    assert detect_plate_regions(image, (0, 0, float("nan"), 100), use_model=False) == ()


def test_undersized_plate_is_rejected_when_below_configured_minimum() -> None:
    image = _plate_canvas([(10, 70, 40, 82, "AB1")], size=(120, 160))
    assert detect_plate_regions(image, min_width=80, min_height=40, use_model=False) == ()


def test_no_plate_region_on_blank_vehicle_crop() -> None:
    blank = np.full((180, 360, 3), 175, dtype=np.uint8)
    assert detect_plate_regions(blank, use_model=False) == ()


def test_vendored_yunet_weights_match_official_hash_and_load() -> None:
    """The learned detector is a verified, locally vendored OpenCV Zoo model."""
    from core.cv.plate_detector_yunet import (
        DETECTION_THRESHOLD,
        METHOD,
        YuNetPlateDetector,
        verify_model_file,
    )

    assert verify_model_file()
    detector = YuNetPlateDetector()
    assert detector.conf_threshold == DETECTION_THRESHOLD
    assert METHOD == "YUNET_LPD_2023MAR"

    image = _plate_canvas([(20, 50, 300, 110, "MH12AB1234")])
    boxes = detector.detect(image)
    assert isinstance(boxes, tuple)
    for bbox, score in boxes:
        assert 0.0 <= score <= 1.0
        assert bbox[0] < bbox[2] and bbox[1] < bbox[3]


def test_default_detector_prefers_verified_model() -> None:
    """With model available, default detection returns model-method regions only."""
    from core.cv.plate_detector_yunet import METHOD as YUNET_METHOD

    image = _plate_canvas([(20, 50, 300, 110, "MH12AB1234")])
    regions = detect_plate_regions(image)
    assert all(region.method == YUNET_METHOD for region in regions)


def test_candidate_boxes_are_clamped_to_vehicle_bbox() -> None:
    image = _plate_canvas([(20, 50, 300, 110, "MH12AB1234")])
    vehicle_bbox = (1000, 500, 1360, 680)
    regions = detect_plate_regions(image, vehicle_bbox, use_model=False)
    assert regions
    for region in regions:
        x0, y0, x1, y1 = region.bbox_original
        assert vehicle_bbox[0] <= x0 < x1 <= vehicle_bbox[2]
        assert vehicle_bbox[1] <= y0 < y1 <= vehicle_bbox[3]


def test_multiple_plate_boxes_are_tight_to_the_drawn_plates() -> None:
    """Joined contours must not inflate the localised plate rectangles."""
    image = _plate_canvas(
        [
            (15, 20, 165, 65, "KA01AA1111"),
            (185, 105, 340, 155, "DL02BB2222"),
        ]
    )
    regions = detect_plate_regions(image, max_candidates=4, use_model=False)
    assert len(regions) >= 2
    boxes = sorted(region.bbox_local for region in regions[:2])
    expected = [(15, 20, 165, 65), (185, 105, 340, 155)]
    for box, target in zip(boxes, expected):
        for actual, wanted in zip(box, target):
            assert abs(actual - wanted) <= 3, (boxes, expected)


def test_undersized_plate_is_not_inflated_past_minimum_by_character_strokes() -> None:
    """Text above a tiny plate must not create a passing contour by itself."""
    image = _plate_canvas([(10, 70, 28, 76, "AB1")], size=(120, 160))
    assert detect_plate_regions(image, use_model=False) == ()
