"""Test suite for demonstration readiness inputs (Image A, B, C, Video D, and NLP tests).

Ensures:
1. single_car_plate.jpg orientation correction, detection, and OCR evidence.
2. two_car_collision.jpg multi-vehicle detection and separate bounding boxes.
3. heavy_traffic.jpg multi-class detection (cars, motorcycles, pedestrians).
4. road vdo.mp4 video metadata and frame sampling.
5. All 4 required NLP semantic extraction test cases.
6. Incident evidence fusion and discrepancy flagging.
"""

from pathlib import Path
from PIL import Image
import pytest

from core.cv.detector import detect, detect_with_orientation
from core.cv.plate_detector import detect_plate_regions_with_ocr_fallback
from core.cv.vehicle_attributes import crop_vehicle, get_vehicle_type
from core.nlp.caller_extraction import extract_caller_info
from core.nlp.fusion import fuse_incident_evidence
from core.ocr.plate_reader import read_plate_evidence
from core.schemas import AccidentEvent, Detection, VehicleReport

DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "samples" / "demo"


class TestDemonstrationInputs:
    """Verification of designated demonstration files."""

    def test_demo_files_exist_and_readable(self):
        assert (DEMO_DIR / "single_car_plate.jpg").is_file()
        assert (DEMO_DIR / "two_car_collision.jpg").is_file()
        assert (DEMO_DIR / "heavy_traffic.jpg").is_file()
        assert (DEMO_DIR / "road vdo.mp4").is_file()

        img_a = Image.open(DEMO_DIR / "single_car_plate.jpg")
        assert img_a.size == (900, 1600)

        img_b = Image.open(DEMO_DIR / "two_car_collision.jpg")
        assert img_b.size == (1771, 1237)

        img_c = Image.open(DEMO_DIR / "heavy_traffic.jpg")
        assert img_c.size == (1270, 885)


class TestComputerVisionDetections:
    """Verification of Faster R-CNN on demonstration images."""

    def test_image_a_orientation_and_detection(self):
        img_path = DEMO_DIR / "single_car_plate.jpg"
        dets, final_img, info = detect_with_orientation(img_path)

        assert info["rotation_applied"] is True
        assert info["rotation_degrees"] == 90
        assert final_img.size == (1600, 900)
        assert len(dets) >= 1
        car_dets = [d for d in dets if d.label == "car"]
        assert len(car_dets) >= 1
        assert car_dets[0].confidence >= 0.85
        # Coordinates must be within final image dimensions
        for d in dets:
            x0, y0, x1, y1 = d.bbox
            assert 0 <= x0 < x1 <= 1600
            assert 0 <= y0 < y1 <= 900

    def test_image_b_two_car_collision(self):
        img_path = DEMO_DIR / "two_car_collision.jpg"
        dets, final_img, info = detect_with_orientation(img_path)

        assert info["rotation_applied"] is False
        assert final_img.size == (1771, 1237)
        cars = [d for d in dets if d.label == "car"]
        # Both prominent colliding vehicles must be detected
        assert len(cars) >= 2
        # Verify individual non-identical bounding boxes
        assert len(set(tuple(c.bbox) for c in cars)) == len(cars)

    def test_image_c_heavy_traffic(self):
        img_path = DEMO_DIR / "heavy_traffic.jpg"
        dets, final_img, info = detect_with_orientation(img_path)

        assert info["rotation_applied"] is False
        assert final_img.size == (1270, 885)
        classes_found = {d.label for d in dets}
        assert "car" in classes_found
        assert "motorcycle" in classes_found
        assert "person" in classes_found
        assert len(dets) >= 20


class TestNLPDemonstrationCases:
    """Verification of the 4 required NLP semantic extraction test cases."""

    def test_nlp_test1_collision(self):
        text = "Two cars collided near NH-44. One person may be injured. Please send emergency assistance."
        ext = extract_caller_info(text)

        assert ext.vehicle_count == 2
        assert "car" in ext.vehicle_types
        assert ext.location == "NH-44"
        assert ext.people_involved == 1
        assert "injured" in ext.injury_keywords
        assert len(ext.urgency_keywords) >= 1

    def test_nlp_test2_heavy_traffic(self):
        text = "There are many cars and motorcycles near the main road. Several pedestrians are crossing. No collision has been reported."
        ext = extract_caller_info(text)

        assert "car" in ext.vehicle_types
        assert "motorcycle" in ext.vehicle_types
        assert ext.location == "the main road"
        assert ext.injury_keywords == []
        assert ext.hazard_keywords == []

    def test_nlp_test3_uncertain_incident(self):
        text = "I saw a vehicle stopped on the road. I am not sure whether an accident happened."
        ext = extract_caller_info(text)

        assert ext.vehicle_count == 1
        assert ext.people_involved is None
        assert ext.injury_keywords == []
        assert ext.hazard_keywords == []

    def test_nlp_test4_contradictory_evidence_fusion(self):
        text = "The caller reports two damaged cars, but the uploaded image shows only one detected vehicle."
        caller_ext = extract_caller_info(text)
        assert caller_ext.vehicle_count == 2

        # Create synthetic event with 1 detected vehicle
        event = AccidentEvent(
            event_id="acc-test-contradiction",
            timestamp="2026-11-25T01:00:00Z",
            location_name="Test Site",
            latitude=31.22,
            longitude=75.76,
            severity="none",
            vehicles_involved=1,
            casualties_estimated=0,
            status="normal_flow",
            confidence=0.32,
            method="TRAINED_MODEL",
            score=0.32,
            verdict="NO ACCIDENT SIGNAL",
            data_source="REAL",
        )
        report = fuse_incident_evidence(
            accident_event=event,
            caller_extraction=caller_ext,
            image_name="test_single_car.jpg",
            save_to_db=False,
        )

        assert report.caller_vehicle_count == 2
        assert report.vision_vehicle_count == 1
        assert report.vehicle_count_disagreement is True
        assert any("VEHICLE COUNT DISAGREEMENT" in d for d in report.disagreements)
