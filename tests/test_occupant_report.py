"""Unit and integration tests for occupant and pedestrian reporting (Phase 4).

Covers:
1. MOCKED detections: occupant counts, vehicle assignment, overlap calculation.
2. Limitation note verification on every report.
3. Strict absence of guessed medical or demographic attributes ('not determined', 'unknown').
4. Real sample image integration test.
"""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from core.cv.detector import detect
from core.cv.vehicle_report import build_vehicle_reports
from core.nlp.occupant_report import (
    DEFAULT_LIMITATION_NOTE,
    build_occupant_reports,
    compute_box_intersection_ratio,
)
from core.schemas import Detection, OccupantReport, VehicleReport
import cv2

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


class TestOccupantReportMocked:
    """MOCKED tests: Verify occupant report generation logic."""

    def test_box_intersection_ratio(self):
        # Disjoint boxes
        box_p = [0.0, 0.0, 50.0, 50.0]
        box_v = [100.0, 100.0, 200.0, 200.0]
        assert compute_box_intersection_ratio(box_p, box_v) == 0.0

        # Partial overlap (half person inside vehicle box)
        box_p2 = [0.0, 0.0, 100.0, 100.0]  # area 10,000
        box_v2 = [50.0, 0.0, 150.0, 100.0] # inter area 50 * 100 = 5,000
        ratio = compute_box_intersection_ratio(box_p2, box_v2)
        assert abs(ratio - 0.50) < 1e-4

    def test_mocked_no_persons_detected(self):
        """MOCKED test: scene with only vehicles and zero persons."""
        detections = [
            Detection(detection_id="d1", label="car", bbox=[10.0, 10.0, 100.0, 80.0], confidence=0.88, data_source="REAL"),
            Detection(detection_id="d2", label="truck", bbox=[120.0, 50.0, 300.0, 200.0], confidence=0.75, data_source="REAL"),
        ]
        reports, summary = build_occupant_reports(detections)
        assert len(reports) == 0
        assert summary["persons_visible_count"] == 0
        assert summary["persons_near_vehicles_count"] == 0

    def test_mocked_disjoint_pedestrian(self):
        """MOCKED test: person walking away from vehicles with zero overlap."""
        detections = [
            Detection(detection_id="v1", label="car", bbox=[10.0, 10.0, 100.0, 80.0], confidence=0.90, data_source="REAL"),
            Detection(detection_id="p1", label="person", bbox=[300.0, 300.0, 340.0, 400.0], confidence=0.85, data_source="REAL"),
        ]
        reports, summary = build_occupant_reports(detections)
        assert len(reports) == 1
        assert summary["persons_visible_count"] == 1
        assert summary["persons_near_vehicles_count"] == 0

        rep = reports[0]
        assert rep.vehicle_id is None
        assert rep.overlap_ratio == 0.0
        assert rep.confidence == 0.85
        assert rep.injury_severity == "not determined"
        assert rep.consciousness == "unknown"
        assert rep.limitation_note == DEFAULT_LIMITATION_NOTE

    def test_mocked_person_overlapping_vehicle(self):
        """MOCKED test: two vehicles and two persons; one touching vehicle 1, one touching vehicle 2."""
        detections = [
            Detection(detection_id="v1", label="car", bbox=[100.0, 100.0, 300.0, 250.0], confidence=0.95, data_source="REAL"),
            Detection(detection_id="v2", label="bus", bbox=[400.0, 100.0, 600.0, 300.0], confidence=0.90, data_source="REAL"),
            Detection(detection_id="p1", label="person", bbox=[250.0, 150.0, 310.0, 260.0], confidence=0.82, data_source="REAL"), # overlaps car
            Detection(detection_id="p2", label="person", bbox=[410.0, 110.0, 450.0, 200.0], confidence=0.79, data_source="REAL"), # overlaps bus
            Detection(detection_id="p3", label="person", bbox=[700.0, 700.0, 730.0, 780.0], confidence=0.70, data_source="REAL"), # disjoint
        ]
        v_reports = [
            VehicleReport(vehicle_id="veh-car-01", vehicle_type="car", data_source="REAL"),
            VehicleReport(vehicle_id="veh-bus-02", vehicle_type="bus", data_source="REAL"),
        ]
        reports, summary = build_occupant_reports(detections, vehicle_reports=v_reports)
        assert len(reports) == 3
        assert summary["persons_visible_count"] == 3
        assert summary["persons_near_vehicles_count"] == 2

        # Verify assignments
        assert reports[0].vehicle_id == "veh-car-01"
        assert reports[0].overlap_ratio > 0.0
        assert reports[1].vehicle_id == "veh-bus-02"
        assert reports[1].overlap_ratio > 0.0
        assert reports[2].vehicle_id is None
        assert reports[2].overlap_ratio == 0.0

        # Verify strict limitations note on all
        for r in reports:
            assert r.limitation_note == DEFAULT_LIMITATION_NOTE
            assert r.injury_severity == "not determined"
            assert r.consciousness == "unknown"


class TestOccupantReportLive:
    """Live integration test on real sample images."""

    def test_live_occupant_generation(self):
        sample_files = list(SAMPLES_DIR.glob("*.jpg"))
        if not sample_files:
            pytest.skip("No real sample images found in data/samples/")

        img_path = sample_files[0]
        img = cv2.imread(str(img_path))
        detections = detect(img_path, conf_threshold=0.5)
        v_reports, _ = build_vehicle_reports(img_path, save_to_db=False)

        reports, summary = build_occupant_reports(detections, v_reports)
        assert summary["persons_visible_count"] >= 0
        assert summary["persons_near_vehicles_count"] <= summary["persons_visible_count"]
        for r in reports:
            assert isinstance(r, OccupantReport)
            assert r.data_source == "REAL"
            assert r.limitation_note == DEFAULT_LIMITATION_NOTE
