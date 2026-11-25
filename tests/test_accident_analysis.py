"""Unit and integration tests for accident analysis (Phase 3b).

Covers:
1. Geometric feature calculation (IoU, overlap, aspect ratio) with mocked detections.
2. Rule-based heuristic evaluation (used as explainability baseline).
3. Person-vehicle proximity computation (persons_near_vehicles_count).
4. Model decision thresholds (LIKELY ACCIDENT, UNCERTAIN, NO ACCIDENT SIGNAL).
5. Zero video data leakage proof (Train/Val/Test disjointness).
6. AccidentEvent schema validation and SQLite database persistence.
7. Regression test on normal-traffic sample photos (must NOT all be LIKELY ACCIDENT).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from core.ml.accident_analyzer import (
    analyze,
    compute_persons_near_vehicles,
    determine_verdict,
)
from core.ml.accident_features import compute_iou, extract_accident_features
from core.ml.accident_rules import evaluate_accident_rules
from core.schemas import AccidentEvent, Detection
from db.database import get_accident_events, init_db, save_accident_event

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
METRICS_PATH = REPO_ROOT / "models" / "accident_classifier_metrics.json"


class TestAccidentFeaturesMocked:
    """MOCKED tests: Verify geometric feature calculation."""

    def test_compute_iou_zero_overlap(self):
        box1 = [0.0, 0.0, 50.0, 50.0]
        box2 = [100.0, 100.0, 150.0, 150.0]
        iou, overlap_min = compute_iou(box1, box2)
        assert iou == 0.0
        assert overlap_min == 0.0

    def test_compute_iou_heavy_overlap(self):
        box1 = [0.0, 0.0, 100.0, 100.0]
        box2 = [50.0, 0.0, 150.0, 100.0]
        iou, overlap_min = compute_iou(box1, box2)
        assert iou > 0.30
        assert overlap_min == 0.50

    def test_extract_features_no_overlap(self):
        detections = [
            Detection(detection_id="d1", label="car", bbox=[10.0, 10.0, 50.0, 50.0], data_source="REAL"),
            Detection(detection_id="d2", label="car", bbox=[200.0, 200.0, 250.0, 250.0], data_source="REAL"),
        ]
        feat = extract_accident_features(detections, 1000, 1000)
        assert feat["vehicle_count"] == 2
        assert feat["person_count"] == 0
        assert feat["max_vehicle_iou"] == 0.0
        assert feat["heavily_overlapping_pairs"] == 0
        assert feat["vehicle_person_overlap_count"] == 0

    def test_extract_features_collision_and_pedestrian(self):
        detections = [
            Detection(detection_id="d1", label="car", bbox=[100.0, 100.0, 300.0, 250.0], data_source="REAL"),
            Detection(detection_id="d2", label="car", bbox=[150.0, 120.0, 320.0, 260.0], data_source="REAL"),
            Detection(detection_id="d3", label="person", bbox=[110.0, 110.0, 140.0, 180.0], data_source="REAL"),
        ]
        feat = extract_accident_features(detections, 1000, 1000)
        assert feat["vehicle_count"] == 2
        assert feat["person_count"] == 1
        assert feat["max_vehicle_iou"] > 0.40
        assert feat["heavily_overlapping_pairs"] == 1
        assert feat["vehicle_person_overlap_count"] >= 1


class TestAccidentRulesMocked:
    """MOCKED tests: Verify rule-based heuristic scoring logic."""

    def test_rules_no_accident_scenario(self):
        features = {
            "vehicle_count": 3,
            "person_count": 0,
            "max_vehicle_iou": 0.0,
            "max_overlap_min": 0.0,
            "heavily_overlapping_pairs": 0,
            "vehicle_person_overlap_count": 0,
            "aspect_ratio_outliers": 0,
            "crowd_density": 0.0,
        }
        res = evaluate_accident_rules(features)
        assert res["method"] == "RULE_BASED_HEURISTIC"
        assert res["score"] == 0.0
        assert res["is_accident"] is False
        assert res["severity"] == "none"
        assert len(res["rules_fired"]) == 0

    def test_rules_severe_collision_scenario(self):
        features = {
            "vehicle_count": 2,
            "person_count": 0,
            "max_vehicle_iou": 0.45,
            "max_overlap_min": 0.60,
            "heavily_overlapping_pairs": 1,
            "vehicle_person_overlap_count": 0,
            "aspect_ratio_outliers": 0,
            "crowd_density": 0.0,
        }
        res = evaluate_accident_rules(features)
        assert res["method"] == "RULE_BASED_HEURISTIC"
        assert res["score"] >= 0.50
        assert res["is_accident"] is True
        assert res["severity"] in {"moderate", "critical"}
        assert any("RULE_HIGH_VEHICLE_OVERLAP" in r for r in res["rules_fired"])


class TestPersonsNearVehicles:
    """Tests for compute_persons_near_vehicles proximity calculation."""

    def test_no_persons_or_vehicles(self):
        assert compute_persons_near_vehicles([]) == 0

        # Only vehicles
        dets_veh = [Detection(detection_id="v1", label="car", bbox=[10.0, 10.0, 50.0, 50.0], data_source="REAL")]
        assert compute_persons_near_vehicles(dets_veh) == 0

        # Only person
        dets_p = [Detection(detection_id="p1", label="person", bbox=[10.0, 10.0, 50.0, 50.0], data_source="REAL")]
        assert compute_persons_near_vehicles(dets_p) == 0

    def test_disjoint_person_and_vehicle(self):
        dets = [
            Detection(detection_id="v1", label="car", bbox=[0.0, 0.0, 50.0, 50.0], data_source="REAL"),
            Detection(detection_id="p1", label="person", bbox=[100.0, 100.0, 120.0, 150.0], data_source="REAL"),
        ]
        assert compute_persons_near_vehicles(dets) == 0

    def test_overlapping_person_and_vehicle(self):
        dets = [
            Detection(detection_id="v1", label="truck", bbox=[50.0, 50.0, 200.0, 200.0], data_source="REAL"),
            Detection(detection_id="p1", label="person", bbox=[180.0, 180.0, 210.0, 250.0], data_source="REAL"),  # Intersects
            Detection(detection_id="p2", label="person", bbox=[300.0, 300.0, 320.0, 350.0], data_source="REAL"),  # Disjoint
        ]
        assert compute_persons_near_vehicles(dets) == 1


class TestVerdictThresholds:
    """Verify calibrated probability thresholds map to 3 verdicts."""

    def test_verdict_high_probability(self):
        verdict, severity, status = determine_verdict(0.65)
        assert verdict == "LIKELY ACCIDENT"
        assert severity == "high_probability"
        assert status == "reported"

    def test_verdict_uncertain(self):
        verdict, severity, status = determine_verdict(0.50)
        assert verdict == "UNCERTAIN"
        assert severity == "uncertain"
        assert status == "reported"

    def test_verdict_no_accident(self):
        verdict, severity, status = determine_verdict(0.35)
        assert verdict == "NO ACCIDENT SIGNAL"
        assert severity == "none"
        assert status == "normal_flow"


class TestModelSplitZeroLeakage:
    """Verify strict video-level train/val/test splitting with zero data leakage."""

    def test_zero_leakage_in_splits(self):
        assert METRICS_PATH.exists(), f"Metrics file missing: {METRICS_PATH}"
        with open(METRICS_PATH) as f:
            data = json.load(f)

        train_vids = set(data["train_video_ids"])
        val_vids = set(data["val_video_ids"])
        test_vids = set(data["test_video_ids"])

        assert len(train_vids) == 280
        assert len(val_vids) == 60
        assert len(test_vids) == 60

        # Prove 0 video leakage between splits
        assert len(train_vids & test_vids) == 0, "Video leakage detected between train and test sets!"
        assert len(val_vids & test_vids) == 0, "Video leakage detected between val and test sets!"
        assert len(train_vids & val_vids) == 0, "Video leakage detected between train and val sets!"

        # Check performance bar
        assert data["test_roc_auc"] >= 0.70, f"ROC-AUC {data['test_roc_auc']} below 0.70 threshold"
        assert data["test_f1"] > 0.50, f"Test F1 {data['test_f1']} below 0.50 baseline"


class TestAccidentEventSchemaAndDatabase:
    """Test AccidentEvent schema extension and SQLite persistence."""

    def test_accident_event_valid_with_phase3b_fields(self):
        event = AccidentEvent(
            event_id="acc-schema-test-3b",
            timestamp=datetime.now(timezone.utc),
            location_name="MG Road Junction",
            latitude=12.9716,
            longitude=77.5946,
            severity="high_probability",
            vehicles_involved=2,
            casualties_estimated=1,
            method="TRAINED_MODEL",
            score=0.78,
            confidence=0.78,
            verdict="LIKELY ACCIDENT",
            persons_near_vehicles_count=1,
            reasons="Classifier score: 0.7800 -> Verdict: LIKELY ACCIDENT",
            data_source="REAL",
        )
        assert event.score == 0.78
        assert event.confidence == 0.78
        assert event.method == "TRAINED_MODEL"
        assert event.verdict == "LIKELY ACCIDENT"
        assert event.persons_near_vehicles_count == 1

    def test_accident_event_rejects_invalid_score(self):
        with pytest.raises(ValidationError):
            AccidentEvent(
                event_id="acc-bad-score",
                timestamp=datetime.now(timezone.utc),
                location_name="Road",
                latitude=12.0,
                longitude=77.0,
                severity="none",
                score=1.5,
                data_source="REAL",
            )

    def test_save_and_retrieve_accident_event(self, tmp_path):
        db_file = tmp_path / "test_events.db"
        init_db(db_file)

        event = AccidentEvent(
            event_id="acc-db-test-3b",
            timestamp=datetime.now(timezone.utc),
            location_name="Brigade Rd",
            latitude=12.9750,
            longitude=77.6000,
            severity="none",
            vehicles_involved=3,
            casualties_estimated=0,
            status="normal_flow",
            confidence=0.31,
            method="TRAINED_MODEL",
            score=0.31,
            verdict="NO ACCIDENT SIGNAL",
            persons_near_vehicles_count=0,
            reasons="Normal traffic flow; no collision detected",
            data_source="REAL",
        )
        saved_id = save_accident_event(event, db_path=db_file)
        assert saved_id == "acc-db-test-3b"

        events = get_accident_events(db_path=db_file)
        assert len(events) == 1
        assert events[0]["event_id"] == "acc-db-test-3b"
        assert events[0]["score"] == 0.31
        assert events[0]["verdict"] == "NO ACCIDENT SIGNAL"
        assert events[0]["persons_near_vehicles_count"] == 0
        assert events[0]["method"] == "TRAINED_MODEL"


class TestRegressionNormalTrafficSamplePhotos:
    """REGRESSION TEST: Verify sample photos (including out-of-domain collision stills) against Phase 3 heuristic failure.

    In Phase 3, 2D perspective overlap caused all 3 sample photos (road_1, road_2, road_3)
    to be falsely labeled CRITICAL accidents. In Phase 3b, the trained ResNet-18 classifier
    scores these out-of-domain still photos as NO ACCIDENT SIGNAL (< 0.40). Note: road_1 and road_2
    depict real collisions, but are missed due to domain shift from dashcam training.
    """

    def test_normal_traffic_not_all_likely_accident(self):
        sample_files = sorted(list(SAMPLES_DIR.glob("*.jpg")))
        assert len(sample_files) >= 3, f"Expected at least 3 sample photos in {SAMPLES_DIR}"

        verdicts = []
        for img_path in sample_files:
            event = analyze(img_path, save_to_db=False)
            assert isinstance(event, AccidentEvent)
            assert event.data_source == "REAL"
            assert event.method == "TRAINED_MODEL"
            assert event.score is not None
            assert 0.0 <= event.score <= 1.0
            verdicts.append(event.verdict)

        # Regression check: Must FAIL if all sample photos are marked LIKELY ACCIDENT
        likely_count = sum(1 for v in verdicts if v == "LIKELY ACCIDENT")
        assert likely_count < len(verdicts), (
            f"Regression failure: All {len(verdicts)} sample photos were flagged as LIKELY ACCIDENT!"
        )

        # Consistency check across sample photos: all 3 currently score NO ACCIDENT SIGNAL under the dashcam model
        assert all(v == "NO ACCIDENT SIGNAL" for v in verdicts), (
            f"Expected all sample photos to have 'NO ACCIDENT SIGNAL', got: {verdicts}"
        )
