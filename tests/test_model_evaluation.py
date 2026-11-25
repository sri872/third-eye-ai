"""Phase 13 tests for the reproducible accident-model evaluation artifact."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "models" / "accident_evaluation_report.json"
METRICS_PATH = ROOT / "models" / "accident_classifier_metrics.json"


def test_evaluation_artifact_reproduces_published_held_out_metrics() -> None:
    report = json.loads(REPORT_PATH.read_text())
    published = json.loads(METRICS_PATH.read_text())

    assert report["matches_published_metrics"] is True
    reproduced = report["reproduced_held_out_raw"]
    assert reproduced["confusion_matrix"] == [[56, 34], [24, 66]]
    assert reproduced["accuracy"] == published["test_accuracy"]
    assert reproduced["precision"] == published["test_precision"]
    assert reproduced["recall"] == published["test_recall"]
    assert reproduced["f1"] == published["test_f1"]
    assert reproduced["roc_auc"] == published["test_roc_auc"]


def test_fpr_fnr_and_video_level_split_are_documented() -> None:
    report = json.loads(REPORT_PATH.read_text())
    reproduced = report["reproduced_held_out_raw"]

    assert reproduced["false_positive_rate"] == 0.3778
    assert reproduced["false_negative_rate"] == 0.2667
    assert all(
        not values for values in report["dataset"]["video_id_intersections"].values()
    )
    assert report["dataset"]["frame_shapes"] == [[224, 224]]


def test_calibration_is_validation_only_and_not_deployed() -> None:
    report = json.loads(REPORT_PATH.read_text())
    assert "NOT deployed" in report["calibration_decision"]
    assert "UNCALIBRATED" in report["calibration_decision"]
    assert report["held_out_test_platt_validation_fit"]["roc_auc"] == 0.7452
    assert report["geometric_rule_evaluator"]["runtime_use"] is False


def test_duplicate_audit_documents_leakage_risk() -> None:
    report = json.loads(REPORT_PATH.read_text())
    duplicates = report["duplicates"]
    assert duplicates["exact_duplicate_groups"] == 0
    assert duplicates["cross_split_near_duplicate_pairs"] > 0


def test_deployment_bands_are_descriptive_only() -> None:
    report = json.loads(REPORT_PATH.read_text())
    bands = report["deployment_band_counts"]
    assert bands["scope"] == "held_out_test_frames"
    assert "not tuned" in bands["note"]
    likely = bands["likely_accident_ge_0.60"]
    uncertain = bands["uncertain_0.40_to_0.60"]
    no_signal = bands["no_signal_lt_0.40"]
    total = (
        likely["true_positive"]
        + likely["false_positive"]
        + uncertain["positive"]
        + uncertain["negative"]
        + no_signal["false_negative"]
        + no_signal["true_negative"]
    )
    assert total == 180
