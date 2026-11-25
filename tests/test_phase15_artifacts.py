"""Phase 15 final artifact and provenance checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from db.database import EXPECTED_TABLES, get_connection, init_db

ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verified_offline_model_artifacts_are_present() -> None:
    onnx = ROOT / "models" / "plate_detector" / "license_plate_detection_lpd_yunet_2023mar.onnx"
    license_file = ROOT / "models" / "plate_detector" / "LICENSE"

    assert onnx.is_file()
    assert _sha256(onnx) == (
        "6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7"
    )
    assert license_file.is_file()
    assert "Apache License" in license_file.read_text(encoding="utf-8")

    craft = ROOT / "models" / "easyocr" / "craft_mlt_25k.pth"
    english = ROOT / "models" / "easyocr" / "english_g2.pth"
    assert _sha256(craft) == (
        "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17"
    )
    assert _sha256(english) == (
        "e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8"
    )


def test_metrics_and_evaluation_artifacts_are_consistent() -> None:
    metrics = json.loads((ROOT / "models" / "accident_classifier_metrics.json").read_text())
    evaluation = json.loads((ROOT / "models" / "accident_evaluation_report.json").read_text())

    assert evaluation["matches_published_metrics"] is True
    assert evaluation["published_held_out_metrics"]["roc_auc"] == metrics["test_roc_auc"]
    assert evaluation["reproduced_held_out_raw"]["confusion_matrix"] == metrics["confusion_matrix"]


def test_routing_and_database_assets_exist() -> None:
    assets = (
        ROOT / "data" / "hospitals_osm.json",
        ROOT / "data" / "phagwara_corridor.osm",
        ROOT / "data" / "phagwara_drive_network.graphml",
        ROOT / "config" / "demo_incident_locations.json",
        ROOT / "config" / "simulated_ambulances.json",
    )
    for asset in assets:
        assert asset.is_file() and asset.stat().st_size > 0, asset


def test_database_schema_contains_phase12_and_phase14_provenance(tmp_path: Path) -> None:
    db_path = tmp_path / "final_schema.db"
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert set(EXPECTED_TABLES) <= tables

        vehicle_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(vehicle_reports)")
        }
        incident_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(incident_reports)")
        }
        assert {
            "plate_detection_status",
            "plate_bbox",
            "plate_detector",
        } <= vehicle_columns
        assert {
            "video_evidence",
            "video_assessment",
            "video_temporal_signal",
        } <= incident_columns
    finally:
        conn.close()
