"""Unit tests for multimodal evidence fusion and cross-modal discrepancy checks (Phase 4).

Validates:
1. Discrepancy flagging: caller says 2 vehicles, vision detects 6 -> flagged, both values retained.
2. Cross-modal alignment: matching counts produce no disagreement warning.
3. Provenance tagging (VISION, CALLER_SIMULATED, MISSING).
4. Persistence to SQLite incident_reports table.
"""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from core.nlp.caller_extraction import extract_caller_info
from core.nlp.fusion import fuse_incident_evidence
from core.schemas import AccidentEvent, CallerExtraction, IncidentReport, OccupantReport, VehicleReport
from db.database import get_incident_reports, init_db

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_caller_only_accident_claim_is_source_disagreement_not_verdict() -> None:
    """A transcript alone never upgrades a visual 'NO ACCIDENT SIGNAL' verdict."""
    caller = extract_caller_info("Caller: Two cars collided on the highway.")
    assert caller.accident_status == "reported"
    assert not caller.injury_keywords
    assert not caller.hazard_keywords

    event = AccidentEvent(
        event_id="acc-caller-only",
        timestamp=datetime.now(timezone.utc),
        location_name="Uploaded media",
        latitude=31.2142,
        longitude=75.7785,
        severity="none",
        vehicles_involved=2,
        casualties_estimated=0,
        status="not determined (out-of-domain still image)",
        confidence=0.2166,
        method="TRAINED_MODEL",
        score=0.2166,
        verdict="NO ACCIDENT SIGNAL",
        data_source="REAL",
    )
    report = fuse_incident_evidence(
        accident_event=event,
        caller_extraction=caller,
        image_name="two_car_collision.jpg",
        save_to_db=False,
    )
    assert report.verdict == "NO ACCIDENT SIGNAL"
    assert report.verdict_source == "VISION"
    assert any(
        "CALLER-REPORTED ACCIDENT — VISUAL MODEL DOES NOT CONFIRM" in item
        for item in report.source_disagreements
    )


class TestFusionDiscrepancyAndDatabase:
    """Test multimodal fusion and discrepancy handling."""

    def test_fusion_flags_disagreement_and_preserves_both(self, tmp_path):
        """When vision and caller disagree, both values must be displayed and flagged."""
        test_db = tmp_path / "fusion_test.db"
        init_db(test_db)

        # Vision detects 6 vehicles and 2 pedestrians in normal flow
        accident_event = AccidentEvent(
            event_id="acc-vis-01",
            timestamp=datetime.now(timezone.utc),
            location_name="MG Road",
            latitude=12.9716,
            longitude=77.5946,
            severity="none",
            vehicles_involved=6,
            verdict="NO ACCIDENT SIGNAL",
            score=0.31,
            data_source="REAL",
        )
        v_reports = [
            VehicleReport(vehicle_id=f"v-{i}", vehicle_type="car", data_source="REAL")
            for i in range(6)
        ]
        o_reports = [
            OccupantReport(occupant_id=f"occ-{i}", vehicle_id=None, data_source="REAL")
            for i in range(2)
        ]

        # Caller claims a catastrophic 2-car collision with 4 injured people
        caller_ext = CallerExtraction(
            call_id="call-contra-01",
            location="MG Road near Trinity",
            vehicle_count=2,  # Differs from vision (6)
            vehicle_types=["car"],
            people_involved=4,  # Differs from vision (2)
            injury_keywords=["bleeding", "unconscious"],
            hazard_keywords=["smoke"],
            urgency_keywords=["immediately"],
            raw_text="Two cars smashed together! Four people bleeding and unconscious. Smoke everywhere!",
            data_source="SIMULATED",
        )

        fused = fuse_incident_evidence(
            accident_event=accident_event,
            vehicle_reports=v_reports,
            occupant_reports=o_reports,
            caller_extraction=caller_ext,
            image_name="road_1.jpg",
            save_to_db=True,
            db_path=test_db,
        )

        assert isinstance(fused, IncidentReport)
        assert fused.data_source == "SIMULATED"

        # Disagreement verification: Both values must be preserved
        assert fused.vehicle_count_disagreement is True
        assert fused.vision_vehicle_count == 6
        assert fused.caller_vehicle_count == 2

        assert fused.people_count_disagreement is True
        assert fused.vision_people_count == 2
        assert fused.caller_people_count == 4

        # Disagreements list must have both warnings
        assert len(fused.disagreements) == 2
        assert any("VEHICLE COUNT DISAGREEMENT" in d for d in fused.disagreements)
        assert any("PEOPLE COUNT DISAGREEMENT" in d for d in fused.disagreements)

        # Summary text must include warning section
        assert "[!] DETECTED CROSS-MODAL DISAGREEMENTS [!]" in fused.summary_text
        assert "Vision detected 6" in fused.summary_text
        assert "Caller reported 2" in fused.summary_text

        # Verify SQLite persistence
        db_records = get_incident_reports(test_db)
        assert len(db_records) == 1
        row = db_records[0]
        assert row["report_id"] == fused.report_id
        assert row["vehicle_count_disagreement"] == 1
        assert row["vision_vehicle_count"] == 6
        assert row["caller_vehicle_count"] == 2
        assert row["data_source"] == "SIMULATED"

    def test_fusion_aligned_sources(self, tmp_path):
        """When vision and caller counts match, disagreement flags remain False."""
        test_db = tmp_path / "fusion_aligned.db"
        init_db(test_db)

        accident_event = AccidentEvent(
            event_id="acc-vis-aligned",
            timestamp=datetime.now(timezone.utc),
            location_name="Brigade Road",
            latitude=12.9716,
            longitude=77.5946,
            severity="moderate",
            vehicles_involved=2,
            verdict="LIKELY ACCIDENT",
            score=0.75,
            data_source="REAL",
        )
        v_reports = [
            VehicleReport(vehicle_id="v-1", vehicle_type="bus", data_source="REAL"),
            VehicleReport(vehicle_id="v-2", vehicle_type="motorcycle", data_source="REAL"),
        ]
        o_reports = [
            OccupantReport(occupant_id="occ-1", vehicle_id="v-2", data_source="REAL")
        ]

        caller_ext = CallerExtraction(
            call_id="call-align-01",
            location="Brigade Road",
            vehicle_count=2,  # Matches vision (2)
            vehicle_types=["bus", "motorcycle"],
            people_involved=1,  # Matches vision (1)
            injury_keywords=["pain"],
            hazard_keywords=[],
            urgency_keywords=["quickly"],
            raw_text="Bus and motorcycle accident at Brigade Road.",
            data_source="SIMULATED",
        )

        fused = fuse_incident_evidence(
            accident_event=accident_event,
            vehicle_reports=v_reports,
            occupant_reports=o_reports,
            caller_extraction=caller_ext,
            image_name="road_2.jpg",
            save_to_db=True,
            db_path=test_db,
        )

        assert fused.vehicle_count_disagreement is False
        assert fused.people_count_disagreement is False
        assert len(fused.disagreements) == 0
        assert "CROSS-MODAL ALIGNMENT" in fused.summary_text

    def test_fused_report_matches_exact_vehicle_report_attributes(self):
        """REGRESSION TEST: Verify fused report reflects exact real VehicleReport attributes.

        Guarantees that vehicle colors, color shares, vehicle IDs, and license plates
        from Phase 2 are preserved exactly in the fused incident summary without mock or fabricated values.
        """
        from core.cv.vehicle_report import build_vehicle_reports
        from core.ml.accident_analyzer import analyze
        from core.nlp.occupant_report import build_occupant_reports
        from core.nlp.caller_extraction import extract_caller_info

        sample_img = REPO_ROOT / "data" / "samples" / "road_1.jpg"
        assert sample_img.exists()

        v_reports, _ = build_vehicle_reports(sample_img, save_to_db=False)
        assert len(v_reports) == 6

        accident_event = analyze(sample_img, save_to_db=False)
        from core.cv.detector import detect
        dets = detect(sample_img, conf_threshold=0.5)
        o_reports, _ = build_occupant_reports(dets, v_reports)

        call_file = REPO_ROOT / "data" / "sample_calls" / "call_01.txt"
        caller_ext = extract_caller_info(call_file.read_text(encoding="utf-8"), call_id="call_01")

        fused = fuse_incident_evidence(
            accident_event=accident_event,
            vehicle_reports=v_reports,
            occupant_reports=o_reports,
            caller_extraction=caller_ext,
            image_name="road_1.jpg",
            save_to_db=False,
        )

        # Compare every single vehicle attribute against summary_text
        for veh in v_reports:
            # Vehicle ID must be present
            assert veh.vehicle_id in fused.summary_text, f"Vehicle ID {veh.vehicle_id} missing from fused summary"
            # Vehicle type must be present
            assert veh.vehicle_type in fused.summary_text, f"Vehicle type {veh.vehicle_type} missing from summary"
            # Dominant color must be present
            assert veh.color in fused.summary_text, f"Color {veh.color} missing from summary"
            # Color share percentage formatted to 1 decimal place must be present
            share_str = f"{veh.color_share * 100:.1f}%"
            assert share_str in fused.summary_text, f"Color share {share_str} missing from summary"

            # License plate check
            if veh.license_plate:
                assert veh.license_plate in fused.summary_text, f"Plate {veh.license_plate} missing from summary"
                if veh.plate_confidence is not None and veh.plate_confidence < 0.70:
                    assert "unverified" in fused.summary_text
            else:
                assert "unreadable / none visible" in fused.summary_text


def test_fusion_preserves_sampled_video_evidence_without_overriding_verdict(tmp_path) -> None:
    """Phase 14 video evidence is retained verbatim and stays explanatory."""
    event = AccidentEvent(
        event_id="acc-video-fusion",
        timestamp=datetime.now(timezone.utc),
        location_name="Monitored Roadway",
        latitude=12.0,
        longitude=77.0,
        severity="uncertain",
        verdict="UNCERTAIN",
        score=0.52,
        data_source="REAL",
    )
    video_evidence = {
        "metadata": {"frame_count": 525, "fps": 25.0, "duration_s": 21.0},
        "frame_rows": [{"Frame": 1, "Score": 0.52}],
        "temporal": {
            "temporal_signal": "ISOLATED HIGH FRAME IN SAMPLE",
            "min_score": 0.20,
            "max_score": 0.52,
            "mean_score": 0.36,
            "high_frame_count": 0,
            "uncertain_frame_count": 1,
            "no_signal_frame_count": 1,
            "sustained_high_run": 0,
            "sustained_uncertain_or_high_run": 1,
        },
        "assessment_text": (
            "The trained model verdict remains UNCERTAIN. Temporal evidence is "
            "explanatory only and does not override it. Not an accident confirmation."
        ),
    }

    fused = fuse_incident_evidence(
        accident_event=event,
        image_name="road_video.mp4",
        media_type="video",
        video_evidence=video_evidence,
        save_to_db=True,
        db_path=tmp_path / "video_fusion.db",
    )

    assert fused.verdict == "UNCERTAIN"
    assert fused.video_evidence == video_evidence
    assert fused.video_temporal_signal == "ISOLATED HIGH FRAME IN SAMPLE"
    assert "REAL VIDEO FRAME/TEMPORAL EVIDENCE" in fused.summary_text
    assert "525" in fused.summary_text
    assert "ISOLATED HIGH FRAME IN SAMPLE" in fused.summary_text
    assert "does not override it" in fused.summary_text
    assert "no accident confirmation" in fused.summary_text.lower()

    stored = get_incident_reports(tmp_path / "video_fusion.db")[0]
    assert stored["video_temporal_signal"] == "ISOLATED HIGH FRAME IN SAMPLE"
    assert stored["video_assessment"] == fused.video_assessment
    assert '"frame_count": 525' in stored["video_evidence"]


def test_fusion_forwards_still_image_media_type_to_narrative() -> None:
    """A still upload must not be described as confirmed normal traffic in the fused report."""
    event = AccidentEvent(
        event_id="acc-still-fusion",
        timestamp=datetime.now(timezone.utc),
        location_name="Uploaded still",
        latitude=12.0,
        longitude=77.0,
        severity="none",
        vehicles_involved=4,
        verdict="NO ACCIDENT SIGNAL",
        score=0.2166,
        data_source="REAL",
    )
    fused = fuse_incident_evidence(
        accident_event=event,
        vehicle_reports=[],
        occupant_reports=[],
        caller_extraction=None,
        image_name="uploaded_photo.jpg",
        save_to_db=False,
        media_type="image",
    )
    assert "normal traffic cannot be confirmed" in fused.summary_text
    assert "normal traffic flow" not in fused.summary_text
