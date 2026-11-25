"""Regression tests for the shared four-tier accident evidence contract.

The contract is deliberately conservative: the raw classifier verdict stays a
diagnostic field, machine confirmation requires a distinct verified impact
record, and every category requires human review. The exact same assessment is
copied into AccidentEvent, IncidentReport, SQLite and the UI layers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.ml.accident_analyzer import analyze
from core.ml.evidence_assessment import (
    IMPLEMENTED_IMPACT_METHODS,
    build_accident_assessment,
    evaluate_temporal_impact,
)
from core.nlp.fusion import fuse_incident_evidence
from core.schemas import (
    FOUR_TIER_ASSESSMENT_CATEGORIES,
    STILL_IMAGE_SCENE_STATUS,
    AccidentEvent,
    EvidenceAssessment,
    IncidentReport,
)
from db.database import get_accident_events, get_incident_reports, save_accident_event, save_incident_report

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_IMAGE = REPO_ROOT / "data" / "samples" / "road_1.jpg"

VISUAL_FEATURES = {
    "vehicle_count": 2,
    "person_count": 1,
    "max_vehicle_iou": 0.32,
    "max_overlap_min": 0.80,
    "heavily_overlapping_pairs": 1,
    "vehicle_person_overlap_count": 1,
}


def _assessment(media_type: str, score: float, features: dict | None = None, **kwargs):
    return build_accident_assessment(
        model_verdict="LIKELY ACCIDENT" if score >= 0.60 else "NO ACCIDENT SIGNAL",
        model_score=score,
        media_type=media_type,
        features=features or {},
        **kwargs,
    )


def _event(assessment: EvidenceAssessment | None = None) -> AccidentEvent:
    return AccidentEvent(
        event_id="acc-assessment-test",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        location_name="Test roadway",
        latitude=12.9716,
        longitude=77.5946,
        severity="uncertain",
        vehicles_involved=2,
        casualties_estimated=0,
        status="reported",
        score=0.8,
        verdict="LIKELY ACCIDENT",
        evidence_assessment=assessment,
        data_source="REAL",
    )


class TestFourTierContract:
    def test_exact_four_categories_are_the_only_allowed_values(self) -> None:
        assert FOUR_TIER_ASSESSMENT_CATEGORIES == (
            "ACCIDENT EVIDENCE DETECTED",
            "POSSIBLE ACCIDENT",
            "INCONCLUSIVE",
            "NO CLEAR ACCIDENT EVIDENCE DETECTED",
        )
        assert tuple(EvidenceAssessment.model_fields["assessment_category"].annotation.__args__) == (
            FOUR_TIER_ASSESSMENT_CATEGORIES
        )

    def test_human_review_cannot_be_disabled(self) -> None:
        with pytest.raises(ValueError, match="human_review_required"):
            EvidenceAssessment(
                assessment_category="INCONCLUSIVE",
                human_review_required=False,
            )

    def test_still_image_high_score_and_overlap_is_only_possible_not_confirmed(self) -> None:
        assessment = _assessment("image", 0.91, VISUAL_FEATURES)
        assert assessment.assessment_category == "POSSIBLE ACCIDENT"
        assert assessment.scene_status == STILL_IMAGE_SCENE_STATUS
        assert assessment.model_verdict == "LIKELY ACCIDENT"
        assert assessment.human_review_required is True
        text = " ".join(assessment.evidence + assessment.contradictory_evidence).lower()
        assert "not proof" in text
        assert "no temporal impact evidence" in text

    def test_still_image_without_cues_is_inconclusive_not_road_safe(self) -> None:
        assessment = _assessment("image", 0.12)
        assert assessment.assessment_category == "INCONCLUSIVE"
        assert assessment.scene_status == STILL_IMAGE_SCENE_STATUS
        assert "road safety cannot be concluded" in " ".join(assessment.contradictory_evidence)

    def test_video_no_clear_evidence_is_not_a_safety_claim(self) -> None:
        temporal = {
            "temporal_signal": "NO SIGNAL",
            "high_frame_count": 0,
            "sustained_high_run": 0,
            "min_score": 0.1,
            "max_score": 0.3,
        }
        assessment = _assessment("video", 0.21, {}, temporal=temporal)
        assert assessment.assessment_category == "NO CLEAR ACCIDENT EVIDENCE DETECTED"
        assert assessment.temporal_evidence_available is True
        text = " ".join(assessment.contradictory_evidence + [assessment.reasoning]).lower()
        assert "safe" not in text or "not" in text
        assert "no accident" not in text or "not" in text

    def test_sustained_high_scores_are_not_validated_impact(self) -> None:
        assert IMPLEMENTED_IMPACT_METHODS == ()
        temporal = {
            "temporal_signal": "SUSTAINED HIGH SCORES",
            "high_frame_count": 5,
            "sustained_high_run": 4,
        }
        assessment = _assessment(
            "video", 0.95, VISUAL_FEATURES, temporal=temporal
        )
        assert assessment.assessment_category == "POSSIBLE ACCIDENT"
        assert evaluate_temporal_impact(temporal, VISUAL_FEATURES) is None

    def test_fake_impact_method_is_rejected(self) -> None:
        temporal = {
            "validated_impact_sequence": {
                "method": "guessed_from_overlap",
                "impact_detected": True,
                "frame_references": [1, 2],
            }
        }
        assert evaluate_temporal_impact(temporal, VISUAL_FEATURES) is None
        assert _assessment("video", 0.9, VISUAL_FEATURES, temporal=temporal).assessment_category != (
            "ACCIDENT EVIDENCE DETECTED"
        )

    def test_explicit_human_impact_record_can_confirm_and_still_requires_review(self) -> None:
        assessment = _assessment(
            "video",
            0.9,
            VISUAL_FEATURES,
            human_verified_impact={
                "impact_verified": True,
                "verified_by": "human reviewer A",
                "description": "Verified impact marks on reviewed frame pair",
            },
        )
        assert assessment.assessment_category == "ACCIDENT EVIDENCE DETECTED"
        assert assessment.scene_status == "HUMAN-VERIFIED IMPACT"
        assert assessment.human_review_required is True
        assert assessment.model_verdict == "LIKELY ACCIDENT"

    def test_analyze_attaches_the_shared_assessment(self) -> None:
        if not REAL_IMAGE.exists():
            pytest.skip(f"Real sample image missing: {REAL_IMAGE}")
        event = analyze(REAL_IMAGE, save_to_db=False)
        assert event.evidence_assessment is not None
        assert event.evidence_assessment.assessment_category in FOUR_TIER_ASSESSMENT_CATEGORIES
        assert event.evidence_assessment.human_review_required is True


class TestAssessmentPropagation:
    def test_fusion_copies_the_exact_assessment_into_report(self) -> None:
        assessment = _assessment("image", 0.91, VISUAL_FEATURES)
        event = _event(assessment)
        report = fuse_incident_evidence(
            accident_event=event,
            image_name="road_1.jpg",
            save_to_db=False,
            media_type="image",
        )
        assert report.evidence_assessment == assessment.as_dict()
        assert "[POSSIBLE ACCIDENT]" in report.summary_text
        assert "Original classifier verdict (separate diagnostic field): LIKELY ACCIDENT" in report.summary_text
        assert "Uncalibrated classifier score (not calibrated confidence): 0.9100" in report.summary_text

    def test_fusion_with_missing_assessment_does_not_invent_one(self) -> None:
        report = fuse_incident_evidence(
            accident_event=_event(None),
            image_name="road_1.jpg",
            save_to_db=False,
            media_type="image",
        )
        assert report.evidence_assessment is None
        assert "FOUR-TIER ACCIDENT EVIDENCE ASSESSMENT" not in report.summary_text

    def test_database_round_trips_exact_assessment(self, tmp_path: Path) -> None:
        db_path = tmp_path / "assessment.db"
        assessment = _assessment("video", 0.21, {}, temporal={"temporal_signal": "NO SIGNAL"})
        assessment_json = assessment.as_dict()

        event = _event(assessment)
        save_accident_event(event, db_path=db_path)
        stored_event = get_accident_events(db_path=db_path)[0]
        assert json.loads(stored_event["evidence_assessment"]) == assessment_json

        report = IncidentReport(
            report_id="report-assessment-test",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            verdict=event.verdict or "LIKELY ACCIDENT",
            summary_text="assessment round-trip",
            data_source="SIMULATED",
            evidence_assessment=assessment_json,
        )
        save_incident_report(report, db_path=db_path)
        stored_report = get_incident_reports(db_path=db_path)[0]
        assert json.loads(stored_report["evidence_assessment"]) == assessment_json
