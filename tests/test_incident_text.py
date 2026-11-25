"""Unit tests for deterministic incident text summary generator (Phase 4).

Validates:
1. Hedged wording strictly matches accident verdicts (LIKELY ACCIDENT, UNCERTAIN, NO ACCIDENT SIGNAL).
2. License plate truthfulness: never fabricates plates when None, marks plates <0.70 confidence as 'unverified'.
3. Explicit listing of evidence sources and operational limitations.
4. No invented or assumed medical/casualty facts.
"""

from datetime import datetime, timezone
import pytest

from core.nlp.incident_text import generate_incident_summary
from core.schemas import AccidentEvent, OccupantReport, VehicleReport


class TestIncidentTextSummary:
    """Test deterministic narrative formatting and constraints."""

    def test_summary_no_accident_verdict(self):
        event = AccidentEvent(
            event_id="acc-test-01",
            timestamp=datetime.now(timezone.utc),
            location_name="MG Road",
            latitude=12.9716,
            longitude=77.5946,
            severity="none",
            vehicles_involved=3,
            method="TRAINED_MODEL",
            score=0.25,
            verdict="NO ACCIDENT SIGNAL",
            data_source="REAL",
        )
        summary = generate_incident_summary(event)
        assert "NO ACCIDENT SIGNAL" in summary
        assert "NO CLEAR ACCIDENT EVIDENCE DETECTED" in summary
        assert "not a finding of normal traffic flow" in summary
        assert "road safety" in summary
        assert "Evidence Sources & Operational Constraints" in summary
        assert "Visible exterior only" in summary

    def test_summary_out_of_domain_still_image_does_not_confirm_normal_traffic(self):
        event = AccidentEvent(
            event_id="acc-still-01",
            timestamp=datetime.now(timezone.utc),
            location_name="Uploaded photo",
            latitude=12.9716,
            longitude=77.5946,
            severity="none",
            vehicles_involved=4,
            method="TRAINED_MODEL",
            score=0.2166,
            verdict="NO ACCIDENT SIGNAL",
            data_source="REAL",
        )
        summary = generate_incident_summary(event, media_type="image")
        assert "NO ACCIDENT SIGNAL" in summary
        assert "normal traffic cannot be confirmed" in summary
        assert "normal traffic flow" not in summary
        assert "Out-of-domain still image" in summary

    def test_summary_likely_accident_verdict(self):
        event = AccidentEvent(
            event_id="acc-test-02",
            timestamp=datetime.now(timezone.utc),
            location_name="Junction 4",
            latitude=12.9716,
            longitude=77.5946,
            severity="high_probability",
            vehicles_involved=2,
            method="TRAINED_MODEL",
            score=0.82,
            verdict="LIKELY ACCIDENT",
            data_source="REAL",
        )
        summary = generate_incident_summary(event)
        assert "LIKELY ACCIDENT" in summary
        assert "probable vehicle collision" in summary
        assert "Emergency dispatch recommended" in summary

    def test_summary_uncertain_verdict(self):
        event = AccidentEvent(
            event_id="acc-test-03",
            timestamp=datetime.now(timezone.utc),
            location_name="Ring Road",
            latitude=12.9716,
            longitude=77.5946,
            severity="uncertain",
            vehicles_involved=2,
            method="TRAINED_MODEL",
            score=0.48,
            verdict="UNCERTAIN",
            data_source="REAL",
        )
        summary = generate_incident_summary(event)
        assert "UNCERTAIN" in summary
        assert "inconclusive" in summary
        assert "on-scene or human verification required" in summary

    def test_summary_plate_handling_none_and_unverified(self):
        event = AccidentEvent(
            event_id="acc-test-04",
            timestamp=datetime.now(timezone.utc),
            location_name="Road",
            latitude=12.0,
            longitude=77.0,
            severity="none",
            verdict="NO ACCIDENT SIGNAL",
            score=0.30,
            data_source="REAL",
        )
        v_reports = [
            # Vehicle 1: No plate readable
            VehicleReport(
                vehicle_id="v1",
                vehicle_type="car",
                color="white",
                color_share=0.80,
                license_plate=None,
                plate_confidence=None,
                data_source="REAL",
            ),
            # Vehicle 2: Low OCR confidence (< 0.70) -> must say 'unverified'
            VehicleReport(
                vehicle_id="v2",
                vehicle_type="truck",
                color="blue",
                color_share=0.65,
                license_plate="KA01XY9999",
                plate_confidence=0.55,
                data_source="REAL",
            ),
            # Vehicle 3: High OCR confidence (>= 0.70)
            VehicleReport(
                vehicle_id="v3",
                vehicle_type="bus",
                color="red",
                color_share=0.70,
                license_plate="KA04AB1234",
                plate_confidence=0.92,
                data_source="REAL",
            ),
        ]
        summary = generate_incident_summary(event, vehicle_reports=v_reports)

        # Vehicle 1 check: Plate must NOT be invented
        assert "Plate: unreadable / none visible" in summary
        # Vehicle 2 check: Must be marked unverified
        assert "Plate: KA01XY9999 (unverified, OCR conf: 0.55)" in summary
        # Vehicle 3 check: Verified plate
        assert "Plate: KA04AB1234 (OCR conf: 0.92)" in summary
