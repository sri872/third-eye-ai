"""Tests for Pydantic shared schemas in core/schemas.py.

Validates that all 8 schemas enforce constraints, require explicit data_source ('REAL' or 'SIMULATED'),
and treat confidence as strictly optional with valid range limits.
"""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from core.schemas import (
    AccidentEvent,
    Ambulance,
    AuditReport,
    Detection,
    Hospital,
    OccupantReport,
    RouteResult,
    VehicleReport,
)


class TestSchemasValidation:
    """Test valid instantiation with REAL and SIMULATED sources."""

    def test_detection_valid(self):
        det_real = Detection(
            detection_id="det-001",
            label="car",
            bbox=[10.0, 20.0, 100.0, 150.0],
            confidence=0.92,
            data_source="REAL",
            timestamp=datetime.now(timezone.utc)
        )
        assert det_real.data_source == "REAL"
        assert det_real.confidence == 0.92

        det_sim = Detection(
            detection_id="det-002",
            label="truck",
            bbox=[0.0, 0.0, 50.0, 50.0],
            confidence=None,  # Optional confidence
            data_source="SIMULATED"
        )
        assert det_sim.data_source == "SIMULATED"
        assert det_sim.confidence is None

    def test_vehicle_report_valid(self):
        veh = VehicleReport(
            vehicle_id="veh-101",
            license_plate="KA01AB1234",
            plate_confidence=0.94,
            vehicle_type="SUV",
            color="white",
            color_share=0.78,
            damage_severity="severe",
            confidence=0.88,
            data_source="REAL"
        )
        assert veh.vehicle_id == "veh-101"
        assert veh.license_plate == "KA01AB1234"
        assert veh.color == "white"
        assert veh.color_share == 0.78
        assert veh.data_source == "REAL"

    def test_occupant_report_valid(self):
        occ = OccupantReport(
            occupant_id="occ-201",
            vehicle_id="veh-101",
            estimated_count=2,
            injury_severity="critical",
            consciousness="unconscious",
            data_source="SIMULATED"
        )
        assert occ.estimated_count == 2
        assert occ.confidence is None
        assert occ.data_source == "SIMULATED"

    def test_accident_event_valid(self):
        event = AccidentEvent(
            event_id="acc-5001",
            timestamp=datetime(2026, 11, 24, 10, 0, 0, tzinfo=timezone.utc),
            location_name="MG Road & Brigade Road Junction",
            latitude=12.9716,
            longitude=77.5946,
            severity="critical",
            vehicles_involved=2,
            casualties_estimated=3,
            status="reported",
            confidence=0.95,
            data_source="REAL"
        )
        assert event.latitude == 12.9716
        assert event.severity == "critical"
        assert event.data_source == "REAL"

    def test_hospital_valid(self):
        hosp = Hospital(
            hospital_id="hosp-01",
            name="City Trauma & Emergency Center",
            latitude=12.9750,
            longitude=77.6000,
            total_beds=100,
            available_beds=25,
            icu_beds_available=5,
            trauma_center_level=1,
            contact_number="+91-80-2222-3333",
            data_source="REAL"
        )
        assert hosp.available_beds == 25
        assert hosp.icu_beds_available == 5
        assert hosp.data_source == "REAL"

    def test_ambulance_valid(self):
        amb = Ambulance(
            ambulance_id="amb-42",
            call_sign="EMS-ALPHA-1",
            status="available",
            latitude=12.9650,
            longitude=77.5900,
            equipment_level="Critical Care",
            confidence=None,
            data_source="SIMULATED"
        )
        assert amb.call_sign == "EMS-ALPHA-1"
        assert amb.data_source == "SIMULATED"

    def test_route_result_valid(self):
        route = RouteResult(
            route_id="route-99",
            ambulance_id="amb-42",
            hospital_id="hosp-01",
            accident_id="acc-5001",
            distance_km=4.8,
            estimated_time_minutes=11.5,
            route_waypoints=[[12.9650, 77.5900], [12.9716, 77.5946], [12.9750, 77.6000]],
            traffic_condition="moderate",
            data_source="SIMULATED"
        )
        assert route.distance_km == 4.8
        assert route.data_source == "SIMULATED"

    def test_audit_report_valid(self):
        audit = AuditReport(
            audit_id="audit-777",
            accident_id="acc-5001",
            generated_at=datetime.now(timezone.utc),
            collision_analysis="Head-on collision at 45 km/h",
            liability_assessment="Vehicle 2 failed to yield",
            regulatory_compliance="Compliant with MoRTH standards",
            timeline_summary="T0: Collision; T+2m: Detected; T+5m: Dispatched",
            confidence=0.89,
            data_source="REAL"
        )
        assert audit.audit_id == "audit-777"
        assert audit.data_source == "REAL"


class TestSchemasRejection:
    """Test that schemas reject invalid data, bad sources, and boundary violations."""

    @pytest.mark.parametrize("schema_cls,sample_kwargs", [
        (Detection, {"detection_id": "d1", "label": "car", "bbox": [0, 0, 10, 10]}),
        (VehicleReport, {"vehicle_id": "v1"}),
        (OccupantReport, {"occupant_id": "o1"}),
        (AccidentEvent, {
            "event_id": "e1",
            "timestamp": datetime.now(timezone.utc),
            "location_name": "Test St",
            "latitude": 0.0,
            "longitude": 0.0,
            "severity": "minor"
        }),
        (Hospital, {
            "hospital_id": "h1",
            "name": "Test Hosp",
            "latitude": 0.0,
            "longitude": 0.0,
            "total_beds": 10,
            "available_beds": 5,
            "icu_beds_available": 2
        }),
        (Ambulance, {"ambulance_id": "a1", "call_sign": "C1", "latitude": 0.0, "longitude": 0.0}),
        (RouteResult, {
            "route_id": "r1",
            "ambulance_id": "a1",
            "hospital_id": "h1",
            "accident_id": "e1",
            "distance_km": 1.0,
            "estimated_time_minutes": 2.0
        }),
        (AuditReport, {
            "audit_id": "au1",
            "accident_id": "e1",
            "generated_at": datetime.now(timezone.utc)
        })
    ])
    def test_reject_invalid_data_source(self, schema_cls, sample_kwargs):
        # Missing data_source must fail
        with pytest.raises(ValidationError):
            schema_cls(**sample_kwargs)

        # Invalid data_source values (e.g., MOCK, FAKE, GENERATED) must fail
        for bad_source in ["MOCK", "FAKE", "SYNTHETIC", "UNKNOWN", "", 123]:
            with pytest.raises(ValidationError):
                schema_cls(**sample_kwargs, data_source=bad_source)

    def test_reject_invalid_confidence(self):
        # Confidence must be between 0.0 and 1.0
        with pytest.raises(ValidationError):
            Detection(
                detection_id="d1",
                label="car",
                bbox=[0, 0, 10, 10],
                confidence=1.5,
                data_source="REAL"
            )
        with pytest.raises(ValidationError):
            Detection(
                detection_id="d1",
                label="car",
                bbox=[0, 0, 10, 10],
                confidence=-0.1,
                data_source="REAL"
            )

    def test_reject_malformed_bbox(self):
        # xmin > xmax
        with pytest.raises(ValidationError):
            Detection(
                detection_id="d1",
                label="car",
                bbox=[100.0, 10.0, 50.0, 100.0],
                data_source="REAL"
            )
        # Not 4 coordinates
        with pytest.raises(ValidationError):
            Detection(
                detection_id="d1",
                label="car",
                bbox=[10.0, 10.0, 50.0],
                data_source="REAL"
            )

    def test_reject_hospital_bed_inconsistency(self):
        # available_beds > total_beds
        with pytest.raises(ValidationError):
            Hospital(
                hospital_id="h1",
                name="Bad Bed Hosp",
                latitude=12.0,
                longitude=77.0,
                total_beds=50,
                available_beds=60,
                icu_beds_available=10,
                data_source="SIMULATED"
            )

        # icu_beds_available > available_beds
        with pytest.raises(ValidationError):
            Hospital(
                hospital_id="h2",
                name="Bad ICU Hosp",
                latitude=12.0,
                longitude=77.0,
                total_beds=50,
                available_beds=20,
                icu_beds_available=25,
                data_source="SIMULATED"
            )

    def test_reject_invalid_coordinates(self):
        # Latitude out of bounds
        with pytest.raises(ValidationError):
            AccidentEvent(
                event_id="e1",
                timestamp=datetime.now(timezone.utc),
                location_name="Pole",
                latitude=95.0,
                longitude=0.0,
                severity="minor",
                data_source="REAL"
            )
        # Longitude out of bounds
        with pytest.raises(ValidationError):
            Ambulance(
                ambulance_id="a1",
                call_sign="EMS",
                latitude=0.0,
                longitude=190.0,
                data_source="SIMULATED"
            )
