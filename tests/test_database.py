"""Tests for SQLite database creation and constraints in db/database.py.

Validates that all tables are created, match schema specifications,
and enforce data integrity constraints (including data_source CHECK constraints).
"""

import json
from pathlib import Path
import sqlite3
import pytest

from core.hospitals.classification import (
    CATEGORY_HOSPITAL,
    CATEGORY_UNRELATED,
    classify_hospital_record,
)
from core.schemas import Hospital
from db.database import (
    EXPECTED_TABLES,
    get_connection,
    get_hospitals,
    get_table_names,
    init_db,
    save_hospitals,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Fixture providing a temporary database file."""
    db_file = tmp_path / "test_third_eye.db"
    return db_file


class TestDatabaseSetup:
    """Verify SQLite database initialization and table schemas."""

    def test_init_db_creates_all_tables(self, temp_db: Path):
        created_tables = init_db(temp_db)
        for expected in EXPECTED_TABLES:
            assert expected in created_tables, f"Table '{expected}' missing from created tables"

        # Verify through get_table_names helper
        listed_tables = get_table_names(temp_db)
        for expected in EXPECTED_TABLES:
            assert expected in listed_tables

    def test_foreign_keys_and_pragma_enabled(self, temp_db: Path):
        init_db(temp_db)
        conn = get_connection(temp_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys;")
        assert cursor.fetchone()[0] == 1
        conn.close()

    def test_insert_and_query_all_records(self, temp_db: Path):
        init_db(temp_db)
        conn = get_connection(temp_db)
        cursor = conn.cursor()

        # Insert detection
        cursor.execute(
            """INSERT INTO detections (detection_id, label, bbox, confidence, timestamp, data_source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("d1", "car", json.dumps([10, 20, 100, 200]), 0.94, "2026-11-24T10:00:00Z", "REAL")
        )

        # Insert vehicle report
        cursor.execute(
            """INSERT INTO vehicle_reports (vehicle_id, license_plate, plate_confidence, vehicle_type, color, color_share, damage_severity, confidence, timestamp, data_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("v1", "KA05MH1234", 0.95, "sedan", "blue", 0.82, "severe", 0.85, "2026-11-24T10:00:00Z", "REAL")
        )

        # Insert occupant report
        cursor.execute(
            """INSERT INTO occupant_reports (occupant_id, vehicle_id, estimated_count, injury_severity, consciousness, confidence, timestamp, data_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("o1", "v1", 2, "critical", "conscious", 0.80, "2026-11-24T10:00:00Z", "SIMULATED")
        )

        # Insert accident event
        cursor.execute(
            """INSERT INTO accident_events (event_id, timestamp, location_name, latitude, longitude, severity, vehicles_involved, casualties_estimated, status, confidence, data_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("e1", "2026-11-24T10:00:00Z", "Indiranagar 100ft Rd", 12.9783, 77.6408, "critical", 2, 2, "reported", 0.95, "REAL")
        )

        # Insert hospital
        cursor.execute(
            """INSERT INTO hospitals (hospital_id, name, latitude, longitude, total_beds, available_beds, icu_beds_available, trauma_center_level, contact_number, confidence, data_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("h1", "Manipal Hospital", 12.9592, 77.6484, 150, 30, 8, 1, "+91-80-2502-4444", 0.99, "REAL")
        )

        # Insert ambulance
        cursor.execute(
            """INSERT INTO ambulances (ambulance_id, call_sign, status, latitude, longitude, equipment_level, confidence, data_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("a1", "AMB-SOUTH-01", "available", 12.9600, 77.6300, "Critical Care", None, "SIMULATED")
        )

        # Insert route result
        cursor.execute(
            """INSERT INTO route_results (route_id, ambulance_id, hospital_id, accident_id, distance_km, estimated_time_minutes, route_waypoints, traffic_condition, confidence, data_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("r1", "a1", "h1", "e1", 3.2, 8.5, json.dumps([[12.9600, 77.6300], [12.9783, 77.6408]]), "moderate", 0.91, "SIMULATED")
        )

        # Insert audit report
        cursor.execute(
            """INSERT INTO audit_reports (audit_id, accident_id, generated_at, collision_analysis, liability_assessment, regulatory_compliance, timeline_summary, confidence, data_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("au1", "e1", "2026-11-24T10:30:00Z", "T-bone collision at junction", "Pending investigation", "Section 134 MV Act", "Event logged", 0.88, "REAL")
        )

        # Insert incident report
        cursor.execute(
            """INSERT INTO incident_reports (report_id, image_name, call_id, timestamp, verdict, verdict_source, location, location_source, vision_vehicle_count, caller_vehicle_count, vehicle_count_disagreement, vision_people_count, caller_people_count, people_count_disagreement, hazards, hazards_source, injuries, injuries_source, urgency, urgency_source, summary_text, disagreements, data_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("ir1", "road_1.jpg", "call_1", "2026-11-24T10:30:00Z", "NO ACCIDENT SIGNAL", "VISION", "MG Road", "CALLER_SIMULATED", 3, 2, 1, 0, 0, 0, json.dumps([]), "MISSING", json.dumps([]), "MISSING", "normal", "CALLER_SIMULATED", "Summary text", json.dumps(["vehicle count"]), "SIMULATED")
        )
        conn.commit()

        # Verify all 8 counts are 1
        for table in EXPECTED_TABLES:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            assert cursor.fetchone()[0] == 1, f"Table {table} did not retain inserted row"

        conn.close()

    def test_save_vehicle_reports_helper(self, temp_db: Path):
        from core.schemas import VehicleReport
        from db.database import get_vehicle_reports, save_vehicle_reports
        init_db(temp_db)

        rep = VehicleReport(
            vehicle_id="veh-test-db-1",
            license_plate="KA03NC5678",
            plate_confidence=0.91,
            plate_detection_status="PLATE REGION DETECTED — READABLE",
            plate_bbox=(101, 202, 321, 254),
            plate_detector="YUNET_LPD_2023MAR",
            vehicle_type="truck",
            color="yellow",
            color_share=0.65,
            confidence=0.88,
            data_source="REAL"
        )
        saved_count = save_vehicle_reports([rep], db_path=temp_db)
        assert saved_count == 1

        stored = get_vehicle_reports(db_path=temp_db)
        assert len(stored) == 1
        assert stored[0]["vehicle_id"] == "veh-test-db-1"
        assert stored[0]["license_plate"] == "KA03NC5678"
        assert stored[0]["color"] == "yellow"
        assert stored[0]["plate_detection_status"] == "PLATE REGION DETECTED — READABLE"
        assert json.loads(stored[0]["plate_bbox"]) == [101, 202, 321, 254]
        assert stored[0]["plate_detector"] == "YUNET_LPD_2023MAR"

    def test_legacy_vehicle_reports_table_is_migrated(self, temp_db: Path):
        conn = get_connection(temp_db)
        conn.execute(
            """CREATE TABLE vehicle_reports (
                vehicle_id TEXT PRIMARY KEY,
                license_plate TEXT,
                plate_confidence REAL,
                vehicle_type TEXT,
                color TEXT,
                color_share REAL,
                damage_severity TEXT,
                confidence REAL,
                timestamp TEXT,
                data_source TEXT NOT NULL
            )"""
        )
        conn.commit()
        conn.close()

        init_db(temp_db)
        conn = get_connection(temp_db)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(vehicle_reports)")}
        conn.close()
        assert {"plate_detection_status", "plate_bbox", "plate_detector"} <= columns

    def test_legacy_incident_reports_table_is_migrated(self, temp_db: Path):
        conn = get_connection(temp_db)
        conn.execute(
            """CREATE TABLE incident_reports (
                report_id TEXT PRIMARY KEY,
                image_name TEXT,
                call_id TEXT,
                timestamp TEXT NOT NULL,
                verdict TEXT NOT NULL,
                verdict_source TEXT NOT NULL,
                location TEXT,
                location_source TEXT NOT NULL,
                vision_vehicle_count INTEGER,
                caller_vehicle_count INTEGER,
                vehicle_count_disagreement INTEGER NOT NULL DEFAULT 0,
                vision_people_count INTEGER,
                caller_people_count INTEGER,
                people_count_disagreement INTEGER NOT NULL DEFAULT 0,
                hazards TEXT,
                hazards_source TEXT NOT NULL,
                injuries TEXT,
                injuries_source TEXT NOT NULL,
                urgency TEXT,
                urgency_source TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                disagreements TEXT,
                data_source TEXT NOT NULL
            )"""
        )
        conn.commit()
        conn.close()

        init_db(temp_db)
        conn = get_connection(temp_db)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(incident_reports)")}
        conn.close()
        assert {"video_evidence", "video_assessment", "video_temporal_signal"} <= columns

    def test_database_enforces_data_source_constraint(self, temp_db: Path):
        init_db(temp_db)
        conn = get_connection(temp_db)
        cursor = conn.cursor()

        # Rejecting invalid data_source ('FAKE') via CHECK constraint
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute(
                """INSERT INTO hospitals (hospital_id, name, latitude, longitude, total_beds, available_beds, icu_beds_available, data_source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("h_bad", "Bad Hospital", 12.0, 77.0, 50, 10, 2, "FAKE")
            )
            conn.commit()

        conn.close()

    def test_database_enforces_bed_capacity_constraint(self, temp_db: Path):
        init_db(temp_db)
        conn = get_connection(temp_db)
        cursor = conn.cursor()

        # Rejecting available_beds > total_beds via CHECK constraint
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute(
                """INSERT INTO hospitals (hospital_id, name, latitude, longitude, total_beds, available_beds, icu_beds_available, data_source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("h_bad_beds", "Bad Bed Hospital", 12.0, 77.0, 50, 60, 2, "REAL")
            )
            conn.commit()

        conn.close()

    def test_source_classification_survives_hospital_save_and_retrieve(self, temp_db: Path):
        init_db(temp_db)
        eligible = Hospital(
            hospital_id="h_eligible",
            name="Civil Hospital Phagwara",
            latitude=31.2264,
            longitude=75.7681,
            emergency="yes",
            data_source="REAL",
            source_facility_category=CATEGORY_HOSPITAL,
            source_dispatch_eligible=True,
        )
        excluded = Hospital(
            hospital_id="h_excluded",
            name="LPU Architecture Deisgn",
            latitude=31.2142,
            longitude=75.7785,
            data_source="REAL",
            source_facility_category=CATEGORY_UNRELATED,
            source_dispatch_eligible=False,
            excluded_from_dispatch_reason="Non-medical building tag",
        )

        assert save_hospitals([eligible, excluded], temp_db) == 2
        rows = {row["hospital_id"]: row for row in get_hospitals(temp_db)}
        assert set(rows) == {"h_eligible", "h_excluded"}

        reloaded_eligible = Hospital(**dict(rows["h_eligible"]))
        reloaded_excluded = Hospital(**dict(rows["h_excluded"]))
        assert reloaded_eligible.source_facility_category == CATEGORY_HOSPITAL
        assert reloaded_eligible.source_dispatch_eligible is True
        assert reloaded_eligible.excluded_from_dispatch_reason is None
        assert reloaded_excluded.source_facility_category == CATEGORY_UNRELATED
        assert reloaded_excluded.source_dispatch_eligible is False
        assert reloaded_excluded.excluded_from_dispatch_reason == "Non-medical building tag"
        assert classify_hospital_record(reloaded_eligible).dispatch_eligible is True
        assert classify_hospital_record(reloaded_excluded).dispatch_eligible is False

    def test_legacy_hospitals_table_is_migrated_additively(self, temp_db: Path):
        conn = get_connection(temp_db)
        conn.execute(
            """CREATE TABLE hospitals (
                hospital_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                total_beds INTEGER,
                available_beds INTEGER,
                icu_beds_available INTEGER,
                trauma_center_level INTEGER,
                contact_number TEXT,
                confidence REAL,
                data_source TEXT NOT NULL
            )"""
        )
        conn.execute(
            """INSERT INTO hospitals (
                hospital_id, name, latitude, longitude, total_beds, data_source
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("h_legacy", "Legacy Hospital", 31.2, 75.7, 100, "REAL"),
        )
        conn.commit()
        conn.close()

        init_db(temp_db)

        conn = get_connection(temp_db)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(hospitals)")}
        row = conn.execute(
            "SELECT * FROM hospitals WHERE hospital_id = ?", ("h_legacy",)
        ).fetchone()
        conn.close()
        assert {
            "excluded_from_dispatch_reason",
            "source_facility_category",
            "source_dispatch_eligible",
        } <= columns
        assert row["name"] == "Legacy Hospital"
        assert row["total_beds"] == 100
        assert row["source_facility_category"] is None
        assert row["source_dispatch_eligible"] is None
