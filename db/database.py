"""SQLite database management for Project Third-Eye.

Manages relational persistence matching the Pydantic schemas in core/schemas.py.
Enforces data_source constraints ('REAL' | 'SIMULATED') at both database and application level.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Generator, List, Optional

# Default database location within the project
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "third_eye.db"

# SQL DDL for table schemas
SCHEMA_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS detections (
    detection_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    bbox TEXT NOT NULL, -- JSON array of [xmin, ymin, xmax, ymax]
    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    timestamp TEXT,
    data_source TEXT NOT NULL CHECK(data_source IN ('REAL', 'SIMULATED'))
);

CREATE TABLE IF NOT EXISTS vehicle_reports (
    vehicle_id TEXT PRIMARY KEY,
    license_plate TEXT,
    plate_confidence REAL CHECK(plate_confidence IS NULL OR (plate_confidence >= 0.0 AND plate_confidence <= 1.0)),
    plate_detection_status TEXT,
    plate_bbox TEXT, -- JSON array of [xmin, ymin, xmax, ymax]
    plate_detector TEXT,
    vehicle_type TEXT,
    color TEXT,
    color_share REAL CHECK(color_share IS NULL OR (color_share >= 0.0 AND color_share <= 1.0)),
    damage_severity TEXT,
    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    timestamp TEXT,
    data_source TEXT NOT NULL CHECK(data_source IN ('REAL', 'SIMULATED'))
);

CREATE TABLE IF NOT EXISTS occupant_reports (
    occupant_id TEXT PRIMARY KEY,
    vehicle_id TEXT,
    bbox TEXT, -- JSON array of [xmin, ymin, xmax, ymax]
    estimated_count INTEGER NOT NULL DEFAULT 1 CHECK(estimated_count >= 0),
    injury_severity TEXT,
    consciousness TEXT,
    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    overlap_ratio REAL CHECK(overlap_ratio IS NULL OR (overlap_ratio >= 0.0 AND overlap_ratio <= 1.0)),
    persons_visible_count INTEGER CHECK(persons_visible_count IS NULL OR persons_visible_count >= 0),
    persons_near_vehicles_count INTEGER CHECK(persons_near_vehicles_count IS NULL OR persons_near_vehicles_count >= 0),
    limitation_note TEXT,
    timestamp TEXT,
    data_source TEXT NOT NULL CHECK(data_source IN ('REAL', 'SIMULATED'))
);

CREATE TABLE IF NOT EXISTS accident_events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    location_name TEXT NOT NULL,
    latitude REAL NOT NULL CHECK(latitude >= -90.0 AND latitude <= 90.0),
    longitude REAL NOT NULL CHECK(longitude >= -180.0 AND longitude <= 180.0),
    severity TEXT NOT NULL,
    vehicles_involved INTEGER NOT NULL DEFAULT 1 CHECK(vehicles_involved >= 0),
    casualties_estimated INTEGER NOT NULL DEFAULT 0 CHECK(casualties_estimated >= 0),
    status TEXT NOT NULL DEFAULT 'reported',
    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    method TEXT DEFAULT 'TRAINED_MODEL',
    score REAL CHECK(score IS NULL OR (score >= 0.0 AND score <= 1.0)),
    verdict TEXT,
    persons_near_vehicles_count INTEGER DEFAULT 0 CHECK(persons_near_vehicles_count >= 0),
    reasons TEXT,
    data_source TEXT NOT NULL CHECK(data_source IN ('REAL', 'SIMULATED'))
);

CREATE TABLE IF NOT EXISTS hospitals (
    hospital_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    latitude REAL NOT NULL CHECK(latitude >= -90.0 AND latitude <= 90.0),
    longitude REAL NOT NULL CHECK(longitude >= -180.0 AND longitude <= 180.0),
    total_beds INTEGER CHECK(total_beds IS NULL OR total_beds >= 0),
    available_beds INTEGER CHECK(available_beds IS NULL OR (available_beds >= 0 AND (total_beds IS NULL OR available_beds <= total_beds))),
    icu_beds_available INTEGER CHECK(icu_beds_available IS NULL OR (icu_beds_available >= 0 AND (available_beds IS NULL OR icu_beds_available <= available_beds))),
    trauma_center_level INTEGER CHECK(trauma_center_level IS NULL OR (trauma_center_level >= 1 AND trauma_center_level <= 5)),
    contact_number TEXT,
    address TEXT,
    emergency TEXT,
    specialties TEXT,
    attribution TEXT DEFAULT '© OpenStreetMap contributors',
    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    data_source TEXT NOT NULL CHECK(data_source IN ('REAL', 'SIMULATED'))
);

CREATE TABLE IF NOT EXISTS ambulances (
    ambulance_id TEXT PRIMARY KEY,
    call_sign TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'available',
    latitude REAL NOT NULL CHECK(latitude >= -90.0 AND latitude <= 90.0),
    longitude REAL NOT NULL CHECK(longitude >= -180.0 AND longitude <= 180.0),
    equipment_level TEXT NOT NULL DEFAULT 'Advanced',
    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    data_source TEXT NOT NULL CHECK(data_source IN ('REAL', 'SIMULATED'))
);

CREATE TABLE IF NOT EXISTS route_results (
    route_id TEXT PRIMARY KEY,
    ambulance_id TEXT NOT NULL,
    hospital_id TEXT NOT NULL,
    accident_id TEXT NOT NULL,
    distance_km REAL NOT NULL CHECK(distance_km >= 0.0),
    estimated_time_minutes REAL NOT NULL CHECK(estimated_time_minutes >= 0.0),
    route_waypoints TEXT, -- JSON array of [lat, lon]
    traffic_condition TEXT,
    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    data_source TEXT NOT NULL CHECK(data_source IN ('REAL', 'SIMULATED'))
);

CREATE TABLE IF NOT EXISTS audit_reports (
    audit_id TEXT PRIMARY KEY,
    accident_id TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    collision_analysis TEXT,
    liability_assessment TEXT,
    regulatory_compliance TEXT,
    timeline_summary TEXT,
    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    data_source TEXT NOT NULL CHECK(data_source IN ('REAL', 'SIMULATED'))
);

CREATE TABLE IF NOT EXISTS incident_reports (
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
    hazards TEXT, -- JSON array
    hazards_source TEXT NOT NULL,
    injuries TEXT, -- JSON array
    injuries_source TEXT NOT NULL,
    urgency TEXT,
    urgency_source TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    disagreements TEXT, -- JSON array
    video_evidence TEXT, -- JSON object with sampled-frame/temporal evidence
    video_assessment TEXT,
    video_temporal_signal TEXT,
    data_source TEXT NOT NULL CHECK(data_source IN ('REAL', 'SIMULATED'))
);

-- Indices for frequent spatial and relation lookups
CREATE INDEX IF NOT EXISTS idx_accident_timestamp ON accident_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_hospital_beds ON hospitals(available_beds, icu_beds_available);
CREATE INDEX IF NOT EXISTS idx_ambulance_status ON ambulances(status);
CREATE INDEX IF NOT EXISTS idx_route_accident ON route_results(accident_id);
"""

EXPECTED_TABLES = [
    "detections",
    "vehicle_reports",
    "occupant_reports",
    "accident_events",
    "hospitals",
    "ambulances",
    "route_results",
    "audit_reports",
    "incident_reports",
]


def get_connection(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    """Create a SQLite connection configured with Row factory and foreign keys."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def db_session(db_path: Optional[Path | str] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding an active database connection with automatic commit/rollback."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path | str] = None) -> List[str]:
    """Initialize SQLite database, execute DDL scripts, and verify all tables exist."""
    with db_session(db_path) as conn:
        conn.executescript(SCHEMA_DDL)
        cursor = conn.cursor()
        # Handle backward-compatible migration for accident_events columns
        cursor.execute("PRAGMA table_info(accident_events);")
        columns = [row["name"] for row in cursor.fetchall()]
        if columns:
            if "verdict" not in columns:
                cursor.execute("ALTER TABLE accident_events ADD COLUMN verdict TEXT;")
            if "persons_near_vehicles_count" not in columns:
                cursor.execute("ALTER TABLE accident_events ADD COLUMN persons_near_vehicles_count INTEGER DEFAULT 0;")

        # Phase 12: preserve dedicated plate-detection provenance in vehicle reports.
        cursor.execute("PRAGMA table_info(vehicle_reports);")
        vehicle_columns = [row["name"] for row in cursor.fetchall()]
        if vehicle_columns:
            if "plate_detection_status" not in vehicle_columns:
                cursor.execute("ALTER TABLE vehicle_reports ADD COLUMN plate_detection_status TEXT;")
            if "plate_bbox" not in vehicle_columns:
                cursor.execute("ALTER TABLE vehicle_reports ADD COLUMN plate_bbox TEXT;")
            if "plate_detector" not in vehicle_columns:
                cursor.execute("ALTER TABLE vehicle_reports ADD COLUMN plate_detector TEXT;")

        # Handle backward-compatible migration for occupant_reports columns
        cursor.execute("PRAGMA table_info(occupant_reports);")
        occ_columns = [row["name"] for row in cursor.fetchall()]
        if occ_columns:
            if "bbox" not in occ_columns:
                cursor.execute("ALTER TABLE occupant_reports ADD COLUMN bbox TEXT;")
            if "overlap_ratio" not in occ_columns:
                cursor.execute("ALTER TABLE occupant_reports ADD COLUMN overlap_ratio REAL;")
            if "persons_visible_count" not in occ_columns:
                cursor.execute("ALTER TABLE occupant_reports ADD COLUMN persons_visible_count INTEGER;")
            if "persons_near_vehicles_count" not in occ_columns:
                cursor.execute("ALTER TABLE occupant_reports ADD COLUMN persons_near_vehicles_count INTEGER;")
            if "limitation_note" not in occ_columns:
                cursor.execute("ALTER TABLE occupant_reports ADD COLUMN limitation_note TEXT;")

        # Phase 14: preserve sampled-frame/temporal evidence in fused reports.
        cursor.execute("PRAGMA table_info(incident_reports);")
        incident_columns = [row["name"] for row in cursor.fetchall()]
        if incident_columns:
            if "video_evidence" not in incident_columns:
                cursor.execute("ALTER TABLE incident_reports ADD COLUMN video_evidence TEXT;")
            if "video_assessment" not in incident_columns:
                cursor.execute("ALTER TABLE incident_reports ADD COLUMN video_assessment TEXT;")
            if "video_temporal_signal" not in incident_columns:
                cursor.execute("ALTER TABLE incident_reports ADD COLUMN video_temporal_signal TEXT;")

        # Handle backward-compatible migration for hospitals columns
        cursor.execute("PRAGMA table_info(hospitals);")
        hosp_info = cursor.fetchall()
        hosp_columns = [row["name"] for row in hosp_info]
        if hosp_columns:
            # If total_beds is NOT NULL, recreate table to allow NULL beds for real OSM facilities
            total_beds_notnull = any(row["name"] == "total_beds" and row["notnull"] == 1 for row in hosp_info)
            if total_beds_notnull:
                cursor.execute("ALTER TABLE hospitals RENAME TO hospitals_old;")
                cursor.execute("""
                    CREATE TABLE hospitals (
                        hospital_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        latitude REAL NOT NULL CHECK(latitude >= -90.0 AND latitude <= 90.0),
                        longitude REAL NOT NULL CHECK(longitude >= -180.0 AND longitude <= 180.0),
                        total_beds INTEGER CHECK(total_beds IS NULL OR total_beds >= 0),
                        available_beds INTEGER CHECK(available_beds IS NULL OR (available_beds >= 0 AND (total_beds IS NULL OR available_beds <= total_beds))),
                        icu_beds_available INTEGER CHECK(icu_beds_available IS NULL OR (icu_beds_available >= 0 AND (available_beds IS NULL OR icu_beds_available <= available_beds))),
                        trauma_center_level INTEGER CHECK(trauma_center_level IS NULL OR (trauma_center_level >= 1 AND trauma_center_level <= 5)),
                        contact_number TEXT,
                        address TEXT,
                        emergency TEXT,
                        specialties TEXT,
                        attribution TEXT DEFAULT '© OpenStreetMap contributors',
                        confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
                        data_source TEXT NOT NULL CHECK(data_source IN ('REAL', 'SIMULATED'))
                    );
                """)
                cursor.execute("""
                    INSERT OR IGNORE INTO hospitals (
                        hospital_id, name, latitude, longitude, total_beds, available_beds,
                        icu_beds_available, trauma_center_level, contact_number, confidence, data_source
                    )
                    SELECT hospital_id, name, latitude, longitude, total_beds, available_beds,
                           icu_beds_available, trauma_center_level, contact_number, confidence, data_source
                    FROM hospitals_old;
                """)
                cursor.execute("DROP TABLE hospitals_old;")
            else:
                if "address" not in hosp_columns:
                    cursor.execute("ALTER TABLE hospitals ADD COLUMN address TEXT;")
                if "emergency" not in hosp_columns:
                    cursor.execute("ALTER TABLE hospitals ADD COLUMN emergency TEXT;")
                if "specialties" not in hosp_columns:
                    cursor.execute("ALTER TABLE hospitals ADD COLUMN specialties TEXT;")
                if "attribution" not in hosp_columns:
                    cursor.execute("ALTER TABLE hospitals ADD COLUMN attribution TEXT DEFAULT '© OpenStreetMap contributors';")
                if "excluded_from_dispatch_reason" not in hosp_columns:
                    cursor.execute("ALTER TABLE hospitals ADD COLUMN excluded_from_dispatch_reason TEXT;")

        # Handle backward-compatible migration for ambulances columns
        cursor.execute("PRAGMA table_info(ambulances);")
        amb_columns = [row["name"] for row in cursor.fetchall()]
        if amb_columns:
            if "base_hospital" not in amb_columns:
                cursor.execute("ALTER TABLE ambulances ADD COLUMN base_hospital TEXT;")
            if "hospital_osm_id" not in amb_columns:
                cursor.execute("ALTER TABLE ambulances ADD COLUMN hospital_osm_id INTEGER;")

        # Handle backward-compatible migration for route_results columns
        cursor.execute("PRAGMA table_info(route_results);")
        rr_columns = [row["name"] for row in cursor.fetchall()]
        if rr_columns:
            if "leg1_waypoints" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN leg1_waypoints TEXT;")
            if "leg2_waypoints" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN leg2_waypoints TEXT;")
            if "ambulance_eta_minutes" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN ambulance_eta_minutes REAL;")
            if "hospital_transit_minutes" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN hospital_transit_minutes REAL;")
            if "straight_line_km" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN straight_line_km REAL;")
            if "driving_vs_straight_ratio" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN driving_vs_straight_ratio REAL;")
            if "chosen_ambulance" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN chosen_ambulance TEXT;")
            if "chosen_hospital" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN chosen_hospital TEXT;")
            if "alternative_ambulances" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN alternative_ambulances TEXT;")
            if "alternative_hospitals" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN alternative_hospitals TEXT;")
            if "attribution" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN attribution TEXT DEFAULT '© OpenStreetMap contributors';")
            if "disclaimer" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN disclaimer TEXT DEFAULT 'Estimated from OSM speed data, no live traffic';")
            if "coverage_note" not in rr_columns:
                cursor.execute("ALTER TABLE route_results ADD COLUMN coverage_note TEXT DEFAULT 'Routing covers this corridor only';")

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row["name"] for row in cursor.fetchall()]
    return tables


def get_table_names(db_path: Optional[Path | str] = None) -> List[str]:
    """List all user tables in the database."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
        return [row["name"] for row in cursor.fetchall()]


def save_vehicle_reports(reports: list, db_path: Optional[Path | str] = None) -> int:
    """Save a list of VehicleReport instances into SQLite vehicle_reports table."""
    if not reports:
        return 0

    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    # Ensure database and tables are initialized
    init_db(target_path)

    inserted_count = 0
    with db_session(target_path) as conn:
        cursor = conn.cursor()
        for report in reports:
            # Handle Pydantic model or dict
            data = report.model_dump() if hasattr(report, "model_dump") else dict(report)
            ts_str = data["timestamp"].isoformat() if data.get("timestamp") else None
            plate_bbox = data.get("plate_bbox")
            plate_bbox_json = json.dumps(list(plate_bbox)) if plate_bbox is not None else None
            cursor.execute(
                """INSERT OR REPLACE INTO vehicle_reports (
                    vehicle_id, license_plate, plate_confidence,
                    plate_detection_status, plate_bbox, plate_detector,
                    vehicle_type, color, color_share, damage_severity, confidence,
                    timestamp, data_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["vehicle_id"],
                    data.get("license_plate"),
                    data.get("plate_confidence"),
                    data.get("plate_detection_status"),
                    plate_bbox_json,
                    data.get("plate_detector"),
                    data.get("vehicle_type"),
                    data.get("color"),
                    data.get("color_share"),
                    data.get("damage_severity"),
                    data.get("confidence"),
                    ts_str,
                    data["data_source"],
                ),
            )
            inserted_count += 1
    return inserted_count


def get_vehicle_reports(db_path: Optional[Path | str] = None) -> List[sqlite3.Row]:
    """Retrieve all stored vehicle reports from the database."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)
    with db_session(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vehicle_reports ORDER BY timestamp DESC;")
        return cursor.fetchall()


def save_accident_event(event: Any, db_path: Optional[Path | str] = None) -> str:
    """Save or update an AccidentEvent into the SQLite accident_events table."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)

    data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
    ts_str = data["timestamp"].isoformat() if hasattr(data["timestamp"], "isoformat") else str(data["timestamp"])

    with db_session(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO accident_events (
                event_id, timestamp, location_name, latitude, longitude,
                severity, vehicles_involved, casualties_estimated, status,
                confidence, method, score, verdict, persons_near_vehicles_count,
                reasons, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["event_id"],
                ts_str,
                data["location_name"],
                data["latitude"],
                data["longitude"],
                data["severity"],
                data.get("vehicles_involved", 1),
                data.get("casualties_estimated", 0),
                data.get("status", "reported"),
                data.get("confidence"),
                data.get("method", "TRAINED_MODEL"),
                data.get("score"),
                data.get("verdict"),
                data.get("persons_near_vehicles_count", 0),
                data.get("reasons"),
                data["data_source"],
            ),
        )
    return data["event_id"]


def get_accident_events(db_path: Optional[Path | str] = None) -> List[sqlite3.Row]:
    """Retrieve all stored accident events from the database."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)
    with db_session(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM accident_events ORDER BY timestamp DESC;")
        return cursor.fetchall()


def save_occupant_reports(reports: list, db_path: Optional[Path | str] = None) -> int:
    """Save a list of OccupantReport instances into SQLite occupant_reports table."""
    if not reports:
        return 0

    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)

    inserted_count = 0
    with db_session(target_path) as conn:
        cursor = conn.cursor()
        for report in reports:
            data = report.model_dump() if hasattr(report, "model_dump") else dict(report)
            ts_str = data["timestamp"].isoformat() if data.get("timestamp") else None
            bbox_str = json.dumps(data.get("bbox")) if data.get("bbox") is not None else None
            cursor.execute(
                """INSERT OR REPLACE INTO occupant_reports (
                    occupant_id, vehicle_id, bbox, estimated_count,
                    injury_severity, consciousness, confidence, overlap_ratio,
                    persons_visible_count, persons_near_vehicles_count, limitation_note,
                    timestamp, data_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["occupant_id"],
                    data.get("vehicle_id"),
                    bbox_str,
                    data.get("estimated_count", 1),
                    data.get("injury_severity", "not determined"),
                    data.get("consciousness", "unknown"),
                    data.get("confidence"),
                    data.get("overlap_ratio"),
                    data.get("persons_visible_count"),
                    data.get("persons_near_vehicles_count"),
                    data.get("limitation_note", "Visible people only. Occupants inside vehicles cannot be reliably counted from a single image."),
                    ts_str,
                    data["data_source"],
                ),
            )
            inserted_count += 1
    return inserted_count


def get_occupant_reports(db_path: Optional[Path | str] = None) -> List[sqlite3.Row]:
    """Retrieve all stored occupant reports from the database."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)
    with db_session(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM occupant_reports ORDER BY occupant_id ASC;")
        return cursor.fetchall()


def save_incident_report(report: Any, db_path: Optional[Path | str] = None) -> str:
    """Save or update an IncidentReport into the SQLite incident_reports table."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)

    data = report.model_dump() if hasattr(report, "model_dump") else dict(report)
    ts_str = data["timestamp"].isoformat() if hasattr(data["timestamp"], "isoformat") else str(data["timestamp"])
    hazards_json = json.dumps(data.get("hazards", []))
    injuries_json = json.dumps(data.get("injuries", []))
    disagreements_json = json.dumps(data.get("disagreements", []))
    video_evidence = data.get("video_evidence")
    video_evidence_json = json.dumps(video_evidence) if video_evidence is not None else None

    with db_session(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO incident_reports (
                report_id, image_name, call_id, timestamp, verdict, verdict_source,
                location, location_source, vision_vehicle_count, caller_vehicle_count,
                vehicle_count_disagreement, vision_people_count, caller_people_count,
                people_count_disagreement, hazards, hazards_source, injuries,
                injuries_source, urgency, urgency_source, summary_text,
                disagreements, video_evidence, video_assessment,
                video_temporal_signal, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["report_id"],
                data.get("image_name"),
                data.get("call_id"),
                ts_str,
                data["verdict"],
                data.get("verdict_source", "VISION"),
                data.get("location"),
                data.get("location_source", "MISSING"),
                data.get("vision_vehicle_count"),
                data.get("caller_vehicle_count"),
                1 if data.get("vehicle_count_disagreement") else 0,
                data.get("vision_people_count"),
                data.get("caller_people_count"),
                1 if data.get("people_count_disagreement") else 0,
                hazards_json,
                data.get("hazards_source", "MISSING"),
                injuries_json,
                data.get("injuries_source", "MISSING"),
                data.get("urgency"),
                data.get("urgency_source", "MISSING"),
                data["summary_text"],
                disagreements_json,
                video_evidence_json,
                data.get("video_assessment"),
                data.get("video_temporal_signal"),
                data["data_source"],
            ),
        )
    return data["report_id"]


def get_incident_reports(db_path: Optional[Path | str] = None) -> List[sqlite3.Row]:
    """Retrieve all stored incident reports from the database."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)
    with db_session(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM incident_reports ORDER BY timestamp DESC;")
        return cursor.fetchall()


def save_hospitals(hospitals: List[Any], db_path: Optional[Path | str] = None) -> int:
    """Save or update Hospital objects in the SQLite hospitals table.
    
    Accepts Pydantic Hospital instances or dicts.
    Returns the number of rows inserted or updated.
    """
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)
    inserted_count = 0

    with db_session(target_path) as conn:
        cursor = conn.cursor()
        for hospital in hospitals:
            data = hospital.model_dump() if hasattr(hospital, "model_dump") else dict(hospital)
            cursor.execute(
                """INSERT OR REPLACE INTO hospitals (
                    hospital_id, name, latitude, longitude, total_beds,
                    available_beds, icu_beds_available, trauma_center_level,
                    contact_number, address, emergency, specialties,
                    attribution, confidence, data_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["hospital_id"],
                    data["name"],
                    data["latitude"],
                    data["longitude"],
                    data.get("total_beds"),
                    data.get("available_beds"),
                    data.get("icu_beds_available"),
                    data.get("trauma_center_level"),
                    data.get("contact_number"),
                    data.get("address"),
                    data.get("emergency"),
                    data.get("specialties"),
                    data.get("attribution", "© OpenStreetMap contributors"),
                    data.get("confidence"),
                    data["data_source"],
                ),
            )
            inserted_count += 1
    return inserted_count


def get_hospitals(db_path: Optional[Path | str] = None) -> List[sqlite3.Row]:
    """Retrieve all stored hospitals from the database."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)
    with db_session(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM hospitals ORDER BY name ASC;")
        return cursor.fetchall()


def save_ambulances(ambulances: List[Any], db_path: Optional[Path | str] = None) -> int:
    """Save or update Ambulance objects in the SQLite ambulances table."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)
    count = 0
    with db_session(target_path) as conn:
        cursor = conn.cursor()
        for amb in ambulances:
            data = amb.model_dump() if hasattr(amb, "model_dump") else dict(amb)
            cursor.execute(
                """INSERT OR REPLACE INTO ambulances (
                    ambulance_id, call_sign, status, latitude, longitude,
                    equipment_level, confidence, base_hospital, hospital_osm_id, data_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["ambulance_id"],
                    data["call_sign"],
                    data.get("status", "available"),
                    data["latitude"],
                    data["longitude"],
                    data.get("equipment_level", "Advanced"),
                    data.get("confidence"),
                    data.get("base_hospital"),
                    data.get("hospital_osm_id"),
                    data["data_source"],
                ),
            )
            count += 1
    return count


def get_ambulances(db_path: Optional[Path | str] = None) -> List[sqlite3.Row]:
    """Retrieve all stored ambulances from the database."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)
    with db_session(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ambulances ORDER BY ambulance_id ASC;")
        return cursor.fetchall()


def save_route_result(route_result: Any, db_path: Optional[Path | str] = None) -> str:
    """Save or update a RouteResult in the SQLite route_results table."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)
    data = route_result.model_dump() if hasattr(route_result, "model_dump") else dict(route_result)

    waypoints_json = json.dumps(data.get("route_waypoints")) if data.get("route_waypoints") is not None else None
    leg1_json = json.dumps(data.get("leg1_waypoints")) if data.get("leg1_waypoints") is not None else None
    leg2_json = json.dumps(data.get("leg2_waypoints")) if data.get("leg2_waypoints") is not None else None
    chosen_amb_json = json.dumps(data.get("chosen_ambulance")) if data.get("chosen_ambulance") is not None else None
    chosen_hosp_json = json.dumps(data.get("chosen_hospital")) if data.get("chosen_hospital") is not None else None
    alt_amb_json = json.dumps(data.get("alternative_ambulances", []))
    alt_hosp_json = json.dumps(data.get("alternative_hospitals", []))

    with db_session(target_path) as conn:
        cursor = conn.cursor()
        # Cleanly replace any previous route result row for this accident scenario
        cursor.execute("DELETE FROM route_results WHERE accident_id = ?", (data["accident_id"],))
        cursor.execute(
            """INSERT OR REPLACE INTO route_results (
                route_id, ambulance_id, hospital_id, accident_id, distance_km,
                estimated_time_minutes, route_waypoints, traffic_condition, confidence,
                leg1_waypoints, leg2_waypoints, ambulance_eta_minutes, hospital_transit_minutes,
                straight_line_km, driving_vs_straight_ratio, chosen_ambulance, chosen_hospital,
                alternative_ambulances, alternative_hospitals, attribution, disclaimer,
                coverage_note, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["route_id"],
                data["ambulance_id"],
                data["hospital_id"],
                data["accident_id"],
                data["distance_km"],
                data["estimated_time_minutes"],
                waypoints_json,
                data.get("traffic_condition"),
                data.get("confidence"),
                leg1_json,
                leg2_json,
                data.get("ambulance_eta_minutes"),
                data.get("hospital_transit_minutes"),
                data.get("straight_line_km"),
                data.get("driving_vs_straight_ratio"),
                chosen_amb_json,
                chosen_hosp_json,
                alt_amb_json,
                alt_hosp_json,
                data.get("attribution", "© OpenStreetMap contributors"),
                data.get("disclaimer", "Estimated from OSM speed data, no live traffic"),
                data.get("coverage_note", "Routing covers this corridor only"),
                data["data_source"],
            ),
        )
    return data["route_id"]


def get_route_results(db_path: Optional[Path | str] = None) -> List[sqlite3.Row]:
    """Retrieve all stored route results from the database."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)
    with db_session(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM route_results ORDER BY route_id ASC;")
        return cursor.fetchall()




