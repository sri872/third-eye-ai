# Phase 0 Acceptance & Foundation Report

**Project**: Project Third-Eye (Multimodal Smart Ambulance Router & Automated Accident Forensic Auditor)  
**Phase**: Phase 0 (Foundation)  
**Date**: 2026-11-24  
**Target Hardware**: MacBook Air M2 (Apple Silicon)  
**Environment**: Python 3.12.3  

---

## 1. System Environment Check

- **Operating System**: macOS (Darwin 24.3.0, arm64)
- **Python Binary**: `/usr/local/bin/python3`
- **Python Version Output**:
  ```text
  Python 3.12.3
  ```
- **Filesystem & Disk Space (`df -h`)**:
  ```text
  Filesystem        Size    Used   Avail Capacity iused ifree %iused  Mounted on
  /dev/disk3s1s1   228Gi    13Gi   2.8Gi    82%    484k   30M    2%   /
  /dev/disk3s5     228Gi   196Gi   2.8Gi    99%    1.6M   30M    5%   /System/Volumes/Data
  ```
  *Status*: **Verified**. Available storage is 2.8 GiB, sufficient for Phase 0 lightweight dependencies.

---

## 2. Directory Structure & Virtual Environment

- **Virtual Environment**: `.venv` created with Python 3.12.3.
- **Phase 0 Dependencies Installed (`requirements.txt`)**:
  - `pytest>=8.0.0` (installed: `pytest-9.2.0`)
  - `pydantic>=2.0.0` (installed: `pydantic-2.14.4`)
  - `numpy>=1.26.0` (installed: `numpy-2.5.5`)
  - `streamlit>=1.30.0` (installed: `streamlit-1.67.0`)
- **Package Hierarchy**:
  - `app/` (with `__init__.py`)
  - `core/` (with `__init__.py`)
  - `core/cv/` (with `__init__.py`)
  - `core/ocr/` (with `__init__.py`)
  - `core/nlp/` (with `__init__.py`)
  - `core/rag/` (with `__init__.py`)
  - `core/ml/` (with `__init__.py`)
  - `data/`
  - `models/`
  - `db/` (with `__init__.py`)
  - `tests/` (with `__init__.py`)
  - `docs/`

---

## 3. Shared Schemas (`core/schemas.py`)

All shared data models are implemented using Pydantic V2 (`BaseModel` with strict assignment validation and `extra="forbid"`):
1. **`Detection`**: Bounding box coordinates (`[xmin, ymin, xmax, ymax]`), label, optional confidence score (`[0.0, 1.0]`), timestamp, and required `data_source`.
2. **`VehicleReport`**: Forensic vehicle properties, damage severity, license plate, optional confidence, and `data_source`.
3. **`OccupantReport`**: Occupant triage counts, injury severity, consciousness indicators, optional confidence, and `data_source`.
4. **`AccidentEvent`**: Incident location, WGS84 latitude/longitude, severity categorization, vehicles involved, casualties, status, optional confidence, and `data_source`.
5. **`Hospital`**: Geographic coordinates, capacity metrics (`total_beds`, `available_beds`, `icu_beds_available`), trauma level, and bed consistency validators (`available_beds <= total_beds`, `icu_beds_available <= available_beds`).
6. **`Ambulance`**: Vehicle identifier, call sign, status, coordinates, equipment tier (`Basic`, `Advanced`, `Critical Care`), and `data_source`.
7. **`RouteResult`**: Ambulance-to-accident-to-hospital routing calculations, distance, estimated time, waypoints, traffic conditions, and `data_source`.
8. **`AuditReport`**: Forensic accident audit, collision analysis, liability assessment, regulatory compliance citations, timeline summary, optional confidence, and `data_source`.

### Core Data Integrity Rules Enforced:
- **`data_source`**: Must be either `"REAL"` or `"SIMULATED"`. Any other string (e.g., `"MOCK"`, `"FAKE"`) is rejected.
- **`confidence`**: Always optional (`None` by default), bounded between `0.0` and `1.0`. Never fabricated or assumed.

---

## 4. SQLite Database Layer (`db/database.py`)

- **Database File**: `db/third_eye.db`
- **Tables Implemented**:
  1. `detections`
  2. `vehicle_reports`
  3. `occupant_reports`
  4. `accident_events`
  5. `hospitals`
  6. `ambulances`
  7. `route_results`
  8. `audit_reports`
- **Integrity Constraints**:
  - `PRAGMA foreign_keys = ON;`
  - `CHECK(data_source IN ('REAL', 'SIMULATED'))` on all tables.
  - Coordinate range checks (`latitude BETWEEN -90 AND 90`, `longitude BETWEEN -180 AND 180`).
  - Hospital bed capacity checks (`available_beds <= total_beds`, `icu_beds_available <= available_beds`).
  - Non-negative constraints for counts, distances, and travel times.
- **Indexes**:
  - `idx_accident_timestamp`, `idx_hospital_beds`, `idx_ambulance_status`, `idx_route_accident`.

---

## 5. UI Launcher & Dashboard (`launch.command` & `app/main.py`)

- **Double-Click Launcher**: `launch.command` created with execution permissions (`chmod +x`). Automatically activates `.venv` and invokes Streamlit.
- **Zero Action Buttons**: The UI contains **no submit buttons, no triggers, and zero required text inputs**. All interactions are driven by dropdown selection with auto-running render updates.
- **Truth In Labelling**: Visual chips display `REAL` (green) and `SIMULATED` (blue) indicators clearly.

---

## 6. Verification & Test Suite Execution

Pytest execution ran with 25 test cases across schema validations, negative boundary tests, and database constraints.

### Test Output:
```text
============================= test session starts ==============================
platform darwin -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0 -- /Users/srishanthbiradar/Projects/third-eye-v2/.venv/bin/python3.12
cachedir: .pytest_cache
rootdir: /Users/srishanthbiradar/Projects/third-eye-v2
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.16.0
collecting ... collected 25 items

tests/test_database.py::TestDatabaseSetup::test_init_db_creates_all_tables PASSED [  4%]
tests/test_database.py::TestDatabaseSetup::test_foreign_keys_and_pragma_enabled PASSED [  8%]
tests/test_database.py::TestDatabaseSetup::test_insert_and_query_all_records PASSED [ 12%]
tests/test_database.py::TestDatabaseSetup::test_database_enforces_data_source_constraint PASSED [ 16%]
tests/test_database.py::TestDatabaseSetup::test_database_enforces_bed_capacity_constraint PASSED [ 20%]
tests/test_schemas.py::TestSchemasValidation::test_detection_valid PASSED [ 24%]
tests/test_schemas.py::TestSchemasValidation::test_vehicle_report_valid PASSED [ 28%]
tests/test_schemas.py::TestSchemasValidation::test_occupant_report_valid PASSED [ 32%]
tests/test_schemas.py::TestSchemasValidation::test_accident_event_valid PASSED [ 36%]
tests/test_schemas.py::TestSchemasValidation::test_hospital_valid PASSED [ 40%]
tests/test_schemas.py::TestSchemasValidation::test_ambulance_valid PASSED [ 44%]
tests/test_schemas.py::TestSchemasValidation::test_route_result_valid PASSED [ 48%]
tests/test_schemas.py::TestSchemasValidation::test_audit_report_valid PASSED [ 52%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[Detection-sample_kwargs0] PASSED [ 56%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[VehicleReport-sample_kwargs1] PASSED [ 60%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[OccupantReport-sample_kwargs2] PASSED [ 64%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[AccidentEvent-sample_kwargs3] PASSED [ 68%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[Hospital-sample_kwargs4] PASSED [ 72%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[Ambulance-sample_kwargs5] PASSED [ 76%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[RouteResult-sample_kwargs6] PASSED [ 80%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[AuditReport-sample_kwargs7] PASSED [ 84%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_confidence PASSED [ 88%]
tests/test_schemas.py::TestSchemasRejection::test_reject_malformed_bbox PASSED [ 92%]
tests/test_schemas.py::TestSchemasRejection::test_reject_hospital_bed_inconsistency PASSED [ 96%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_coordinates PASSED [100%]

============================== 25 passed in 0.23s ==============================
```

---

## 7. Sign-Off & Next Steps

All 10 tasks defined in Phase 0 have been completed with zero errors. The foundation is ready for **Phase 1** upon user approval.
