# Phase 2 Acceptance & Implementation Report

**Project**: Project Third-Eye (Multimodal Smart Ambulance Router & Automated Accident Forensic Auditor)  
**Phase**: Phase 2 (Vehicle Report Generator & Plate OCR)  
**Date**: 2026-11-24  
**Target Hardware**: MacBook Air M2 (Apple Silicon)  
**Environment**: Python 3.12.3 (.venv)  

---

## 1. Disk Space Tracking

- **Free Disk Space Before Installation**:
  ```text
  Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
  /dev/disk3s5   228Gi   160Gi    39Gi    81%    1.6M  408M    0%   /System/Volumes/Data
  ```
  *Initial Free Space*: **39 GiB**
- **Free Disk Space After Installation & Model Download**:
  ```text
  Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
  /dev/disk3s5   228Gi   160Gi    39Gi    81%    1.6M  405M    0%   /System/Volumes/Data
  ```
  *Final Free Space*: **39 GiB**

---

## 2. Dependency & OCR Model Details

- **Installed Library**: `easyocr==1.7.2` (pinned in `requirements.txt`).
- **OCR Engine**: CRAFT detector + English ResNet/LSTM recognizer (`easyocr.Reader(['en'])`).
- **Downloaded OCR Models**:
  - `craft_mlt_25k.pth`: **79.30 MB**
  - `english_g2.pth`: **14.44 MB**
  - **Total OCR Model Size**: **93.74 MB**
- **Device Used**: **CPU** (EasyOCR on Apple Silicon runs on CPU; logged on startup as `[Third-Eye OCR] Initializing EasyOCR Reader on device: CPU`).

---

## 3. Schema & Database Schema Extensions

To store forensic attributes extracted from vehicles, the existing `VehicleReport` schema in `core/schemas.py` and the SQLite `vehicle_reports` table in `db/database.py` were carefully extended:

### Fields Added to `VehicleReport`:
- `color: Optional[str] = Field(default=None, description="Dominant vehicle color: white, black, grey, red, blue, green, yellow, other")`
- `color_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Dominant color pixel share measure [0.0 - 1.0]")`
- `plate_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="OCR plate reading confidence [0.0 - 1.0]")`

### SQLite Table Update (`db/database.py`):
```sql
CREATE TABLE IF NOT EXISTS vehicle_reports (
    vehicle_id TEXT PRIMARY KEY,
    license_plate TEXT,
    plate_confidence REAL CHECK(plate_confidence IS NULL OR (plate_confidence >= 0.0 AND plate_confidence <= 1.0)),
    vehicle_type TEXT,
    color TEXT,
    color_confidence REAL CHECK(color_confidence IS NULL OR (color_confidence >= 0.0 AND color_confidence <= 1.0)),
    damage_severity TEXT,
    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    timestamp TEXT,
    data_source TEXT NOT NULL CHECK(data_source IN ('REAL', 'SIMULATED'))
);
```
All added fields are strictly optional (`None` when not detected or unreadable) with valid `[0.0, 1.0]` constraints, preserving 100% backward compatibility with Phase 0 and Phase 1 tests.

---

## 4. Components Created

### `core/ocr/plate_reader.py`
- Implements lazy loading of `easyocr.Reader(['en'])`.
- Normalizes plate text (`uppercase`, removes spaces and non-alphanumeric symbols).
- Never guesses or invents characters. Returns `(None, None)` if no readable plate text is found.
- Returns `(plate_text, real_ocr_confidence)`.

### `core/cv/vehicle_attributes.py`
- `crop_vehicle`: Safe cropping with coordinate boundary clamping and out-of-bounds error handling.
- `dominant_color`: Performs OpenCV `cv2.kmeans` clustering and maps cluster centers in HSV space to a named palette (`white`, `black`, `grey`, `red`, `blue`, `green`, `yellow`, `other`). Returns `(color_name, pixel_share)`.
- `get_vehicle_type`: Converts detection labels (`car`, `bus`, `truck`, `motorcycle`, `bicycle`) into standardized vehicle types. Rejects non-vehicles (e.g. `person`).

### `core/cv/vehicle_report.py`
- Orchestrates detector -> safe crop -> dominant color -> plate OCR -> `VehicleReport` schema (`data_source="REAL"`).
- Automatically persists reports into SQLite table `vehicle_reports`.
- Draws forensic overlays on cropped vehicle images and saves them to `data/outputs/crops/`.

### `scripts/run_vehicle_reports.py`
- Zero-typing CLI batch runner.
- Processes all images in `data/samples/`.
- Prints per-vehicle summary and honest plate counts.

---

## 5. Live Processing & Honest Count Results

Execution of `scripts/run_vehicle_reports.py` on real images in `data/samples/`:

```text
==================================================================
Project Third-Eye | Vehicle Report Generator (Phase 2)
Models: Faster R-CNN ResNet-50 FPN + EasyOCR (English)
UI Principle: Zero typing, automated batch execution
==================================================================

Discovered 3 image(s) in /Users/srishanthbiradar/Projects/third-eye-v2/data/samples.

[1/3] Processing: road_1.jpg
[Third-Eye CV] Initializing Faster R-CNN ResNet-50 FPN on device: CPU
[Third-Eye OCR] Initializing EasyOCR Reader on device: CPU
  Extracted 6 vehicle report(s):
    - ID: veh-road_1-01-333d           | Type: car        | Color: red (share: 0.44)      | Plate: 'CAN09G' (conf: 0.5412)          | DetConf: 0.9990
    - ID: veh-road_1-02-da46           | Type: car        | Color: green (share: 0.30)    | Plate: not readable                     | DetConf: 0.9988
    - ID: veh-road_1-03-3b2e           | Type: car        | Color: red (share: 0.58)      | Plate: not readable                     | DetConf: 0.9912
    - ID: veh-road_1-04-34f6           | Type: car        | Color: grey (share: 0.73)     | Plate: not readable                     | DetConf: 0.9155
    - ID: veh-road_1-05-9ee0           | Type: car        | Color: grey (share: 0.59)     | Plate: not readable                     | DetConf: 0.6039
    - ID: veh-road_1-06-cb88           | Type: car        | Color: green (share: 0.49)    | Plate: not readable                     | DetConf: 0.5256
  Saved 6 annotated crop(s) to data/outputs/crops/

[2/3] Processing: road_2.jpg
  Extracted 6 vehicle report(s):
    - ID: veh-road_2-01-d225           | Type: car        | Color: red (share: 0.44)      | Plate: 'CAN09G' (conf: 0.5412)          | DetConf: 0.9990
    - ID: veh-road_2-02-3724           | Type: car        | Color: green (share: 0.30)    | Plate: not readable                     | DetConf: 0.9988
    - ID: veh-road_2-03-bd51           | Type: car        | Color: red (share: 0.58)      | Plate: not readable                     | DetConf: 0.9912
    - ID: veh-road_2-04-723f           | Type: car        | Color: grey (share: 0.73)     | Plate: not readable                     | DetConf: 0.9155
    - ID: veh-road_2-05-ab78           | Type: car        | Color: grey (share: 0.61)     | Plate: not readable                     | DetConf: 0.6039
    - ID: veh-road_2-06-2b31           | Type: car        | Color: green (share: 0.49)    | Plate: not readable                     | DetConf: 0.5256
  Saved 6 annotated crop(s) to data/outputs/crops/

[3/3] Processing: road_3.jpg
  Extracted 18 vehicle report(s):
    - ID: veh-road_3-01-666c           | Type: car        | Color: grey (share: 0.72)     | Plate: not readable                     | DetConf: 0.9946
    - ID: veh-road_3-02-829e           | Type: car        | Color: grey (share: 0.70)     | Plate: not readable                     | DetConf: 0.9919
    - ID: veh-road_3-03-883e           | Type: car        | Color: grey (share: 0.87)     | Plate: not readable                     | DetConf: 0.9831
    - ID: veh-road_3-04-97bc           | Type: car        | Color: grey (share: 0.46)     | Plate: not readable                     | DetConf: 0.9800
    - ID: veh-road_3-05-c943           | Type: car        | Color: grey (share: 0.80)     | Plate: not readable                     | DetConf: 0.9714
    - ID: veh-road_3-06-e653           | Type: car        | Color: grey (share: 0.62)     | Plate: not readable                     | DetConf: 0.9713
    - ID: veh-road_3-07-1179           | Type: car        | Color: grey (share: 0.67)     | Plate: not readable                     | DetConf: 0.9658
    - ID: veh-road_3-08-f6d9           | Type: car        | Color: grey (share: 0.74)     | Plate: not readable                     | DetConf: 0.9572
    - ID: veh-road_3-09-ce5e           | Type: car        | Color: grey (share: 0.83)     | Plate: not readable                     | DetConf: 0.9206
    - ID: veh-road_3-10-b5d9           | Type: car        | Color: grey (share: 0.55)     | Plate: not readable                     | DetConf: 0.9169
    - ID: veh-road_3-11-1b16           | Type: truck      | Color: grey (share: 0.79)     | Plate: not readable                     | DetConf: 0.8861
    - ID: veh-road_3-12-ee29           | Type: car        | Color: grey (share: 0.85)     | Plate: not readable                     | DetConf: 0.8719
    - ID: veh-road_3-13-4e80           | Type: car        | Color: grey (share: 0.75)     | Plate: not readable                     | DetConf: 0.7454
    - ID: veh-road_3-14-563c           | Type: car        | Color: grey (share: 1.00)     | Plate: not readable                     | DetConf: 0.7051
    - ID: veh-road_3-15-66fd           | Type: car        | Color: grey (share: 0.91)     | Plate: not readable                     | DetConf: 0.6271
    - ID: veh-road_3-16-be6c           | Type: car        | Color: grey (share: 0.75)     | Plate: not readable                     | DetConf: 0.5414
    - ID: veh-road_3-17-ae67           | Type: truck      | Color: white (share: 0.52)    | Plate: not readable                     | DetConf: 0.5108
    - ID: veh-road_3-18-ce38           | Type: car        | Color: grey (share: 0.78)     | Plate: not readable                     | DetConf: 0.5010
  Saved 18 annotated crop(s) to data/outputs/crops/

==================================================================
PHASE 2 VEHICLE REPORT SUMMARY (HONEST COUNT):
  Total Images Processed   : 3
  Total Vehicles Detected  : 30
  Vehicles with READABLE Plate   : 2
  Vehicles with NOT READABLE Plate: 28
  Annotated Crops Directory: data/outputs/crops
  Database Records Stored  : db/third_eye.db -> table 'vehicle_reports'
==================================================================
```

### Honest Count Analysis:
- **Total Vehicles Detected**: **30**
- **Vehicles with READABLE Plate**: **2**
  - Detected on the high-resolution foreground vehicle in `road_1.jpg` and `road_2.jpg` (`'CAN09G'`, real OCR confidence `0.5412`).
- **Vehicles with NOT READABLE Plate**: **28**
  - Distant/lower-resolution vehicles where plate text is below OCR threshold or blurred are recorded honestly as `license_plate=None` and `plate_confidence=None`. Zero plates fabricated.

---

## 6. Test Suite Execution (`pytest`)

**Command:** `.venv/bin/pytest`
```text
============================= test session starts ==============================
platform darwin -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0 -- /Users/srishanthbiradar/Projects/third-eye-v2/.venv/bin/python3.12
cachedir: .pytest_cache
rootdir: /Users/srishanthbiradar/Projects/third-eye-v2
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.16.0
collecting ... collected 43 items

tests/test_database.py::TestDatabaseSetup::test_init_db_creates_all_tables PASSED [  2%]
tests/test_database.py::TestDatabaseSetup::test_foreign_keys_and_pragma_enabled PASSED [  4%]
tests/test_database.py::TestDatabaseSetup::test_insert_and_query_all_records PASSED [  6%]
tests/test_database.py::TestDatabaseSetup::test_save_vehicle_reports_helper PASSED [  9%]
tests/test_database.py::TestDatabaseSetup::test_database_enforces_data_source_constraint PASSED [ 11%]
tests/test_database.py::TestDatabaseSetup::test_database_enforces_bed_capacity_constraint PASSED [ 13%]
tests/test_detector.py::TestDetectorLogicMocked::test_filter_detections_class_filtering PASSED [ 16%]
tests/test_detector.py::TestDetectorLogicMocked::test_filter_detections_confidence_threshold PASSED [ 18%]
tests/test_detector.py::TestDetectorLogicMocked::test_filter_detections_reject_out_of_bounds_and_degenerate_boxes PASSED [ 20%]
tests/test_detector.py::TestDetectorLogicMocked::test_device_selection_mps_or_cpu PASSED [ 23%]
tests/test_detector.py::TestDetectorLogicMocked::test_draw_detections_renders_and_saves PASSED [ 25%]
tests/test_detector.py::TestRealModelIntegration::test_live_detector_on_sample_image PASSED [ 27%]
tests/test_plate_reader.py::TestPlateReaderMocked::test_normalize_plate_text PASSED [ 30%]
tests/test_plate_reader.py::TestPlateReaderMocked::test_read_plate_with_mocked_success PASSED [ 32%]
tests/test_plate_reader.py::TestPlateReaderMocked::test_read_plate_with_mocked_unreadable PASSED [ 34%]
tests/test_plate_reader.py::TestPlateReaderMocked::test_read_plate_selects_highest_confidence PASSED [ 37%]
tests/test_plate_reader.py::TestPlateReaderLive::test_live_ocr_on_real_sample_photo PASSED [ 39%]
tests/test_schemas.py::TestSchemasValidation::test_detection_valid PASSED [ 41%]
tests/test_schemas.py::TestSchemasValidation::test_vehicle_report_valid PASSED [ 44%]
tests/test_schemas.py::TestSchemasValidation::test_occupant_report_valid PASSED [ 46%]
tests/test_schemas.py::TestSchemasValidation::test_accident_event_valid PASSED [ 48%]
tests/test_schemas.py::TestSchemasValidation::test_hospital_valid PASSED [ 51%]
tests/test_schemas.py::TestSchemasValidation::test_ambulance_valid PASSED [ 53%]
tests/test_schemas.py::TestSchemasValidation::test_route_result_valid PASSED [ 55%]
tests/test_schemas.py::TestSchemasValidation::test_audit_report_valid PASSED [ 58%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[Detection-sample_kwargs0] PASSED [ 60%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[VehicleReport-sample_kwargs1] PASSED [ 62%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[OccupantReport-sample_kwargs2] PASSED [ 65%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[AccidentEvent-sample_kwargs3] PASSED [ 67%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[Hospital-sample_kwargs4] PASSED [ 69%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[Ambulance-sample_kwargs5] PASSED [ 72%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[RouteResult-sample_kwargs6] PASSED [ 74%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[AuditReport-sample_kwargs7] PASSED [ 76%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_confidence PASSED [ 79%]
tests/test_schemas.py::TestSchemasRejection::test_reject_malformed_bbox PASSED [ 81%]
tests/test_schemas.py::TestSchemasRejection::test_reject_hospital_bed_inconsistency PASSED [ 83%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_coordinates PASSED [ 86%]
tests/test_vehicle_attributes.py::TestVehicleCropping::test_crop_vehicle_clamping PASSED [ 88%]
tests/test_vehicle_attributes.py::TestVehicleCropping::test_crop_vehicle_invalid_box_raises PASSED [ 90%]
tests/test_vehicle_attributes.py::TestVehicleTypeMapping::test_valid_vehicle_types PASSED [ 93%]
tests/test_vehicle_attributes.py::TestVehicleTypeMapping::test_non_vehicle_labels_rejected PASSED [ 95%]
tests/test_vehicle_attributes.py::TestDominantColorExtraction::test_classify_pure_bgr_colors PASSED [ 97%]
tests/test_vehicle_attributes.py::TestDominantColorExtraction::test_dominant_color_on_real_sample_crop PASSED [100%]

======================== 43 passed in 7.24s ========================
```

---

## 7. Problems Encountered & Resolutions

1. **EasyOCR Sandbox Storage Path**: By default, EasyOCR attempts to access and write to `~/.EasyOCR/`. Inside sandbox boundaries, this caused permission restrictions. Resolved by configuring `model_storage_directory` and `user_network_directory` directly inside the workspace at `models/easyocr/`.
2. **Crop Clamping on Fully Out-Of-Bounds Coordinates**: When given bounding box coordinates completely outside image bounds, standard clamping yielded 1x1 edge pixel slices. Added explicit boundary validation in `crop_vehicle` to raise `ValueError` when bounding boxes fall completely outside the image frame.
3. **Automatic Table Initialization**: Enhanced `save_vehicle_reports` to ensure the SQLite schema is initialized automatically if invoked on a fresh database.
