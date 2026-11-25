# Phase 1 Acceptance & Implementation Report

**Project**: Project Third-Eye (Multimodal Smart Ambulance Router & Automated Accident Forensic Auditor)  
**Phase**: Phase 1 (Computer Vision Detector)  
**Date**: 2026-11-24  
**Target Hardware**: MacBook Air M2 (Apple Silicon)  
**Environment**: Python 3.12.3 (.venv)  

---

## 1. Disk Space Tracking

- **Disk Space Before Installation**:
  ```text
  Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
  /dev/disk3s5   228Gi   163Gi    35Gi    83%    1.6M  368M    0%   /System/Volumes/Data
  ```
  *Initial Free Space*: **35 GiB**
- **Disk Space After Installation & Model Download**:
  ```text
  Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
  /dev/disk3s5   228Gi   163Gi    35Gi    83%    1.6M  369M    0%   /System/Volumes/Data
  ```
  *Final Free Space*: **35 GiB** (Clean installation with `--no-cache-dir`)

---

## 2. Dependency Installation & Pinned Versions

The following libraries were installed into `.venv` and pinned in `requirements.txt`:
- `pytest==9.2.0`
- `pydantic==2.14.4`
- `numpy==2.5.5`
- `streamlit==1.67.0`
- `torch==2.15.0`
- `torchvision==0.29.2`
- `opencv-python==5.1.0.95`
- `pillow==13.0.0`

Installation executed without errors.

---

## 3. Model Architecture & Device Execution

- **Architecture**: Faster R-CNN ResNet-50 FPN (`torchvision.models.detection.fasterrcnn_resnet50_fpn`)
- **Pretrained Weights**: `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` (COCO V1)
- **Model Checkpoint Size**: **159.74 MB** (`fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`)
- **Acceleration Device**:
  - Auto-selects Apple Silicon **MPS (Metal Performance Shaders)** when available:
    `[Third-Eye CV] Initializing Faster R-CNN ResNet-50 FPN on device: MPS`
  - Automatic fallback to **CPU** if MPS initialization or inference encounters an exception.
- **Allowed Class Filter**:
  Only traffic and accident-relevant classes are extracted:
  `{"person", "bicycle", "car", "motorcycle", "bus", "truck"}`
- **Box & Confidence Filtering**:
  - Confidence threshold configurable (default `0.50`). Confidence scores are extracted directly from model tensor outputs and are **never fabricated**.
  - Geometrically invalid boxes and out-of-bounds boxes (`xmin < 0`, `ymin < 0`, `xmax > width`, `ymax > height`, `xmin >= xmax`, `ymin >= ymax`) are rejected.
  - Every valid bounding box creates a verified `Detection` schema instance with `data_source="REAL"`.

---

## 4. Components Created

### `core/cv/detector.py`
- Lazy-loads Faster R-CNN model once as a singleton.
- Implements `get_target_device()` with MPS probe and CPU fallback.
- `detect(image, conf_threshold=0.5)` supports file path, PIL Image, or NumPy array.
- Returns `List[Detection]`.

### `core/cv/draw.py`
- `draw_detections(image_input, detections, output_path=None)` renders color-coded bounding boxes and readable labels with real confidence scores using OpenCV.
- Saves annotated images to `data/outputs/`.

### `scripts/run_detection.py`
- Dropdown-free, zero-typing CLI runner.
- Automatically discovers all supported images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) in `data/samples/`.
- Runs detection, prints real confidence scores and bounding boxes, and saves annotated images to `data/outputs/`.
- When `data/samples/` has no images, prompts the user clearly to add real photos without crashing or requiring arguments.

---

## 5. Test Suite Verification

Pytest executed with **30 passed, 1 skipped**:
- Phase 0 database and schema tests: **25 passed**.
- Phase 1 mocked detector tests: **4 passed** (class filtering, confidence thresholding, boundary rejection, device selection, rendering).
- Phase 1 real model integration test: **1 skipped** (skips cleanly because `data/samples/` currently contains no real photos; strictly adheres to the rule prohibiting synthetic or fake images).

### Test Output:
```text
============================= test session starts ==============================
platform darwin -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0 -- /Users/srishanthbiradar/Projects/third-eye-v2/.venv/bin/python3.12
cachedir: .pytest_cache
rootdir: /Users/srishanthbiradar/Projects/third-eye-v2
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.16.0
collecting ... collected 31 items

tests/test_database.py::TestDatabaseSetup::test_init_db_creates_all_tables PASSED [  3%]
tests/test_database.py::TestDatabaseSetup::test_foreign_keys_and_pragma_enabled PASSED [  6%]
tests/test_database.py::TestDatabaseSetup::test_insert_and_query_all_records PASSED [  9%]
tests/test_database.py::TestDatabaseSetup::test_database_enforces_data_source_constraint PASSED [ 12%]
tests/test_database.py::TestDatabaseSetup::test_database_enforces_bed_capacity_constraint PASSED [ 16%]
tests/test_detector.py::TestDetectorLogicMocked::test_filter_detections_class_filtering PASSED [ 19%]
tests/test_detector.py::TestDetectorLogicMocked::test_filter_detections_confidence_threshold PASSED [ 22%]
tests/test_detector.py::TestDetectorLogicMocked::test_filter_detections_reject_out_of_bounds_and_degenerate_boxes PASSED [ 25%]
tests/test_detector.py::TestDetectorLogicMocked::test_device_selection_mps_or_cpu PASSED [ 29%]
tests/test_detector.py::TestDetectorLogicMocked::test_draw_detections_renders_and_saves PASSED [ 32%]
tests/test_detector.py::TestRealModelIntegration::test_live_detector_on_sample_image SKIPPED [ 35%]
tests/test_schemas.py::TestSchemasValidation::test_detection_valid PASSED [ 38%]
tests/test_schemas.py::TestSchemasValidation::test_vehicle_report_valid PASSED [ 41%]
tests/test_schemas.py::TestSchemasValidation::test_occupant_report_valid PASSED [ 45%]
tests/test_schemas.py::TestSchemasValidation::test_accident_event_valid PASSED [ 48%]
tests/test_schemas.py::TestSchemasValidation::test_hospital_valid PASSED [ 51%]
tests/test_schemas.py::TestSchemasValidation::test_ambulance_valid PASSED [ 54%]
tests/test_schemas.py::TestSchemasValidation::test_route_result_valid PASSED [ 58%]
tests/test_schemas.py::TestSchemasValidation::test_audit_report_valid PASSED [ 61%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[Detection-sample_kwargs0] PASSED [ 64%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[VehicleReport-sample_kwargs1] PASSED [ 67%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[OccupantReport-sample_kwargs2] PASSED [ 70%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[AccidentEvent-sample_kwargs3] PASSED [ 74%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[Hospital-sample_kwargs4] PASSED [ 77%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[Ambulance-sample_kwargs5] PASSED [ 80%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[RouteResult-sample_kwargs6] PASSED [ 83%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_data_source[AuditReport-sample_kwargs7] PASSED [ 87%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_confidence PASSED [ 90%]
tests/test_schemas.py::TestSchemasRejection::test_reject_malformed_bbox PASSED [ 93%]
tests/test_schemas.py::TestSchemasRejection::test_reject_hospital_bed_inconsistency PASSED [ 96%]
tests/test_schemas.py::TestSchemasRejection::test_reject_invalid_coordinates PASSED [100%]

======================== 30 passed, 1 skipped in 2.05s =========================
```

---

## 6. Action Item for Live Photo Verification

To run the live Faster R-CNN model end-to-end on real photo data:
1. Place any real traffic, vehicle, or accident scene photo in `data/samples/` (e.g. `data/samples/accident1.jpg`).
2. Run `python scripts/run_detection.py` or re-run `pytest` — the skipped test will execute automatically and verify live detections.
