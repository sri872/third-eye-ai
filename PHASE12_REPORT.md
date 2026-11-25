# Phase 12 Report — Dedicated Plate Detection

Date: 2026-11-25  
Workspace: `/Users/srishanthbiradar/Projects/third-eye-v2`  
Python: `3.12.3`  
Environment constraint: no dependencies installed or upgraded; existing FastAPI/Streamlit/SQLite architecture and all six tabs preserved.

## 1. Phase 0 baseline and preservation

A fresh baseline suite was run before implementation:

```text
.venv/bin/pytest --cache-clear -q
149 passed, 12 warnings in 62.72s (0:01:02)
```

Saved output: `/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase12_phase0_baseline.txt`.

Source inventory/hashes were captured before edits at:
`/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase12_phase0_source_hashes.txt`.

The workspace is **not a Git repository** (`fatal: not a git repository`). No reset, deletion, or overwrite of user data was performed. The Phase 11 trust rules were preserved:

- `PLATE_TRUST_THRESHOLD = 0.7` unchanged.
- `PlateOCREvidence.raw_text` / `raw_confidence` remain exact EasyOCR output.
- `NOT READABLE (UNVERIFIED)` remains the display for missing/failed reads.
- Model verdict remains authoritative; geometric rules remain explanatory and out of runtime use.
- Still-image verdict remains `NOT DETERMINED (out-of-domain still image)`.

## 2. Audit finding and detector decision

Initial local audit found no trusted plate detector wired into the project. EasyOCR itself was fully offline:

- `easyocr==1.7.2`, `download_enabled=False`, CPU.
- Local CRAFT and English gen2 weights exactly match EasyOCR configured MD5s.
- No in-repo plate annotations, ONNX plate model, Haar plate cascade, `onnxruntime`, `ultralytics`, `paddleocr`, or plate-specific package was present.

A later audit found a genuine OpenCV Zoo LPD-YuNet ONNX already on the user's Desktop. It was copied into the project, never downloaded as a model:

```text
models/plate_detector/license_plate_detection_lpd_yunet_2023mar.onnx
SHA-256: 6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7
Size:    4146213 bytes
```

This hash exactly matches the official OpenCV Zoo / Hugging Face LFS object. The Apache-2.0 license was vendored at `models/plate_detector/LICENSE`.

The upstream model card explicitly warns the model was trained on **Chinese license plates**. Therefore this detector is used only to propose candidate plate regions. It is not presented as validated for Indian plates, and its score is never treated as a calibrated probability or OCR confidence. A transparent deterministic OpenCV heuristic remains as a fallback only if the ONNX cannot load or infer.

## 3. Implementation

### New/modified code

- `core/cv/plate_detector_yunet.py`
  - SHA-256-verified model loading.
  - CPU `cv2.dnn` inference at the fixed upstream `320x240` input size, which keeps prior/output alignment correct.
  - Upstream-style prior boxes, score fusion `sqrt(class_score * iou_score)`, decoding, and OpenCV DNN NMS.
  - Graceful `None` on load failure.

- `core/cv/plate_detector.py`
  - `PlateRegion` now reports candidate bbox in local crop pixels and source-image pixels.
  - Default path prefers YuNet; deterministic `HEURISTIC_OPENCV_V1` is used only as documented fallback.
  - Invalid boxes, empty crops, and non-positive limits are rejected instead of guessed.

- `core/ocr/plate_reader.py`
  - Added `preprocess_plate_region()` with grayscale, CLAHE, light denoising, and aspect-preserving upscale capped at 4x. No binarisation or character synthesis.
  - `PlateOCREvidence` records detector status, plate bbox, detector method, and explicit OCR statuses: `EMPTY CROP`, `CROP TOO SMALL`, `OCR INITIALIZATION FAILURE`, `OCR INFERENCE FAILURE`, `NO OCR RESULTS`, `OK`.
  - Raw EasyOCR text/confidence are unchanged.

- `app/main.py`
  - Vehicle crops now run dedicated plate localisation first.
  - Only detected plate regions are passed through `preprocess_plate_region()` and EasyOCR.
  - Tab 2 shows vehicle ID/type/color, plate detection status, source-pixel plate bbox, exact raw OCR text, OCR confidence, OCR status, plate status, and REAL badge.
  - UI limitation text now states YuNet provenance, Chinese-plate training limitation, heuristic fallback, and the unchanged 0.7 verification rule.

- `core/cv/vehicle_report.py`
  - Batch/file-based vehicle reports now use the same detector-first pipeline and persist detector provenance in `VehicleReport`.

- `core/schemas.py`
  - `VehicleReport` carries optional `plate_detection_status`, `plate_bbox`, and `plate_detector`.

- `db/database.py`
  - Schema and backward-compatible migration add the three plate-provenance columns.
  - `save_vehicle_reports()` persists them; bbox is stored as JSON text.

- `tests/test_plate_detector.py` (new)
  - Valid bbox mapping, multiple plates, invalid bboxes, undersized plates, blank crop, prior/hash/model load, model preference, and bbox clamping.

- `tests/test_plate_reader.py`, `tests/test_app_ui.py`, `tests/test_database.py`
  - Updated for preprocessing, explicit OCR failures, no-region vs unreadable distinction, readable/unreadable statuses, and database provenance/migration.

## 4. Test evidence

Targeted Phase 12 suite:

```text
.venv/bin/pytest -q tests/test_plate_detector.py tests/test_plate_reader.py \
  tests/test_app_ui.py tests/test_schemas.py tests/test_database.py \
  tests/test_incident_text.py tests/test_fusion.py
67 passed, 6 warnings in 28.00s
```

Detector-only suite after the YuNet fixed-input correction:

```text
.venv/bin/pytest -q tests/test_plate_detector.py
8 passed in 0.30s
```

Full suite after all Phase 12 changes:

```text
.venv/bin/pytest --cache-clear -q
185 passed, 6 warnings in 42.96s
```

Saved output: `/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase12_full_suite.txt`.

No failing test was weakened or deleted. Baseline 149 tests remain covered; 36 new/updated assertions expand coverage.

## 5. Real-sample evidence

Diagnostic script:
`/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase12_real_plate_diag.py`  
Output:
`/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase12_real_plate_diag.out`

### `/Users/srishanthbiradar/Downloads/img1.jpeg`

- SHA-256: `3fdb9739daeba222e2782aabf02952577ee3a3dc6a81b3a6ba7883e02009c3c9`
- 1771x1237, 7 vehicle crops.
- YuNet at the production threshold 0.50 detected 0 plate regions in all 7 vehicles.
- At the experimental threshold 0.10 it produced 0, 0, 0, 1, 0, 1, 2 raw candidate boxes per vehicle, demonstrating that some low-confidence regions exist but are **not trusted**.
- Production result: all vehicles show `NO PLATE REGION DETECTED`; no plate text is invented.

### `data/samples/road_1.jpg`

- SHA-256: `b0fb3e4cd517c0d584b1e44a97c20b3fac135c9f25bc677cd967c1f2ae2588f7`
- 6 vehicle crops.
- YuNet at 0.50 detected candidate regions on vehicles 4 and 6:
  - veh04: score `0.5225`, bbox `(976, 562, 1024, 607)`, raw OCR `None`, plate `None`, `ocr_status=NO OCR RESULTS`.
  - veh06: score `0.8408`, bbox `(1016, 560, 1039, 606)`, raw OCR `None`, plate `None`, `ocr_status=NO OCR RESULTS`.
- No registration number is claimed.

### `data/samples/road_3.jpg`

- SHA-256: `916c2a0759ca4e12c22c9029e48c6e45f126f9e55f88dba1a74b6d377356f61e`
- 18 vehicle crops.
- YuNet at 0.50 detected one candidate on vehicle 13:
  - score `0.5936`, bbox `(938, 637, 978, 676)`, raw OCR `None`, plate `None`, `ocr_status=NO OCR RESULTS`.
- No registration number is claimed.

These are diagnostic observations, not an accuracy evaluation. There is no labelled plate ground truth in the project.

## 6. Honest limitations

- The learned detector is hash-verified and locally vendored, but it was trained on Chinese plates. Indian-plate recall/precision is **not measured** here.
- OCR produced no raw text on the three real candidate regions above. `NO OCR RESULTS` is reported explicitly and stays `NOT READABLE (UNVERIFIED)`.
- The heuristic fallback is not a learned detector and is not used when YuNet loads successfully.
- A candidate bbox is not proof of a physical plate or a registration number.
- No accident claim is made from plate detection. No live dispatch, live traffic, or confirmed accident was simulated or claimed.
- No held-out test data was used to tune plate OCR or detector thresholds.
- Git diff/status are unavailable because the workspace has no `.git` directory.

## 7. Phase 12 outcome

Phase 12 is complete with a dedicated plate-region detector integrated into both UI and batch paths, verified model provenance, explicit failure states, full test coverage, and real-sample diagnostic evidence. The implementation is honest about domain mismatch and does not claim verified plate reads where none exist.
