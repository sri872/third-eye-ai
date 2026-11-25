# Phase 15 — Final Integrated Verification

Date: 2026-11-25  
Workspace: `/Users/srishanthbiradar/Projects/third-eye-v2`  
Python: `3.12.3`  
Final status: **implemented and verified with documented limitations**.

This document is the final deliverable for Phases 12–15. Prior phase reports:

- `PHASE12_REPORT.md` — dedicated plate detection.
- `PHASE13_REPORT.md` — scene/model evaluation.
- `PHASE14_REPORT.md` — video verification.

## 1. Preservation and baseline

| Check | Result |
|---|---|
| Phase 0 baseline | `149 passed, 12 warnings in 62.72s` |
| Final suite | `204 passed, 6 warnings in 36.80s` |
| Dependencies installed/upgraded | None |
| User files reset/deleted | None |
| Git diff/status | Unavailable: workspace has no `.git` directory |
| Six tabs | Present and verified by AppTest |
| Python | `3.12.3` |
| Platform | macOS / Apple Silicon, CPU fallbacks preserved |

Final suite output:
`/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase15_final_full_suite.txt`

Syntax/import compile check:

```text
.venv/bin/python -m compileall -q app core db scripts
compileall=ok
```

No failing test was deleted or weakened. Phase 11 truth rules remain in force:

- `PLATE_TRUST_THRESHOLD = 0.7`.
- `PlateOCREvidence` compatibility preserved.
- Exact raw OCR text/confidence preserved.
- `NOT READABLE (UNVERIFIED)` preserved.
- Model verdict authoritative; geometric rules explanatory only.
- Still images remain out-of-domain with `NOT DETERMINED (out-of-domain still image)`.

## 2. Startup and six-tab UI verification

Final AppTest smoke script:
`/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase15_app_smoke.py`  
Output:
`/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase15_app_smoke.out`

Initial run with no upload:

```text
initial_tabs ['1. Scene', '2. Vehicles', '3. Accident check', '4. Incident report', '5. Hospitals & emergency numbers', '6. Dispatch']
initial_number_inputs ['Video frames to sample']
initial_exception []
```

Then the real video was uploaded with the sample count set to 4:

```text
uploaded_exception []
uploaded_error []
tabs_after_upload ['1. Scene', '2. Vehicles', '3. Accident check', '4. Incident report', '5. Hospitals & emergency numbers', '6. Dispatch']
has_temporal_section True
has_video_result True
has_db_notice True
table_count 16
```

The launcher remains `launch.command` → `streamlit run app/main.py`. No FastAPI module exists in this workspace; no new web framework was introduced. The verified serving/UI surface is the existing Streamlit app plus script/module entry points.

A non-fatal offline Hugging Face warning appears during startup because the embedding model uses a cached commit while network access is disabled. The cached model loaded and the app ran with no exception.

## 3. Model and OCR provenance

Verified hashes:

| Artifact | SHA-256 |
|---|---|
| Plate detector ONNX | `6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7` |
| EasyOCR CRAFT | `4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17` |
| EasyOCR English gen2 | `e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8` |
| Accident classifier | `672f5fc05909bab42c85e70718f44bb4d15431724fbf4984b212350fb0e967a4` |
| Metrics artifact | `961b50ef98b8db723d2616183128146a40ca7c67c540b28147de1aa4dee87958` |
| Evaluation artifact | `16e93468dee6edb4b27c40562c70dba0bdf177c8419430b1ee32bf9c281777d2` |

OpenCV Zoo LPD-YuNet Apache-2.0 license is vendored at `models/plate_detector/LICENSE`. The model card warns it was trained on Chinese plates; it is used only to propose candidate regions, never as proof of a registration number.

Accident classifier facts:

- Frozen ResNet-18 + sklearn logistic regression.
- Held-out accuracy `0.6778`, precision `0.6600`, recall `0.7333`, F1 `0.6947`, ROC-AUC `0.7452`.
- Confusion matrix `[[56, 34], [24, 66]]`: FPR `0.3778`, FNR `0.2667`.
- Scores remain **uncalibrated ranking scores**; validation-only Platt scaling was investigated but not deployed.
- At default `0.50`, the model is imperfect. No threshold was tuned on held-out test data.

## 4. Real-media evidence carried forward

### Still image

`/Users/srishanthbiradar/Downloads/img1.jpeg`

- SHA-256: `3fdb9739daeba222e2782aabf02952577ee3a3dc6a81b3a6ba7883e02009c3c9`
- 7 vehicles; YuNet at 0.50 found no plate regions in production.
- Classifier score `0.2166` → `NO ACCIDENT SIGNAL`.
- Still-image status: `NOT DETERMINED (out-of-domain still image)`.

### Real video

`/Users/srishanthbiradar/Desktop/road vdo.mp4`

- SHA-256: `cdc01a90b62616b62abeeaa8110de9314a60c4dce0d140820cbd7d54c7a76809`
- 525 frames, 25 fps, 1920x1080, 21.0 s.
- 12-frame sample: peak `0.4201` at frame `381` / `15.24 s`, `UNCERTAIN`, isolated uncertain frame.
- 24-frame sample: peak `0.4594` at frame `410` / `16.4 s`, `UNCERTAIN`, sustained uncertain signal across 3 sampled frames.
- No `>=0.60` frame in either sample. No accident claim.

The final AppTest smoke used 4 sampled frames to verify integration, not accuracy.

## 5. Cross-module provenance

The final paths now carry evidence end-to-end:

1. **Detection/plate path**  
   `core/cv/detector.py` → `core/cv/plate_detector_yunet.py` → `core/cv/plate_detector.py` → `core/ocr/plate_reader.py` → `app/main.py` → Tab 2 → `VehicleReport` → SQLite.

2. **Video path**  
   `app/video_utils.py` → `app/main.py::_process_video` → `app/video_analysis.py` → `result["temporal_evidence"]` / `result["video_evidence"]` → Tab 3 → in-memory report → fusion → `IncidentReport` → SQLite.

3. **Fusion path**  
   `core/nlp/fusion.py` retains REAL VISION, SIMULATED CALLER, and REAL VIDEO frame/temporal evidence in separate narrative sections. It never lets caller evidence override the authoritative vision verdict.

4. **Database path**  
   Read-only verification found all nine expected tables:

```text
accident_events, ambulances, audit_reports, detections, hospitals,
incident_reports, occupant_reports, route_results, vehicle_reports
```

`vehicle_reports` contains Phase 12 columns:

```text
plate_detection_status, plate_bbox, plate_detector
```

`incident_reports` contains Phase 14 columns:

```text
video_evidence, video_assessment, video_temporal_signal
```

The read-only check did not write to the database:
`/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase15_db_routing_checks.out`

## 6. Routing and hospital data

Read-only checks:

```text
hospital elements: 116
route graph: 3247 nodes, 8582 edges
demo locations: 4
```

Assets are local:

- `data/hospitals_osm.json`
- `data/phagwara_corridor.osm`
- `data/phagwara_drive_network.graphml`
- `config/demo_incident_locations.json`
- `config/simulated_ambulances.json`

Routing remains offline/OSM-based. No live traffic or live dispatch was claimed.

## 7. Test coverage by phase

| Phase | Targeted evidence | Full suite |
|---|---|---|
| 12 | `67 passed, 6 warnings in 28.00s` | `185 passed, 6 warnings in 42.96s` |
| 13 | `26 passed, 5 warnings in 13.74s` | `193 passed, 6 warnings in 74.17s` |
| 14 | `32 passed, 5 warnings in 26.80s` | `200 passed, 6 warnings in 36.25s` |
| 15 | Artifact/AppTest integration checks | `204 passed, 6 warnings in 36.80s` |

Exact outputs:

- Phase 12: `/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase12_full_suite.txt`
- Phase 13: `/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase13_full_suite.txt`
- Phase 14: `/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase14_full_suite.txt`
- Phase 15: `/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase15_final_full_suite.txt`

## 8. Honest limitations

- No labelled ground truth exists for Indian license plates or accident-confirmed real video clips. Plate-detection and video-level accuracy are therefore not measured.
- YuNet was trained on Chinese plates; domain mismatch is documented.
- Video analysis samples evenly spaced frames; it is not full-clip inference or temporal tracking.
- Temporal labels are explanatory and do not override the trained model.
- The classifier is imperfect (FPR `0.3778`, FNR `0.2667` at 0.50) and uncalibrated.
- Cross-split near-duplicate frames indicate possible optimism in held-out metrics.
- No live dispatch, live traffic, confirmed accident, or verified registration number was produced.
- Git history/diff is unavailable because the workspace is not a Git repository.
- `data/third_eye.db` is an empty zero-byte file; the active database inspected here is `db/third_eye.db`.

## 9. Final sign-off

Phases 12–15 are complete with the following deliverables:

- Dedicated, hash-verified plate-region detection with heuristic fallback.
- Reproducible model evaluation and documented non-deployment of calibration.
- Configurable sampled-frame video verification with frame-level/temporal evidence.
- Full UI, fusion, schema, and SQLite propagation.
- Four phase reports plus this final verification document.
- Final full suite: **204 passed, 6 warnings in 36.80s**.

The project remains an evidence-assistance prototype requiring human verification. It does not claim accident confirmation, verified plates, live dispatch, or live traffic.
