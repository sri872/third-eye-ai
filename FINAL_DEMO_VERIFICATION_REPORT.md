# THIRD EYE AI — Final Demonstration Verification Report

> **Architecture supersession (2026-11-25):** This report records measurements from the earlier hospital retrieval implementation. The FAISS/SentenceTransformers vector-search path has been removed on branch `remove-rag-retrieval`; the current branch uses deterministic OpenStreetMap facility classification, source-reported emergency tags, facility category, and Haversine distance. Historical measurements below do not validate the replacement ranking. See [docs/SYSTEM_CARD.md](docs/SYSTEM_CARD.md) for the current architecture.


**Repository:** <https://github.com/sri872/third-eye-ai> (public)
**Workspace:** `~/Projects/third-eye-v2`
**Date:** 2026-11-25
**Final test status:** `232 passed, 14 warnings in 68.05s (0:01:08)` — 0 failures, 0 skips
**Overall status:** **DEMONSTRATION READY — all four mandatory inputs verified with real inference and all six Streamlit tabs verified.**

This is the authoritative report. Earlier phase documents, including `DEMO_READINESS_REPORT.md`, contain superseded mid-project values and are retained only as historical evidence. All numbers below were measured from the committed code and the four mandatory demo files; no result is hardcoded.

---

## 1. What was actually verified

| Verification layer | Method | Result |
|---|---|---|
| Real inference on four inputs | `scripts/verify_demo_assets.py` | PASS — detector, YuNet, EasyOCR, classifier and video sampler all executed on the real files |
| Six-tab Streamlit workflow | `scripts/verify_streamlit_workflow.py` (AppTest, real bytes, no model monkeypatching) | PASS — 4/4 inputs × 6/6 tabs |
| Live browser UI | Streamlit on `localhost:8512`, real image upload and simulated-call selection | PASS — tabs render, source disagreement visible, no browser console errors |
| Repeated-input / stale-cache behaviour | AppTest switching A → B → A and changing the sidebar location | PASS — SHA-256 and coordinates follow the current selection |
| Full regression suite | `.venv/bin/python -m pytest -q` | PASS — 232 passed, 14 warnings |
| Secret scan | Recursive text regex scan excluding binaries/caches | PASS — 0 credentials found |
| GitHub publication | `origin` → `https://github.com/sri872/third-eye-ai.git` | Public repository, initial release pushed; final follow-up commit prepared |

Machine-readable evidence:

- `data/outputs/demo_verification.json` — real detector/OCR/classifier/video output
- `data/outputs/streamlit_workflow.json` — six-tab AppTest results and input-switching evidence
- `data/outputs/dispatch_summary.json` — stored simulated dispatch replay used by Tab 6
- `GITHUB_FILE_AUDIT.md` — complete tracked/excluded asset inventory and hashes

---

## 2. Mandatory input integrity

| File | Bytes | SHA-256 | Decoded / verified |
|---|---:|---|---|
| `single_car_plate.jpg` | 114,036 | `246cf1cd4dc1cdc58592c4b92878289c70a71c051039d3c71e4a417322fb612b` | JPEG 900×1600, analysed after 90° rotation as 1600×900 |
| `two_car_collision.jpg` | 312,179 | `3fdb9739daeba222e2782aabf02952577ee3a3dc6a81b3a6ba7883e02009c3c9` | JPEG 1771×1237 |
| `heavy_traffic.jpg` | 172,786 | `eb25eb9229002de506d4c622bd43793c461b6ddeb51afb5f4605d25be77967b9` | JPEG 1270×885 |
| `road vdo.mp4` | 14,323,936 | `cdc01a90b62616b62abeeaa8110de9314a60c4dce0d140820cbd7d54c7a76809` | H.264 MP4, 1920×1080, 25 fps, 525 frames, 21.0 s |

The SHA-256 shown in Tab 1 matches the uploaded bytes. Uploaded images stay in memory; videos use an auto-deleted temporary file outside the project; SQLite is opened read-only (`mode=ro`). No uploaded byte is written into the repository.

---

## 3. Measured output by input

### 3.1 `single_car_plate.jpg`

- **Orientation:** source had zero accepted detections; a 90° analysis rotation produced 1 detection with top confidence **0.9199**. The source file is unchanged.
- **Faster R-CNN box:** `car`, confidence **0.9199**, bbox `[353.91, 119.53, 1274.37, 879.43]` in the displayed 1600×900 analysis image.
- **Scene counts (browser):** 1 vehicle, 0 persons; 0 heavily overlapping vehicle-box pairs.
- **Plate region:** YuNet candidate bbox `(628, 780, 949, 840)`.
- **EasyOCR evidence:** raw text `IND`, raw confidence **0.9457**; registration-format candidate `TS07JA0566`, candidate confidence **0.4348**.
- **0.70 trust gate:** because `0.4348 < 0.70`, the Plate column and Plate status read **`NOT READABLE (UNVERIFIED)`**. The candidate is retained separately as `TS07JA0566 (0.43)` and the raw evidence as `IND (0.95)`. No registration number is presented as verified.
- **Accident check:** score **0.4230** → `UNCERTAIN`; scene status explicitly `NOT DETERMINED (out-of-domain still image)`.
- **Incident report:** vision-only provenance; no caller evidence is invented.

### 3.2 `two_car_collision.jpg`

- **Faster R-CNN:** 9 real detections — 7 cars and 2 persons. Top cars scored **0.9992** and **0.9978**.
- **Classifier:** score **0.2166** → **`NO ACCIDENT SIGNAL`**. This is an uncalibrated ranking signal from a model trained on forward-facing dashcam clips, not proof that no collision happened.
- **Caller pairing:** with `call_demo_01_collision.txt`, the caller reports two cars, one possible injury and emergency assistance. These remain `SIMULATED`.
- **Source disagreement (browser-confirmed):** the report displays
  `SOURCE DISAGREEMENT: CALLER-REPORTED ACCIDENT — VISUAL MODEL DOES NOT CONFIRM.`
  followed by the REAL vision verdict and the SIMULATED caller evidence. The visual verdict is **not** overridden.
- **Other disagreements retained separately:** caller reported 2 vehicles vs 7 detected; caller mentioned 1 person vs 2 detected; image-associated demo location NH-44 vs caller location NH-44 (location comparison shown explicitly, not silently merged).

### 3.3 `heavy_traffic.jpg`

- **Faster R-CNN:** 45 detections — `person=23`, `motorcycle=10`, `car=12` (22 vehicles total).
- **Classifier:** score **0.3903** → **`NO ACCIDENT SIGNAL`**. Density alone is not treated as a collision.
- **Incident report:** vision source labels remain `REAL (VISION)`; caller evidence is absent or `MISSING` as appropriate.

### 3.4 `road vdo.mp4`

- **Container metadata:** 525 frames, 25 fps, 21.0 s, 1920×1080.
- **Sampling:** 12 evenly spaced frames; frame number and timestamp are preserved. Every sampled frame gets real detection and classifier inference.
- **Scores:** range **0.1915–0.4201**, mean **0.2872**. Peak at source frame **381** / **15.24 s** with **0.4201** → `UNCERTAIN`.
- **Temporal signal:** **`ISOLATED UNCERTAIN FRAME IN SAMPLE`**.
- **Assessment wording:** the app states this is an even-sample review, not full-clip inference, and not an accident confirmation. Temporal evidence is explanatory only and never overrides the trained-model verdict.
- **OCR:** runs only on the peak-scoring sampled frame; no plate candidate is promoted without the 0.70 gate.

---

## 4. Six technology verification

| # | Technology | Real implementation and measured evidence | Status |
|---|---|---|---|
| 1 | Computer Vision | TorchVision Faster R-CNN ResNet-50 FPN detects 1 / 9 / 45 objects on the three images and vehicles on all 12 video samples; 90° orientation retry recovers the portrait car photo; YuNet ONNX proposes plate regions. MPS with CPU fallback. | **PASS** |
| 2 | OCR | Local EasyOCR reads the plate candidate on `single_car_plate.jpg`: raw `IND (0.9457)`, candidate `TS07JA0566 (0.4348)`. Raw text and candidate are retained; the 0.70 gate keeps the verified Plate field `NOT READABLE (UNVERIFIED)`. | **PASS** |
| 3 | Machine Learning | Trained logistic-regression classifier in `models/accident_classifier.joblib` consumes frozen ResNet-18 embeddings. Measured scores 0.4230 / 0.2166 / 0.3903 / video peak 0.4201. Held-out ROC-AUC 0.7452 on 60 videos; score is explicitly uncalibrated. | **PASS** |
| 4 | Deep Learning | Neural inference in Faster R-CNN (detection), ResNet-18 (embeddings) and SentenceTransformer `all-MiniLM-L6-v2` (hospital embeddings + FAISS index). Device labels shown in the UI: MPS / CPU / `mps:0`. | **PASS** |
| 5 | NLP | spaCy-based caller extraction identifies vehicles, people, location, injuries, urgency, accident status and contradictions. Fusion keeps REAL vision and SIMULATED caller facts in separate tables and emits the disagreement text required by the brief. | **PASS** |
| 6 | RAG | Cached OSM records: 115 raw facilities → 100 dispatch-eligible after classification; 17 phones, 3 emergency tags, 2 specialty records. SentenceTransformer + FAISS retrieval with rule-built query and proximity re-ranking. Attribution `© OpenStreetMap contributors`; emergency tags are source-reported, not independently verified. | **PASS** |

Supporting paths verified: `core/routing/` (A*, simulated dispatch replay), `core/schemas.py` (`REAL`/`SIMULATED` provenance), `db/database.py` (SQLite schema used by CLI scripts), `app/dispatch_ui.py` (read-only summary rendering).

---

## 5. Six-tab UI verification

| Tab | Verification evidence | Status |
|---|---|---|
| **1. Scene** | Browser upload of `single_car_plate.jpg`: real detector boxes, 1 car / 0 persons, orientation note, SHA-256, detection threshold and occlusion note all render. | **PASS** |
| **2. Vehicles** | Browser table shows plate status `NOT READABLE (UNVERIFIED)`, bbox `628,780,949,840`, candidate `(0.43)` and raw `IND (0.95)` in separate columns; YuNet method and limitation text visible. | **PASS** |
| **3. Accident check** | Browser shows `Model verdict: UNCERTAIN`, uncalibrated score `0.4230`, held-out metrics (ROC-AUC 0.7452) and model-card limitations. AppTest shows video metadata, 12 frame rows with timestamps and the temporal assessment. | **PASS** |
| **4. Incident report** | Browser shows source-labelled tables (`REAL (VISION)`, `SIMULATED (CALLER)`, `MISSING`, `OPERATIONAL RULE`) and the caller-accident disagreement without overriding the visual verdict. | **PASS** |
| **5. Hospitals & emergency numbers** | Browser shows simulated location/coordinates, 115 real OSM records, 100 eligible facilities, rule-based query, Akash Hospital as the source-confirmed emergency candidate, straight-line distances and `not determined` for unverified fields. | **PASS** |
| **6. Dispatch** | Browser shows `SIMULATED DISPATCH — NO REAL AMBULANCE HAS BEEN CONTACTED`, chosen ambulance `amb-phagwara-04` with ETA `545.5 s` and road distance `3.98 km`, real OSM hospital record, route legs, map fallback and Google Maps links. Distinct Tab 5 and Tab 6 locations are retained without substitution. | **PASS** |

AppTest also verified that all six tabs render for all four mandatory inputs, with `report_has_source_separation`, `hospital_tab_mentions_osm` and `dispatch_tab_is_simulated` true in each run.

### Live browser limitation

The browser automation uploader has a **5 MiB total per-upload limit**. `road vdo.mp4` is **14,323,936 bytes**, so the tool rejected it with `Upload is 14323936 bytes; the limit is 5242880 bytes`. The video path was therefore verified with real bytes through AppTest and `scripts/verify_demo_assets.py`; the local Streamlit app itself accepts the video through the normal browser uploader (200 MB configured limit). This is a harness limitation, not an app limitation.

---

## 6. Stale-cache and provenance checks

- **Input switching:** AppTest uploaded collision → single-car → collision. SHA-256 values were `3fdb97…` → `246cf1…` → `3fdb97…`; the final result matched the first, proving no stale result leaked.
- **Location change:** changing the sidebar location from the NH-44 demo location to Banga Road changed `uploaded_result_key` and the `AccidentEvent` coordinates, so Tab 4 can no longer show old coordinates while Tabs 5/6 use a new location.
- **Byte change under the same filename:** a regression test confirms that different bytes force reprocessing.
- **Caller claims:** caller-reported accidents, injuries, hazards and urgency are labelled SIMULATED and never become vision predictions. If the visual model does not confirm the caller's accident claim, the exact disagreement sentence is displayed.
- **Report provenance:** every report field carries its own `Source` column; unknown values are `not determined` or `MISSING`. No silent merge of image and caller facts occurs.

---

## 7. Errors found and corrections applied

| Issue | Correction | Regression coverage |
|---|---|---|
| `scripts/verify_demo_assets.py` failed with `ModuleNotFoundError: No module named 'core'` | Inserted repository root into `sys.path` | Script rerun successfully |
| Video verification printed hardcoded `Vehicles: 1 / Persons: 0` | Replaced with real `detect(frame.image)` counts for each sampled frame | `demo_verification.json` frame rows |
| Standalone verification used a different plate crop than the UI, giving different OCR values | Mirrored `crop_vehicle`, source bbox and raw/preprocessed candidate selection from `app/main.py`; rerun now matches Tab 2 exactly (`TS07JA0566`, `0.4348`, raw `IND`, `0.9457`) | Real demo-plate AppTest + direct script |
| Low-confidence plate text was promoted into the Plate column as `CAN09G (unverified)` | Plate column now stays `NOT READABLE (UNVERIFIED)`; raw text and candidate moved to separate columns | `test_vehicles_tab_shows_low_confidence_plate_as_not_readable`, `test_real_demo_plate_stays_not_readable_below_threshold` |
| Changing the sidebar location left cached event coordinates stale | Added `location` to `upload_key` | `test_changing_demo_location_invalidates_cached_vision_result` |
| A caller saying only “accident” did not trigger source disagreement | Added `accident_status == "reported"` to conflict details | `test_caller_only_accident_claim_is_source_disagreement_not_verdict` |
| RAG text claimed `24/7 emergency department available` from an OSM tag | Changed to source-reported, with 24/7 operation not independently verified | `test_tab5_hospitals` still passes |
| Browser initially loaded old code on ports 8501/8502 from earlier sessions | Fresh server on port 8512; removed stale “Phase 8a” title/caption in favour of `Third Eye AI | Six-Tab Demonstration` | Browser confirmed new title and six tabs |
| Harness video upload limit of 5 MiB | Documented and used AppTest/direct pipeline for the 14 MB video | Browser error captured; AppTest video checks pass |

No failure was hidden by changing a threshold, mocking a model, or hardcoding an expected result.

---

## 8. Test and verification commands

```bash
cd ~/Projects/third-eye-v2
source .venv/bin/activate

# Full suite
.venv/bin/python -m pytest -q

# Real inference on all four mandatory files
.venv/bin/python scripts/verify_demo_assets.py

# Real Streamlit AppTest workflow across all six tabs and four files
.venv/bin/python scripts/verify_streamlit_workflow.py
```

Launch the app:

```bash
cd ~/Projects/third-eye-v2
source .venv/bin/activate
streamlit run app/main.py
```

Open <http://localhost:8501>. On macOS, `launch.command` does the same thing.

---

## 9. Professor demonstration sequence

1. **Launch:** run `./launch.command`; open `http://localhost:8501`.
2. **Single car:** upload `data/samples/demo/single_car_plate.jpg`. In Tab 1 point out the 90° analysis rotation and the real car box at `0.9199`. In Tab 2 show the plate candidate, raw `IND (0.95)` and **`NOT READABLE (UNVERIFIED)`** because the candidate is only `0.43 < 0.70`.
3. **Collision + caller:** upload `two_car_collision.jpg`, then select `call_demo_01_collision.txt`. Tab 3 shows the real `0.2166` / `NO ACCIDENT SIGNAL` result. Tab 4 shows the caller's report in a separate `SIMULATED` section and the exact `CALLER-REPORTED ACCIDENT — VISUAL MODEL DOES NOT CONFIRM` disagreement.
4. **Heavy traffic:** upload `heavy_traffic.jpg`. Tab 1 shows 23 people / 10 motorcycles / 12 cars; Tab 3 shows `0.3903` / `NO ACCIDENT SIGNAL`, demonstrating no false collision alarm from density.
5. **Road video:** upload `road vdo.mp4` with 12 samples. Tab 3 shows real metadata, 12 timestamped frame scores, the isolated uncertain frame at 15.24 s (`0.4201`) and the explicit sampled-frame-only disclaimer.
6. **Hospitals & Dispatch:** choose a simulated location in the sidebar. Tab 5 shows source-labelled OSM retrieval; Tab 6 shows the clearly labelled simulated ambulance replay with no live GPS, traffic or real contact.
7. **Switch inputs repeatedly:** upload A → B → A. The SHA-256 and all tab content follow the current upload; changing the location also invalidates the cached event coordinates.

---

## 10. Known limitations

- The accident classifier was trained on forward-facing dashcam clips. Still images and CCTV angles are out of domain; its score is an uncalibrated ranking signal, not a calibrated probability. A low score is not proof of no accident, and a high frame is not proof of an accident.
- Video analysis uses evenly spaced samples, never every frame; OCR runs only on the highest-scoring sampled frame.
- YuNet was trained on Chinese plates and may miss Indian or blurred plates; OCR below 0.70 is never promoted to a verified registration number.
- Caller transcripts, demo locations and ambulances are simulated. Dispatch is a replay of stored results: no live GPS, traffic, bed availability, hospital capability verification or ambulance contact exists.
- OSM records may omit phone numbers or emergency tags. Missing values stay `not determined`; proximity or ranking score is not clinical verification; 24/7 operation is not independently verified.
- Engineered collision geometry is explanatory only and never overrides the trained model verdict.
- No root software licence has been selected by the project owner; the vendored YuNet model keeps its own Apache-2.0 licence.

**Final conclusion:** all six technologies, all six Streamlit tabs, all four mandatory inputs, stale-cache invalidation, dispatch simulation, and incident-report provenance were exercised with real code and real data. The repository is demonstration ready and published at <https://github.com/sri872/third-eye-ai>.
