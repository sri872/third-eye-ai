# Third Eye AI 🚑

**Local multimodal accident-analysis and simulated ambulance-dispatch prototype**

Third Eye AI analyses an uploaded road image or traffic video, detects vehicles and people, attempts number-plate OCR, runs a trained accident classifier, extracts structured facts from a caller transcript, retrieves source-labelled hospital records, and displays a simulated dispatch plan. It is designed to run entirely on a MacBook Air M2 using local model weights and cached OpenStreetMap data — no paid API, no cloud inference, no Docker.

> **Prototype, not an emergency service.** The app does not place calls, contact ambulances, verify GPS or live traffic, or confirm an accident from a single image. Every displayed value is labelled `REAL`, `SIMULATED`, or `NOT DETERMINED`. See [FINAL_DEMO_VERIFICATION_REPORT.md](FINAL_DEMO_VERIFICATION_REPORT.md) for the actual verified outputs and limitations, and [GITHUB_FILE_AUDIT.md](GITHUB_FILE_AUDIT.md) for the complete file inventory.

---

## Five core technologies — where each one actually runs

| # | Technology | Implementation (real code) | What executes |
|---|---|---|---|
| 1 | Computer Vision | [`core/cv/detector.py`](core/cv/detector.py), [`core/cv/plate_detector.py`](core/cv/plate_detector.py), [`core/cv/plate_detector_yunet.py`](core/cv/plate_detector_yunet.py) | TorchVision Faster R-CNN ResNet-50 FPN (COCO) detects `person`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`; orientation retry for physically rotated photos; YuNet ONNX + OpenCV heuristic + OCR fallback propose plate regions. MPS → CPU fallback. |
| 2 | OCR | [`core/ocr/plate_reader.py`](core/ocr/plate_reader.py) | Local EasyOCR English reader reads detected plate crops. Raw text and confidence are retained verbatim; text normalisation never invents characters. Verification threshold is **0.70** in [`app/main.py`](app/main.py); below it the Plate column reads `NOT READABLE (UNVERIFIED)`. |
| 3 | Machine Learning | [`core/ml/accident_analyzer.py`](core/ml/accident_analyzer.py), [`core/ml/accident_features.py`](core/ml/accident_features.py), [`models/accident_classifier.joblib`](models/accident_classifier.joblib) | A trained logistic-regression classifier consumes frozen ResNet-18 embeddings and returns an **uncalibrated** score: `>=0.60` LIKELY ACCIDENT, `0.40–0.60` UNCERTAIN, `<0.40` NO ACCIDENT SIGNAL. Geometric overlap features are explanatory only and never decide the verdict. |
| 4 | Deep Learning | [`core/cv/detector.py`](core/cv/detector.py), [`core/ml/accident_analyzer.py`](core/ml/accident_analyzer.py), [`core/rag/index.py`](core/rag/index.py) | Neural-network inference: Faster R-CNN for detection and ResNet-18 feature extraction for the accident classifier. |
| 5 | NLP | [`core/nlp/caller_extraction.py`](core/nlp/caller_extraction.py), [`core/nlp/fusion.py`](core/nlp/fusion.py), [`core/nlp/incident_text.py`](core/nlp/incident_text.py), [`core/nlp/occupant_report.py`](core/nlp/occupant_report.py) | spaCy-based rule extraction of caller-reported vehicles, people, location, injuries, hazards, urgency and accident status. Fusion keeps REAL vision evidence separate from SIMULATED caller evidence and never lets the transcript override the visual verdict. |
| 6 | Geospatial intelligence (supporting module) | `core/rag/` | Cached OpenStreetMap facility records are classified using source tags and facility names, then ranked using deterministic emergency-tag, facility-category and Haversine-distance rules. No embeddings, vector database or semantic retrieval are used. |

Supporting modules: [`core/routing/`](core/routing/) contains the A* road-network planner, trajectory simulation and map rendering used by the offline dispatch scripts and the stored summary shown in the UI; [`core/schemas.py`](core/schemas.py) enforces explicit `REAL`/`SIMULATED` provenance; [`db/database.py`](db/database.py) holds the SQLite schema used by the CLI pipeline scripts. The Streamlit app opens SQLite read-only and writes no uploaded bytes to the project.

---

## Six Streamlit tabs

The entry point is [`app/main.py`](app/main.py). There are no action buttons and no required typing: upload one file and the pipeline runs; dropdowns select a simulated call, a demo location, the number of video frames to sample, and a dispatch replay step.

1. **Scene** — analysed image (or highest-scoring sampled video frame), REAL detector boxes, vehicle/person counts, SHA-256 fingerprint, orientation correction and occlusion diagnostics.
2. **Vehicles** — per-vehicle crop attributes, plate candidate bbox and detection method, raw OCR text/confidence, and verified/unverified plate status. No plate is invented below 0.70 OCR confidence.
3. **Accident check** — trained-model score and verdict with held-out metrics; for video, real container metadata, per-frame scores with frame numbers and timestamps, score chart, temporal signal and an explicitly sampled-frame-only assessment.
4. **Incident report** — source-separated tables and generated narrative combining REAL vision facts with optional SIMULATED caller facts; caller claims that conflict with the visual model are labelled `CALLER-REPORTED ACCIDENT — VISUAL MODEL DOES NOT CONFIRM`.
5. **Hospitals & emergency numbers** — SIMULATED incident location, caller-location consistency, REAL OSM facility suggestions with source provenance, and plain-text emergency numbers (the app never places calls).
6. **Dispatch** — replay of a stored SIMULATED dispatch summary: simulated ambulance placement/ETA, REAL OSM hospital record, route legs, static map fallback, Google Maps navigation links and explicit "no live GPS / no live traffic" notices.

---

## Architecture and workflow

```text
one upload (image or video)
   │
   ├─ CV ── Faster R-CNN ───────────────► REAL detections / boxes / counts
   │        └─ YuNet + OCR fallback ───► plate candidate bbox
   │                 └─ EasyOCR ───────► raw text + confidence ──► 0.70 gate ──► Vehicles tab
   │
   ├─ ML/DL ─ ResNet-18 + joblib ──────► uncalibrated score ──► verdict ──► Accident tab
   │
   ├─ NLP ─── caller transcript (SIMULATED, optional)
   │        └─ fusion ─────────────────► source-separated report ──► Incident report tab
   │
   ├─ Geospatial rules ─ cached OSM records ─► labelled hospital suggestions ──► Hospitals tab
   │
   └─ dispatch summary (SIMULATED) ─────► replay, map, ETAs ──► Dispatch tab
```

For video, frames are evenly sampled across the whole clip (configurable 4–60, default 12). Every sampled frame gets real detection and classifier inference; frame number and timestamp are preserved. OCR and vehicle attributes run only on the highest-scoring sampled frame. The UI never claims full-clip analysis.

---

## Installation

Python 3.12 and macOS are the tested environment. From a clone of this repository:

```bash
cd third-eye-v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is pinned and includes the `en_core_web_sm` spaCy model wheel. Hospital search uses deterministic spatial and attribute rules and requires no embedding or vector packages. No `.env` file or API key is required; `.env.example` documents the optional override variables.

### Bundled and cached model assets

| Asset | Location | Notes |
|---|---|---|
| EasyOCR CRAFT detector + English recogniser | `models/easyocr/*.pth` (~94 MB) | Bundled in the repository; `download_enabled=False`. |
| YuNet license-plate ONNX + vendored Apache-2.0 licence | `models/plate_detector/` (~4 MB) | SHA-256 verified at load time. |
| Trained accident classifier + metrics | `models/accident_classifier*.{joblib,json}` | Bundled. |
| Faster R-CNN ResNet-50 FPN COCO weights | TorchVision cache (`~/.cache/torch/hub/checkpoints`) | Downloaded once by TorchVision on first use (~160 MB); automatic MPS→CPU fallback. |
| ResNet-18 weights | TorchVision cache (~45 MB) | Downloaded once on first classification. |
| Hospital retrieval | Local code | Deterministic source-tag, category and distance ranking; no model download. |

See [`GITHUB_FILE_AUDIT.md`](GITHUB_FILE_AUDIT.md) for exact sizes, hashes and retrieval instructions for every asset.

---

## Launch

```bash
cd third-eye-v2
source .venv/bin/activate
streamlit run app/main.py
```

Open <http://localhost:8501>. On macOS you can also double-click [`launch.command`](launch.command), or run `bash launch.command`.

Headless verification:

```bash
streamlit run app/main.py --server.headless true --server.port 8501
```

---

## Professor demonstration (exact sequence)

Mandatory inputs are in [`data/samples/demo/`](data/samples/demo/). Upload them through the sidebar; the app deliberately does not auto-load samples.

1. **Single car:** upload `single_car_plate.jpg` → **Scene** shows a rotated-to-landscape image with one REAL car box → **Vehicles** shows the plate candidate region, raw EasyOCR evidence and `NOT READABLE (UNVERIFIED)` unless OCR confidence reaches 0.70 → **Incident report** shows vision-only source labels.
2. **Collision image:** upload `two_car_collision.jpg` → **Scene** shows REAL car/person boxes → select call `call_demo_01_collision.txt` → **Accident check** shows the actual (possibly non-confirming) model verdict → **Incident report** keeps caller urgency/hazards/injuries separate and reports any source disagreement without changing the visual verdict.
3. **Heavy traffic:** upload `heavy_traffic.jpg` → **Scene** shows real crowded-scene counts → **Accident check** shows the real classifier result; close vehicles and many people are not treated as a confirmed collision.
4. **Road video:** upload `road vdo.mp4` → **Accident check** shows real metadata, sampled frame numbers/timestamps/scores, temporal signal and the sampled-frame-only assessment.
5. Select a SIMULATED demo location in the sidebar → **Hospitals** shows REAL OSM suggestions with provenance → **Dispatch** shows the clearly labelled simulated replay. Switch inputs repeatedly: cached results are keyed by file bytes, sample count and location, so old results cannot leak.

The measured outputs for these exact files are recorded in [`FINAL_DEMO_VERIFICATION_REPORT.md`](FINAL_DEMO_VERIFICATION_REPORT.md) and machine-readable form in [`data/outputs/demo_verification.json`](data/outputs/demo_verification.json) and [`data/outputs/streamlit_workflow.json`](data/outputs/streamlit_workflow.json).

---

## Tests and verification commands

```bash
# Full suite (from the repository root)
.venv/bin/python -m pytest -q

# Real inference on all four mandatory files (detector + YuNet + OCR + classifier + video sampling)
.venv/bin/python scripts/verify_demo_assets.py

# Real Streamlit AppTest workflow across all six tabs and four files
.venv/bin/python scripts/verify_streamlit_workflow.py
```

The suite covers schemas, database constraints, real detector/OCR integration, plate localisation, caller extraction, source-separated fusion, deterministic hospital-ranking honesty, routing, all six UI tabs, video sampling and the demo-input regressions. Tests that need external dashcam data skip explicitly rather than inventing evidence.

---

## Known limitations

- The accident classifier was trained on forward-facing dashcam clips; still images, CCTV angles and post-impact wreckage are out of domain. Scores are **uncalibrated ranking signals**, not true probabilities. Held-out ROC-AUC is 0.7452; false positives and false negatives occur. A single high frame is not proof of an accident, and a low score is not proof that no accident happened.
- Engineered collision geometry is explanatory only; no rule-based heuristic is presented as a trained model prediction.
- YuNet was trained on Chinese plates and may miss Indian or heavily blurred plates. Low-confidence OCR candidates are never promoted to verified registration numbers.
- Caller transcripts, incident locations and ambulances are simulated. Dispatch replay uses stored results; no live GPS, traffic, bed availability, hospital capability verification, or real ambulance contact exists.
- Video analysis uses evenly spaced samples, never every frame, and OCR runs only on the peak-scoring sampled frame.
- The OSM cache may omit phone numbers or emergency tags; missing values stay `not determined`. Proximity or ranking score is not clinical verification.
- Further documentation: [DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md), [SYSTEM_CARD.md](docs/SYSTEM_CARD.md), [accident model card](docs/model_card_accident.md), and phase reports in [`docs/`](docs/) plus [`PHASE15_FINAL_VERIFICATION.md`](PHASE15_FINAL_VERIFICATION.md).

No licence has been selected by the project owner; no root `LICENSE` file is included. The vendored YuNet ONNX model retains its own Apache-2.0 licence in `models/plate_detector/`.
