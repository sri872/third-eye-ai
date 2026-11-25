# System Card — Project Third-Eye (`third-eye-v2`)

**Date**: 2026-11-25

**Scope**: Local prototype for manual-upload accident inspection, forensic-style vehicle/occupant reporting, hospital suggestion, and simulated emergency dispatch.

**Status**: Post-RAG-removal review build. Hospital suggestions use `core/hospitals/` deterministic ranking. This is a research/demo prototype — **not** a certified safety system and **not** connected to any emergency service.

---

## 1. What the system does

| Capability | Implementation | Input | Output |
| :--- | :--- | :--- | :--- |
| Vehicle / person detection | TorchVision Faster R-CNN ResNet-50 FPN, COCO-pretrained, MPS/CPU | One uploaded image or video frame | `Detection` records with real confidences |
| Vehicle attributes | OpenCV crop + k-means dominant colour; class-to-type map; EasyOCR plate reader | Detection crops | Colour, share, type, plate text + real OCR confidence |
| Accident signal | Frozen ResNet-18 embeddings + logistic regression (`models/accident_classifier.joblib`) | Same frame | Probability + verdict (`LIKELY ACCIDENT` ≥ 0.60, `UNCERTAIN` 0.40–0.60, `NO ACCIDENT SIGNAL` < 0.40) |
| Geometric explanation | IoU / aspect-ratio features and transparent rules (`method="RULE_BASED_HEURISTIC"`) | Detections | Explanation text only; never overrides the trained model score |
| Caller NLP / fusion | spaCy `en_core_web_sm`, rule-based extraction | Simulated call text + vision result | Occupant report, fused incident summary |
| Hospital suggestion | Deterministic OSM facility classification and source-tag/category/Haversine ranking | Incident location + caller-reported needs | Ranked real OSM facilities with transparent heuristic score; no semantic retrieval |
| Dispatch & routing | Offline OSM graph + admissible A* + deterministic trajectory simulation | 4 simulated incidents, 6 simulated ambulances | `data/outputs/dispatch_summary.json`, route PNG/GeoJSON, SQLite `route_results` |
| UI | Streamlit single-page dashboard, manual upload only | User-selected file + dropdowns | Tabs for `1. Scene`, `2. Vehicles`, `3. Accident check`, `4. Incident report`, `5. Hospitals & emergency numbers`, `6. Dispatch` |

Every persisted record carries an explicit `data_source` of `REAL` or `SIMULATED`. Model confidences are real or `None`; rule scores are labelled as heuristics. No YOLO / Ultralytics component is used anywhere.

---

## 2. Accident classifier — measured, not assumed

All numbers in this section were read directly from `models/accident_classifier_metrics.json` (no values recalled from memory).

- Architecture: frozen `torchvision.models.resnet18` (`ResNet18_Weights.DEFAULT`, final layer `Identity`, 512-d embedding) + scikit-learn `LogisticRegression` with `best_C = 0.01`.
- Held-out split: **60 test videos / 180 test frames** (training: 280 videos / 840 frames; validation: 60 videos / 180 frames).
- Majority-class baseline accuracy: **0.5**.

| Metric | Held-out test value |
| :--- | :---: |
| Accuracy | 0.6778 |
| Precision | 0.66 |
| Recall | 0.7333 |
| F1 | 0.6947 |
| ROC-AUC | 0.7452 |

Confusion matrix (rows = true negative / true positive, columns = predicted negative / predicted positive): `[[56, 34], [24, 66]]` → TN 56, FP 34, FN 24, TP 66.

**Reading the matrix honestly**: the model misses 24 of 90 positive frames (26.7% false-negative rate) and falsely alarms on 34 of 90 negative frames (37.8% false-positive rate). ROC-AUC 0.7452 is above the 0.70 project threshold but the probability scores are not calibrated for clinical or legal use. Accuracy 0.6778 is only 17.8 percentage points above the majority baseline.

---

## 3. Data provenance — REAL vs SIMULATED

- **REAL**: cached OpenStreetMap road graph (3,247 nodes, 8,582 edges, ~1,466 km, `© OpenStreetMap contributors`); 116 raw / 115 deduplicated OSM healthcare facilities in the corridor; detector/OCR/model outputs on uploaded media; all held-out video scores in `models/accident_classifier_metrics.json`.
- **SIMULATED**: incident coordinates (`config/demo_incident_locations.json`), 6 ambulance positions and speeds (`config/simulated_ambulances.json`), replay trajectories, and the choice of severity for the demo scenarios. Dispatch ETAs are computed on the real road graph with assumed speeds and **no live traffic**.
- **External**: Google Maps links in the Dispatch tab leave the app; they are not the project's own route.

### Hospital ranking architecture
Hospital suggestions use `core/hospitals/` with deterministic source-backed facility classification, source-reported emergency tags, facility category, and Haversine distance. Caller keywords are mapped to an explanatory assumed-needs string only; they are not embedded or semantically searched. FAISS, SentenceTransformers, hospital embedding generation, vector indexes, and vector retrieval are not part of the current application. The former `core/rag/` namespace has been removed; frozen ResNet-18 visual embeddings remain only for the accident classifier.

### OpenStreetMap coverage limits (from `docs/phase5_report.md`)

Of 116 facilities: 17 (14.66%) have a phone tag, 3 (2.59%) have `emergency=yes/no`, and 2 (1.72%) have `healthcare:speciality`. Trauma suitability in dispatch is therefore a **name-based rule**, not verified clinical data.

---

## 4. Known limitations (do not overclaim)

1. The classifier was trained on three evenly spaced frames from each Nexar clip (60/75/90% positions) at 180 frames total for testing; it is a forward-facing dashcam prototype, not a general accident detector. Single still photos from other viewpoints (e.g. roadside, elevated, or oblique angles) are out of domain. For example, `data/samples/road_1.jpg` is a documented miss: although it depicts a real two-car collision, the classifier outputs score `0.3179` (`NO ACCIDENT SIGNAL`), while the geometric overlap context detects `max vehicle IoU=0.12, heavily overlapping pairs=7`.
2. Two held-out *positive* clips shown in `docs/DEMO_SCRIPT.md` demonstrate the range: `00075.mp4` reaches 8/12 frames ≥ 0.60, while `01023.mp4` never reaches 0.60. A single low frame score does **not** rule out a crash.
3. Detection is COCO-pretrained, not fine-tuned on crash damage; vehicle counts in cluttered clips are approximate.
4. Only 1 of 6 vehicles in the demo image produced plate text, at OCR confidence 0.5412 — below the dashboard's 0.7 `unverified` threshold. Plate reading is diagnostic, not identification.
5. Ambulance fleet, ETAs and replay are simulated. There is no live GPS, no dispatcher integration, and no hospital confirmation.
6. This version has **no** FastAPI backend, **no** login/tenants, **no** SHA-256 evidence custody, **no** multi-object tracking, and **no** audio transcription. Do not present it as a deployed emergency platform.
7. There is no live-traffic feed; routing covers only the Phagwara corridor (75.69–75.80°E, 31.20–31.27°N).

---

## 5. Intended use

Demonstration, teaching and internal evaluation of a multimodal prototype on manually supplied media. Any real emergency must be handled by calling local emergency services; this system neither detects nor dispatches in real time.
