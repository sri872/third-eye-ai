# THIRD EYE AI — CV, Indian Number-Plate OCR, NLP & SOS Hospital Routing

THIRD EYE AI is a local Streamlit demonstration that combines three deterministic AI-focused disciplines—classical Computer Vision, template-based number-plate OCR, and rule-based NLP—with conventional SOS location validation, hospital catalog filtering, and road-network routing.

It does not use Machine Learning, Deep Learning, RAG, pretrained neural models, neural OCR, or LLMs.

> **Emergency safety notice:** This is a demonstration and decision-support prototype. It does not place emergency calls, contact or assign an ambulance, confirm hospital acceptance, verify staffing or bed availability, or use live traffic. Human confirmation is mandatory.

## Core workflow

1. **Computer Vision:** Enhance an uploaded road image, calculate image diagnostics, and propose geometric vehicle regions using OpenCV contours and morphology. These are candidate regions, not learned vehicle detections or proof of a collision.
2. **OCR:** Localize plate-like regions, segment glyphs, compare them with the bundled A–Z/0–9 templates, and validate Indian registration formats. Raw evidence and uncertainty are retained; missing characters are never guessed.
3. **NLP:** Extract caller-reported vehicles, locations, injuries and hazards using Python rules, then generate a transparent urgency description and contradiction checks.
4. **Accident evidence assessment:** A deterministic heuristic combines available candidate geometry and caller text. It is not a medical/legal determination and must not be treated as proof of a collision.
5. **SOS location:** Accept an imported SOS JSON event, explicitly permitted browser GPS when the device is at the scene, an operator-selected map pin, or clearly marked simulated Phagwara demo coordinates.
6. **Hospital selection:** Filter a dated OpenStreetMap snapshot to source-tagged hospitals; do not suggest clinics as hospitals or infer emergency capabilities from names.
7. **Road routing:** Calculate a directed path over the saved Phagwara OpenStreetMap GraphML road network. Routes outside graph coverage or without a connected path are reported unavailable; straight-line distance is never labelled as road distance.
8. **Combined report:** Preserve provenance tags for visual evidence, OCR, caller NLP, SOS source, hospital records, and route calculations.

## Technology boundaries

| Area | Implementation | Not used |
|---|---|---|
| CV | OpenCV, NumPy, contours, Canny, morphology, geometry | YOLO, Faster R-CNN, RCNN, ResNet, CNN, pretrained detectors |
| OCR | OpenCV preprocessing, connected components, character templates, regex validation | EasyOCR, Tesseract, neural OCR |
| NLP | Python regular expressions, keyword dictionaries and deterministic rules | spaCy models, Transformers, LLMs, embeddings |
| SOS/hospitals/routes | Browser Geolocation API with explicit permission, SOS JSON import, map pin, OpenStreetMap data, NetworkX, Folium | ML/DL, live dispatch, live traffic, claimed hospital bed feeds |

## SOS location sources

The SOS interface offers four modes:

- **Incoming SOS event JSON import:** Upload or paste an event exported from an SOS provider. The application validates coordinates and requires an event timestamp to detect stale messages. The file/text import is not cryptographically authenticated by this app and is not a real-time webhook. Verify the sender separately. A production integration needs an authenticated provider API/webhook adapter.
- **Browser GPS:** Click the GPS button and grant browser permission. Use this only if the device is physically at the incident. Captured coordinates, accuracy and timestamp are returned to Streamlit.
- **Operator-selected map pin:** Click the interactive map where the accident occurred. This is an operator-selected location, not a GPS reading; confirm it with the caller. Accuracy remains unknown.
- **Phagwara demo scenario:** Preset coordinates are labelled SIMULATED_DEMO; no false GPS accuracy is attached.

Locations with invalid coordinates or timestamps older than 15 minutes are blocked from routing. Browser GPS can require HTTPS or localhost and user permission. The dispatcher’s computer location is not substituted automatically for the incident location.

## Hospital source and route logic

The bundled hospital snapshot is an OpenStreetMap extract dated 2026-11-24. It may be incomplete or outdated. The app includes only named facilities tagged as hospitals (amenity=hospital or healthcare=hospital) and filters clinics and other outpatient services.

Ranking is transparent:
1. Down-rank an apparent single-specialty facility when the caller description does not indicate a matching need. Name hints do not prove clinical capability.
2. Use an explicit OpenStreetMap emergency=yes tag only as a secondary source-data indicator; this does not prove hours, staffing, 24/7 service or current readiness.
3. Order remaining hospitals by estimated travel time on the directed saved road graph.

The route follows the network’s static edge travel-time attributes. The short connectors from the exact SOS pin to the snapped road node and from the destination node to the hospital are straight-line approximations with a stated connector-speed assumption. There is no live traffic or ambulance ETA. If no path exists or coordinates are outside the saved network, no route is presented.

An external Google Maps driving-directions link is provided for navigation recalculation by that service.

## Streamlit interface

The five tabs are:
1. Computer Vision
2. Plate OCR
3. NLP & Evidence
4. SOS & Hospital Route
5. Combined Report

SOS routing remains usable even when no image is uploaded. CV/OCR tabs show an upload prompt instead of blocking the entire app.

## Installation

Python 3.10+ is recommended.

    git clone https://github.com/sri872/third-eye-ai.git
    cd third-eye-ai
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Launch

    streamlit run app/main.py

Open http://localhost:8501.

## Test

    pytest -v
    python -m compileall -q app core tests
    git diff --check

## Limitations and accuracy

Classical contour proposals are not reliable object detectors across arbitrary scenes. Template OCR is sensitive to fonts, blur, perspective, dirt, glare and occlusion. Template self-matching and synthetic examples are not real-world Indian plate accuracy. NLP rules only recognize terms and patterns covered by their dictionaries. The hospital database is a dated community-maintained map extract, not a verified clinical directory.

No accuracy target is guaranteed by design. Reported metrics must be calculated using a labelled held-out dataset and must distinguish synthetic/template checks from real-world tests.

## Data attribution

Hospital locations and the saved road graph are derived from OpenStreetMap. © OpenStreetMap contributors; Open Database License (ODbL) 1.0. The cached hospital snapshot is dated 2026-11-24. Static graph speeds do not represent current traffic.
