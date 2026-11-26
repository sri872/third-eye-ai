# THIRD EYE AI — Architecture

## Purpose

A compact emergency incident analysis demonstration based on three deterministic AI disciplines—classical Computer Vision, classical template OCR, and rule-based NLP—supported by SOS location validation and conventional hospital road routing.

## Processing and evidence flow

    Optional incident image -> OpenCV image processing -> geometric candidate regions
                             -> plate localization -> glyph segmentation -> template OCR
    Caller incident text ----------------------------> rule-based NLP entities/urgency
    SOS event / browser GPS / operator map pin ------> validated incident location
    CV + OCR + NLP + SOS source details -------------> provenance-aware report
    Validated SOS location + OSM hospital snapshot --> source-based hospital filter/ranker
    SOS coordinates + hospital coordinates + OSM GraphML --> directed NetworkX road path

## Technology boundaries

- CV: OpenCV and NumPy only; geometric candidate boxes are not learned detections.
- OCR: classical image processing, connected-component segmentation and fixed glyph templates. No neural OCR engine.
- NLP: Python regex/keyword dictionaries. No trained language model.
- Hospital and routing logic: conventional Python, NetworkX and Folium. These are supporting software systems, not additional AI models.
- ML, DL, RAG, pretrained neural weights, PyTorch, TorchVision, EasyOCR, spaCy models and LLM integrations are excluded.

## SOS provenance

Supported location sources:
1. Authenticated provider adapter payload (GPS_SOS_PAYLOAD is reserved for a future trusted integration; none is connected by this prototype).
2. Imported SOS JSON file/text (SOS_JSON_IMPORT; sender is not authenticated by this application).
3. Explicit browser geolocation after the user clicks the capture button and grants permission (DEVICE_GEOLOCATION).
4. Operator-selected map pin (OPERATOR_MANUAL_SELECTION).
5. Demonstration-only coordinates (SIMULATED_DEMO).

Coordinates are range-checked; 0,0 is rejected. Accuracy and timestamp are retained only when supplied. Imported event payloads must include their event timestamp; locations older than 15 minutes are blocked from routing. The browser/device location is not silently used as the incident location unless the user explicitly selects it and confirms that the device is at the scene. A map pin is operator-selected and has no GPS accuracy claim.

## Hospital selection and routing

The dated hospital cache is from OpenStreetMap. Only named hospital-tagged facilities are considered; clinics and outpatient services are excluded. Specialty words and tags can trigger transparent down-ranking, but do not prove clinical capability. The emergency tag is displayed as source data only.

The routing engine loads a directed GraphML graph, finds the nearest node to the SOS origin and each hospital, and computes the minimum-travel-time path using travel_time edge weights. Parallel edges are compared by their travel-time value. Off-network connectors are estimated using a stated fixed connector speed. The system refuses to show a route if graph coverage or a directed path is missing; no Haversine road-distance fallback is presented.

Static route-time estimates are not live traffic or an ambulance ETA. Hospital acceptance, bed availability, emergency department staffing and current readiness remain unknown unless a trusted live integration supplies them.

## User interface

The five tabs are Computer Vision, Plate OCR, NLP & Evidence, SOS & Hospital Route, and Combined Report. SOS routing remains accessible even when the operator has not uploaded an image. Missing CV evidence remains NOT RUN rather than being fabricated.

## Safety

The application prepares information for human review. It does not contact 112/108, issue a real dispatch, reserve a bed, or assert that an ambulance is en route. Human verification is required before real emergency action.
