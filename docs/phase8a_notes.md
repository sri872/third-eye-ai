# Phase 8a-2 Notes — Manual-Upload Streamlit Dashboard

## Scope and file ownership

Created/edited only:

- `app/main.py` — single-page manual-upload dashboard (replaces the earlier automatic-sample Phase 8a UI).
- `app/video_utils.py` — real-video frame sampling helper (frame metadata + 12 evenly spaced frames).
- `tests/test_app_ui.py` — AppTest coverage, no-upload guard, widget-policy guard.
- `tests/test_app_video.py` — frame-sampler tests on a real Nexar clip cut into a temp file.
- `docs/phase8a_notes.md` — this file.

No files under `core/`, `db/`, `config/`, `scripts/`, `requirements.txt`, `README.md`,
`launch.command`, or any other existing test were edited. No packages were installed.
No process was killed or restarted.

## Manual CV input — nothing automatic

- The only CV input is `st.file_uploader` in the sidebar, accepting
  `jpg, jpeg, png, mp4, mov, avi`. This is the sole permitted use of the uploader;
  every other interactive control is a `st.selectbox`.
- With no upload, the app shows a short instruction and stops before any model
  loader, detector, classifier, OCR reader, or FAISS index is touched. The new test
  `test_no_upload_runs_no_model` monkeypatches those entry points to raise if called.
- `data/samples/` is never read. Sample call **names** are listed to populate the
  call dropdown, but call text is read only after the user selects that call.
- The call dropdown defaults to `No call (vision only)` and offers the 8 real files in
  `data/sample_calls/call_*.txt`, each labelled `SIMULATED SAMPLE CALL`. A selected
  call is paired arbitrarily with the upload and labelled `ARBITRARY DEMO PAIRING`.
- The incident-location dropdown continues to use the four `SIMULATED` entries from
  `config/demo_incident_locations.json`; it affects hospitals only.

## Uploaded data never written to the project or DB

- Images are decoded from uploaded bytes with `PIL.Image.open(io.BytesIO(...))`.
- Videos are written only to a `tempfile.TemporaryDirectory(prefix="third_eye_upload_")`;
  the directory is removed when processing returns or raises. No upload is written
  under `data/` or anywhere else in the repository.
- Processed results are held in `st.session_state` keyed by filename + byte length +
  SHA-256, so changing a dropdown does not reprocess the media. Clearing the upload
  removes the cached result. This is in memory only.
- SQLite is opened only as `sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)`
  with `SELECT` statements. There is no `init_db()`, no `save_*`, no writable
  connection, and no commit anywhere in `app/`.

## In-memory core functions used

`core.cv.vehicle_report.build_vehicle_reports` is deliberately **not** called: it
always creates annotated crop files on disk. The same Phase 2 information is produced
in memory from these in-memory core functions:

- `core.cv.detector.detect` — accepts a PIL image, returns REAL `Detection` objects.
- `core.cv.vehicle_attributes.crop_vehicle` / `dominant_color` / `get_vehicle_type`.
- `core.ocr.plate_reader.read_plate` — accepts an in-memory crop and returns plate text
  plus its real OCR confidence, or `(None, None)`.
- `core.nlp.occupant_report.build_occupant_reports` — in-memory `OccupantReport` list.
- `core.ml.accident_analyzer.classify_frame_probability` / `determine_verdict` /
  `compute_persons_near_vehicles` — no file path and no DB save.
- `core.ml.accident_features.extract_accident_features` — explanation text only.
- `core.nlp.incident_text.generate_incident_summary` — vision-only report text.
- `core.nlp.fusion.fuse_incident_evidence(..., save_to_db=False)` — simulated-call
  report, returned in memory without writing to SQLite.
- `core.rag.hospital_data.fetch_or_load_hospitals(save_to_db=False)` and the FAISS
  retriever — hospital suggestions computed in memory.

`AccidentEvent` objects are constructed in the app with REAL vision results. Because
the schema requires latitude/longitude, the coordinates of the currently selected
**SIMULATED** demo location are stored there solely for the downstream hospital
lookup; the UI labels those coordinates as SIMULATED and never presents them as
location inferred from the upload.

## Image behaviour

Run on the uploaded image only: detector boxes, vehicle type/color/plate OCR,
accident score and verdict, occupant reports, and incident text. The Scene tab shows
in-memory PIL-drawn boxes (no annotated file is saved). Plates below 0.7 OCR
confidence are shown as `unverified`; missing plates are `not readable`.

## Video behaviour

- `app/video_utils.py` uses OpenCV to read real metadata (frame count, fps, width,
  height, duration) and to sample **12** frames evenly across the entire clip,
  including the first and last frames. Seeking is used first, with a sequential
  decode fallback for containers that cannot seek.
- Every sampled frame gets the trained classifier score/verdict and detector-based
  vehicle/person counts. The Accident check tab shows a table and a line chart of
  score per frame, plus highest-scoring frame with its timestamp, the number of
  sampled frames at or above 0.60, and the exact statement:
  `This is a prototype frame classifier trained on dashcam clips (held-out ROC-AUC 0.7452); a single high frame is not proof of an accident.`
- Vehicle attributes and plate OCR run **only** on the highest-scoring frame; the UI
  states this explicitly. Other frames use detector counts only.
- `st.progress` advances during sampling and per-frame classification. For videos
  longer than 3 minutes, the app still samples only 12 frames and shows an explicit
  work-cap note; there is no full-clip OCR or full-clip forensic report.
- Unreadable uploads are caught and shown with `st.error` with a clear message;
  the app does not crash.

## Models, caching, and devices

- `st.cache_resource`: detector, ResNet-18 classifier, EasyOCR reader, and FAISS
  hospital retriever/index.
- `st.cache_data`: call names/text, demo locations, held-out metrics, read-only DB
  table/row results, and Phase 6 output discovery. Uploaded media results are kept in
  `st.session_state` instead, because the uploaded bytes are transient.
- `st.spinner` wraps image processing, video processing, and hospital retrieval.
- The device note is rendered only after a real upload. On this machine the detector
  and accident extractor use `MPS`, EasyOCR uses `CPU`, and sentence-transformers
  reports `cpu` for the embedding index.

## Honesty, badges, and limitations

Every displayed value carries a `REAL` or `SIMULATED` badge:

- REAL: uploaded-media vision output, OCR confidence, held-out metrics, OSM facility
  fields, verified emergency numbers, video metadata.
- SIMULATED: sample call, pairing with a sample call, demo location, and distances
  computed from that location.
- Unknown values remain `not determined`.

Model-card limitations, the occupant-count limitation, straight-line-only distance
wording, `not listed` phones, `not determined` beds, and `© OpenStreetMap contributors`
attribution are retained. The classifier score is labelled `uncalibrated model score`.
Vision-only reports show no disagreement flags.

## Dispatch tab (unchanged, still Phase 6-agnostic)

Read-only checks only:

1. `sqlite_master` table names containing `route` or `dispatch` (currently only
   `route_results`, with 0 rows).
2. `data/outputs/routes/*.png` (directory absent at the time of writing).

When neither source has data, the tab renders exactly **"Dispatch results not
available yet"** plus the checked paths. No Phase 6 schema, filename, or column
meaning is guessed; no change was made for the concurrent Phase 6 agent's work.

## Tests added/updated

- `tests/test_app_ui.py`: widget policy (uploader allowed only once in `main.py`,
  all other active widgets forbidden), no-upload model guard, real image upload via
  `AppTest`, and Dispatch fallback.
- `tests/test_app_video.py`: cuts a 1–2 second clip from a real Nexar MP4 into a
  temp directory with OpenCV, then verifies real metadata and 12 evenly spaced real
  frames spanning the whole clip; skips with a clear message if the dataset is absent.
  No synthetic video is generated.

Only `pytest tests/test_app_*.py` was run.
