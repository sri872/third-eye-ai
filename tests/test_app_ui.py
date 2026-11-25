"""Phase 8a-2 UI tests: real AppTest runs plus a widget-policy guard.

Only tests/test_app_*.py is run for this phase. No synthetic media is used: the
image-upload test uploads bytes from a real file under data/samples.
"""

from __future__ import annotations

import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.ocr.plate_reader import PlateOCREvidence

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_FILE = REPO_ROOT / "app" / "main.py"
APP_DIR = REPO_ROOT / "app"
REAL_IMAGE = REPO_ROOT / "data" / "samples" / "road_1.jpg"
OTHER_IMAGE = REPO_ROOT / "data" / "samples" / "road_3.jpg"

FORBIDDEN_WIDGETS = (
    "st.button",
    "st.text_input",
    "st.text_area",
    "st.chat_input",
    "st.form_submit_button",
    "st.download_button",
    "st.link_button",
)


def _run_app() -> AppTest:
    return AppTest.from_file(str(APP_FILE), default_timeout=300).run()


def test_widget_policy_allows_only_the_cv_uploader() -> None:
    """Uploader is allowed only for CV input; every other active widget is banned."""
    offenses: list[str] = []
    uploaders: list[str] = []
    for path in sorted(APP_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(REPO_ROOT))
        for widget in FORBIDDEN_WIDGETS:
            name = widget.split(".", 1)[1]
            pattern = rf"\bst(?:\.sidebar)?\.{re.escape(name)}\s*\("
            if re.search(pattern, source):
                offenses.append(f"{relative}: {widget}")
        count = len(re.findall(r"\bst(?:\.sidebar)?\.file_uploader\s*\(", source))
        if count:
            uploaders.append(f"{relative}: {count}")

    assert not offenses, "Forbidden widgets found: " + ", ".join(offenses)
    assert uploaders == ["app/main.py: 1"], (
        "st.file_uploader must appear exactly once and only in app/main.py: " + str(uploaders)
    )


def test_no_upload_runs_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no upload, every model/inference entry point must stay untouched."""
    import core.cv.detector as detector
    import core.cv.vehicle_attributes as vehicle_attributes
    import core.ml.accident_analyzer as analyzer
    import core.nlp.fusion as fusion
    import core.nlp.incident_text as incident_text
    import core.nlp.occupant_report as occupant_report
    import core.ocr.plate_reader as plate_reader
    import core.rag.hospital_data as hospital_data
    import core.rag.index as rag_index

    forbidden_calls = (
        (detector, "detect"),
        (detector, "get_target_device"),
        (detector, "get_detector_model"),
        (analyzer, "classify_frame_probability"),
        (analyzer, "load_model_and_extractor"),
        (analyzer, "get_device"),
        (vehicle_attributes, "dominant_color"),
        (plate_reader, "read_plate"),
        (plate_reader, "read_plate_evidence"),
        (occupant_report, "build_occupant_reports"),
        (incident_text, "generate_incident_summary"),
        (fusion, "fuse_incident_evidence"),
        (hospital_data, "fetch_or_load_hospitals"),
        (rag_index, "HospitalIndex"),
        (rag_index, "get_embedding_model"),
    )

    def forbidden(name: str):
        def _raise(*args, **kwargs):
            raise AssertionError(f"Model/inference function called with no upload: {name}")

        return _raise

    for module, attribute in forbidden_calls:
        monkeypatch.setattr(module, attribute, forbidden(attribute), raising=True)

    at = _run_app()
    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]
    assert any("Upload one image or video in the sidebar" in item.value for item in at.info)
    assert [tab.label for tab in at.tabs] == [
        "1. Scene",
        "2. Vehicles",
        "3. Accident check",
        "4. Incident report",
        "5. Hospitals & emergency numbers",
        "6. Dispatch",
    ]
    assert len(at.sidebar.file_uploader) == 1
    assert [box.label for box in at.sidebar.number_input] == ["Video frames to sample"]
    assert [box.label for box in at.sidebar.selectbox] == [
        "Emergency call",
        "Incident location (for hospitals only)",
    ]
    assert at.sidebar.selectbox[0].options[0] == "No call (vision only)"
    assert all("SIMULATED SAMPLE CALL" in option for option in at.sidebar.selectbox[0].options[1:])


def test_real_image_upload_runs_and_shows_vision_only_report() -> None:
    """AppTest uploads a real sample image; no automatic sample processing occurs."""
    assert REAL_IMAGE.exists(), f"Real sample image missing: {REAL_IMAGE}"
    at = _run_app()
    at.sidebar.file_uploader[0].upload(
        REAL_IMAGE.name, REAL_IMAGE.read_bytes(), "image/jpeg"
    ).run()

    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]
    assert len(at.tabs) == 6

    scene_text = " ".join(node.value for node in at.tabs[0].get("markdown"))
    assert "uploaded image" in scene_text
    assert "ARBITRARY DEMO PAIRING" not in scene_text

    report_text = " ".join(
        node.value
        for kind in ("markdown", "caption", "info", "text")
        for node in at.tabs[3].get(kind)
    )
    assert "REAL VISION EVIDENCE" in report_text
    assert "SIMULATED CALLER EVIDENCE" in report_text
    assert "INCIDENT ASSESSMENT" in report_text
    report_table_values = [
        str(value)
        for node in at.tabs[3].table
        for value in node.value["Value"].tolist()
    ]
    assert "VISION ONLY" in report_table_values
    assert not at.tabs[3].get("code"), "The report must not be a monospaced code block"


def _vehicles_table(at: AppTest):
    """Return the single Vehicles-table DataFrame rendered by AppTest."""
    vehicles = next(tab for tab in at.tabs if tab.label.startswith("2."))
    assert len(vehicles.table) == 1, [node.value for node in vehicles.table]
    return vehicles.table[0].value


def _upload_real_image_and_get_vehicles(monkeypatch, evidence: PlateOCREvidence):
    """Upload the real sample with controlled detector/OCR evidence and read the Vehicles table."""
    import core.cv.plate_detector as plate_detector
    import core.ocr.plate_reader as plate_reader

    from core.cv.plate_detector import PlateRegion

    region = PlateRegion(
        bbox_local=(0, 0, 120, 48),
        bbox_original=(10, 20, 130, 68),
        heuristic_score=0.75,
        character_candidates=8,
        aspect_ratio=2.5,
        edge_density=0.2,
    )
    monkeypatch.setattr(
        plate_detector, "detect_plate_regions", lambda crop, bbox=None, **kwargs: (region,)
    )
    monkeypatch.setattr(plate_reader, "read_plate_evidence", lambda crop: evidence)
    at = _run_app()
    at.sidebar.file_uploader[0].upload(
        REAL_IMAGE.name, REAL_IMAGE.read_bytes(), "image/jpeg"
    ).run()
    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]
    return _vehicles_table(at)


def test_vehicles_tab_shows_readable_verified_plate(monkeypatch) -> None:
    """A real OCR result above 0.7 is shown verbatim, with its confidence and verified status."""
    evidence = PlateOCREvidence(
        plate="KA01AB1234",
        plate_confidence=0.94,
        raw_text="KA01AB1234",
        raw_confidence=0.94,
        candidate_count=1,
    )
    table = _upload_real_image_and_get_vehicles(monkeypatch, evidence)
    assert table["Plate"].iloc[0] == "KA01AB1234"
    assert table["OCR confidence"].iloc[0] == "0.94"
    assert table["Raw OCR text (unverified)"].iloc[0] == "KA01AB1234 (0.94)"
    assert table["Plate status"].iloc[0] == "verified"
    assert "(unverified)" not in table["Plate"].iloc[0]
    assert table["Plate detection status"].iloc[0] == "PLATE REGION DETECTED — READABLE"
    assert table["Plate bbox (source px)"].iloc[0] == "10,20,130,68"


def test_vehicles_tab_shows_low_confidence_plate_as_not_readable(monkeypatch) -> None:
    """Below 0.7 the Plate column stays NOT READABLE; raw evidence is retained."""
    evidence = PlateOCREvidence(
        plate="CAN09G",
        plate_confidence=0.55,
        raw_text="can09g",
        raw_confidence=0.55,
        candidate_count=1,
    )
    table = _upload_real_image_and_get_vehicles(monkeypatch, evidence)
    assert table["Plate"].iloc[0] == "NOT READABLE (UNVERIFIED)"
    assert table["Plate status"].iloc[0] == "NOT READABLE (UNVERIFIED)"
    assert table["Plate detection status"].iloc[0] == "PLATE REGION DETECTED — UNREADABLE (LOW OCR CONFIDENCE)"
    assert table["OCR confidence"].iloc[0] == "0.55"
    assert table["Unverified OCR candidate"].iloc[0] == "CAN09G (0.55)"
    assert table["Raw OCR text (unverified)"].iloc[0] == "can09g (0.55)"


def test_vehicles_tab_shows_raw_text_without_inventing_a_plate(monkeypatch) -> None:
    """Raw OCR text can be visible even when normalization finds no valid plate."""
    evidence = PlateOCREvidence(
        raw_text="car0.99",
        raw_confidence=0.70,
        candidate_count=1,
    )
    table = _upload_real_image_and_get_vehicles(monkeypatch, evidence)
    assert table["Plate"].iloc[0] == "NOT READABLE (UNVERIFIED)"
    assert table["Plate status"].iloc[0] == "NOT READABLE (UNVERIFIED)"
    assert table["OCR confidence"].iloc[0] == "not determined"
    assert table["Raw OCR text (unverified)"].iloc[0] == "car0.99 (0.70)"


def test_vehicles_tab_shows_unreadable_plate_when_ocr_has_no_result(monkeypatch) -> None:
    """No OCR result means exact 'NOT READABLE (UNVERIFIED)' and no invented confidence."""
    table = _upload_real_image_and_get_vehicles(monkeypatch, PlateOCREvidence())
    assert table["Plate"].iloc[0] == "NOT READABLE (UNVERIFIED)"
    assert table["Plate status"].iloc[0] == "NOT READABLE (UNVERIFIED)"
    assert table["OCR confidence"].iloc[0] == "not determined"
    assert table["Raw OCR text (unverified)"].iloc[0] == "not determined"


def test_vehicles_tab_distinguishes_no_plate_region(monkeypatch) -> None:
    """No candidate region is distinct from a detected but unreadable plate."""
    import core.cv.plate_detector as plate_detector

    monkeypatch.setattr(
        plate_detector, "detect_plate_regions", lambda crop, bbox=None, **kwargs: ()
    )
    at = _run_app()
    at.sidebar.file_uploader[0].upload(
        REAL_IMAGE.name, REAL_IMAGE.read_bytes(), "image/jpeg"
    ).run()
    assert not at.exception, [item.value for item in at.exception]
    table = _vehicles_table(at)
    assert table["Plate detection status"].iloc[0] == "NO PLATE REGION DETECTED"
    assert table["Plate bbox (source px)"].iloc[0] == "not determined"
    assert table["OCR status"].iloc[0] == "NOT RUN (NO PLATE REGION)"
    assert table["Plate"].iloc[0] == "NOT READABLE (UNVERIFIED)"


def test_vehicles_tab_distinguishes_ocr_initialization_failure(monkeypatch) -> None:
    """A detected region with failed OCR is not reported as an unreadable plate read."""
    table = _upload_real_image_and_get_vehicles(
        monkeypatch, PlateOCREvidence(ocr_status="OCR INITIALIZATION FAILURE")
    )
    assert table["Plate detection status"].iloc[0] == "OCR INITIALIZATION FAILURE"
    assert table["OCR status"].iloc[0] == "OCR INITIALIZATION FAILURE"
    assert table["Plate"].iloc[0] == "NOT READABLE (UNVERIFIED)"


def test_accident_check_still_image_warning() -> None:
    """The still-photo caveat appears for a real image upload, never with no upload."""
    expected = (
        "The accident classifier was trained on dashcam video frames (held-out "
        "ROC-AUC 0.7452 on 60 videos). This uploaded still image is outside that "
        "training domain: the score below is an uncalibrated ranking signal, not "
        "proof of an accident or of normal traffic. Human review is required."
    )

    empty = _run_app()
    assert not empty.exception, [item.value for item in empty.exception]
    empty_accident = next(tab for tab in empty.tabs if tab.label.startswith("3."))
    assert not empty_accident.warning, [item.value for item in empty_accident.warning]

    assert REAL_IMAGE.exists(), f"Real sample image missing: {REAL_IMAGE}"
    at = _run_app()
    at.sidebar.file_uploader[0].upload(
        REAL_IMAGE.name, REAL_IMAGE.read_bytes(), "image/jpeg"
    ).run()

    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]
    accident = next(tab for tab in at.tabs if tab.label.startswith("3."))
    warnings = [item.value for item in accident.warning]
    assert any(expected in value for value in warnings), warnings

    accident_text = " ".join(
        node.value for kind in ("markdown", "caption", "text") for node in accident.get(kind)
    )
    assert "Model verdict" in accident_text
    assert "Uncalibrated model score" in accident_text
    assert "NOT DETERMINED (out-of-domain still image)" in accident_text


def test_uploaded_media_fingerprint_is_shared_across_tabs_and_invalidates_cache() -> None:
    """All tabs describe the current bytes; changing bytes under one name forces reprocessing."""
    assert REAL_IMAGE.exists() and OTHER_IMAGE.exists()
    at = _run_app()
    at.sidebar.file_uploader[0].upload(
        REAL_IMAGE.name, REAL_IMAGE.read_bytes(), "image/jpeg"
    ).run()
    assert not at.exception, [item.value for item in at.exception]
    first_scene = " ".join(node.value for node in at.tabs[0].get("markdown"))
    first_vehicles = " ".join(node.value for node in at.tabs[1].get("markdown"))
    assert "Media fingerprint (SHA-256)" in first_scene
    assert "Media fingerprint (SHA-256)" in first_vehicles

    # Same filename, different bytes: SHA check must invalidate the cached result.
    at.sidebar.file_uploader[0].upload(
        REAL_IMAGE.name, OTHER_IMAGE.read_bytes(), "image/jpeg"
    ).run()
    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]
    second_scene = " ".join(node.value for node in at.tabs[0].get("markdown"))
    assert "Media fingerprint (SHA-256)" in second_scene
    assert _sha_from_text(first_scene) != _sha_from_text(second_scene)


def _sha_from_text(text: str) -> str:
    """Extract the displayed SHA-256 fingerprint from rendered markdown."""
    match = re.search(r"Media fingerprint \(SHA-256\):\*\* ([0-9a-f]{64})", text)
    assert match, text
    return match.group(1)


def test_real_demo_plate_stays_not_readable_below_threshold() -> None:
    """Regression: real demo OCR below 0.70 never becomes a verified plate."""
    demo_plate = REPO_ROOT / "data" / "samples" / "demo" / "single_car_plate.jpg"
    assert demo_plate.is_file()
    at = _run_app()
    at.sidebar.file_uploader[0].upload(
        demo_plate.name, demo_plate.read_bytes(), "image/jpeg"
    ).run()
    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]
    table = _vehicles_table(at)
    assert table["Plate"].iloc[0] == "NOT READABLE (UNVERIFIED)"
    assert table["Plate status"].iloc[0] == "NOT READABLE (UNVERIFIED)"
    conf = float(table["OCR confidence"].iloc[0])
    assert conf < 0.7
    # Raw evidence is retained verbatim even though no plate is verified.
    assert table["Raw OCR text (unverified)"].iloc[0] != "not determined"


def test_changing_demo_location_invalidates_cached_vision_result() -> None:
    """Regression: cached event coordinates must follow the selected location."""
    at = _run_app()
    at.sidebar.file_uploader[0].upload(
        REAL_IMAGE.name, REAL_IMAGE.read_bytes(), "image/jpeg"
    ).run()
    assert not at.exception, [item.value for item in at.exception]
    first_key = at.session_state["uploaded_result_key"]
    first_lat = at.session_state["uploaded_result"]["accident_event"].latitude

    at.sidebar.selectbox[1].set_value(
        "Banga Road Commercial Intersection | SIMULATED LOCATION"
    ).run()
    assert not at.exception, [item.value for item in at.exception]
    assert at.session_state["uploaded_result_key"] != first_key
    assert at.session_state["uploaded_result"]["accident_event"].latitude != first_lat


def test_dispatch_tab_shows_not_available_with_injected_missing_summary(
    monkeypatch, tmp_path: Path
) -> None:
    """The missing-summary path is injected; this test never skips."""
    from app.dispatch_ui import DISPATCH_SUMMARY_ENV

    missing = tmp_path / "dispatch_summary_not_present.json"
    assert not missing.exists()
    monkeypatch.setenv(DISPATCH_SUMMARY_ENV, str(missing))

    at = _run_app()
    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]
    dispatch = next(tab for tab in at.tabs if tab.label.startswith("6."))
    rendered = [
        str(getattr(node, "value", ""))
        for kind in ("info", "markdown", "text", "warning")
        for node in dispatch.get(kind)
    ]
    assert any("Dispatch results not available yet" in text for text in rendered), rendered

