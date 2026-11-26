"""Integration tests for the deterministic CV/OCR/NLP/SOS workflow."""
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from core.cv.image_processing import load_image
from core.cv.vehicle_localization import localize_vehicle_candidates
from core.ocr.plate_reader import ClassicalPlateReader
from core.nlp.incident_analysis import analyze_incident
from core.nlp.report_generation import generate_structured_report
from core.sos.location import SOURCE_SIMULATED_DEMO, create_sos_location
from core.hospitals.ranker import HospitalRanker
from core.accident.evidence_assessment import assess_accident_evidence


def test_component_imports():
    from app.components import cv_ui, ocr_ui, nlp_ui, sos_ui, routing_ui, report_ui
    from app.components.sos_gps_component import capture_device_location

    assert all(
        module is not None
        for module in (cv_ui, ocr_ui, nlp_ui, sos_ui, routing_ui, report_ui)
    )
    assert callable(capture_device_location)


def test_browser_gps_component_returns_coordinates_to_streamlit():
    component_file = (
        Path(__file__).resolve().parents[1]
        / "app" / "components" / "sos_gps_frontend" / "index.html"
    )
    source = component_file.read_text(encoding="utf-8")
    assert "navigator.geolocation.getCurrentPosition" in source
    assert "window.Streamlit.setComponentValue" in source
    assert "accuracy_meters" in source
    assert "position.timestamp" in source


def test_app_keeps_sos_and_report_tabs_available_without_an_image():
    app_file = Path(__file__).resolve().parents[1] / "app" / "main.py"
    app = AppTest.from_file(str(app_file), default_timeout=90).run()
    assert not app.exception, [item.value for item in app.exception]
    image_source = next(item for item in app.radio if item.label == "Image source")
    image_source.set_value("Upload custom image").run()
    assert not app.exception, [item.value for item in app.exception]
    assert len(app.tabs) == 5


def test_end_to_end_pipeline_integration():
    image = np.full((300, 400, 3), 128, dtype=np.uint8)
    bgr, _rgb = load_image(image)
    candidates = localize_vehicle_candidates(bgr)
    ocr_result = ClassicalPlateReader().read_plate(bgr[:50, :150])
    nlp_result = analyze_incident(
        "Two cars had an accident on Banga Road with fuel leak."
    )
    assessment = assess_accident_evidence(
        cv_candidates=candidates,
        cv_diagnostics={"width": 400, "height": 300},
        nlp_analysis=nlp_result,
    )

    sos_location, message = create_sos_location(
        31.2185,
        75.7852,
        source=SOURCE_SIMULATED_DEMO,
        location_name="Banga Road demo point",
    )
    assert sos_location is not None, message
    routing_data = HospitalRanker().rank_hospitals(
        sos_location,
        urgency_level=nlp_result["urgency_level"],
        incident_text=nlp_result["entities"]["normalized_text"],
    )

    report = generate_structured_report(
        cv_diagnostics={"width": 400, "height": 300},
        cv_candidates=candidates,
        plate_ocr_result=ocr_result,
        nlp_analysis=nlp_result,
        sos_location=sos_location,
        hospital_routing_result=routing_data,
        accident_assessment=assessment,
    )
    assert "markdown_report" in report
    for marker in (
        "[REAL CV]", "[CLASSICAL OCR]", "[CALLER REPORTED]",
        "[SOS GPS]", "[HOSPITAL ROUTE]", "[COMPUTED COMPARISON]",
    ):
        assert marker in report["markdown_report"]
