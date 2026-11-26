"""Tests for Application Modules and End-to-End Workflow.

Verifies Streamlit component imports, module interfaces,
and end-to-end integration without neural models.
"""

import os
import pytest
import numpy as np

from core.cv.image_processing import load_image
from core.cv.vehicle_localization import localize_vehicle_candidates
from core.ocr.plate_reader import ClassicalPlateReader
from core.nlp.incident_analysis import analyze_incident
from core.nlp.report_generation import generate_structured_report
from core.sos.location import create_sos_location, SOURCE_GPS_SOS_PAYLOAD
from core.hospitals.ranker import HospitalRanker
from core.accident.evidence_assessment import assess_accident_evidence


def test_component_imports():
    """Verify Streamlit UI components import cleanly."""
    from app.components import cv_ui
    from app.components import ocr_ui
    from app.components import nlp_ui
    from app.components import sos_ui
    from app.components import routing_ui
    from app.components import report_ui
    assert cv_ui is not None
    assert ocr_ui is not None
    assert nlp_ui is not None
    assert sos_ui is not None
    assert routing_ui is not None
    assert report_ui is not None


def test_end_to_end_pipeline_integration():
    """Test complete pipeline from image to OCR, NLP, SOS location, routing, and combined report."""
    # 1. Create a dummy image
    img = np.full((300, 400, 3), 128, dtype=np.uint8)
    bgr, rgb = load_image(img)

    # 2. Localize vehicle candidates
    candidates = localize_vehicle_candidates(bgr)

    # 3. Read plate
    reader = ClassicalPlateReader()
    ocr_res = reader.read_plate(bgr[:50, :150])

    # 4. Analyze NLP
    nlp_res = analyze_incident("Two cars had an accident on Banga Road with fuel leak.")

    # 5. Multimodal Accident Assessment
    assessment = assess_accident_evidence(
        cv_candidates=candidates,
        cv_diagnostics={"width": 400, "height": 300},
        nlp_analysis=nlp_res,
    )
    assert assessment["assessment_category"] is not None

    # 6. SOS Location & Hospital Routing
    sos_loc, _ = create_sos_location(31.2185, 75.7852, source=SOURCE_GPS_SOS_PAYLOAD)
    ranker = HospitalRanker()
    routing_data = ranker.rank_hospitals(sos_loc, urgency_level="HIGH")
    assert routing_data["primary_recommendation"] is not None

    # 7. Combined Report
    report = generate_structured_report(
        cv_diagnostics={"width": 400, "height": 300},
        cv_candidates=candidates,
        plate_ocr_result=ocr_res,
        nlp_analysis=nlp_res,
        sos_location=sos_loc,
        hospital_routing_result=routing_data,
        accident_assessment=assessment,
    )
    assert report is not None
    assert "markdown_report" in report
    md = report["markdown_report"]
    assert "[REAL CV]" in md
    assert "[CLASSICAL OCR]" in md
    assert "[CALLER REPORTED]" in md
    assert "[SOS GPS]" in md
    assert "[HOSPITAL ROUTE]" in md
    assert "[COMPUTED COMPARISON]" in md
