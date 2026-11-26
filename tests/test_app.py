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


def test_component_imports():
    """Verify Streamlit UI components import cleanly."""
    from app.components import cv_ui
    from app.components import ocr_ui
    from app.components import nlp_ui
    from app.components import report_ui
    assert cv_ui is not None
    assert ocr_ui is not None
    assert nlp_ui is not None
    assert report_ui is not None


def test_end_to_end_pipeline_integration():
    """Test complete pipeline from image to OCR to NLP to combined report."""
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

    # 5. Combined Report
    report = generate_structured_report(
        cv_diagnostics={"width": 400, "height": 300},
        cv_candidates=candidates,
        plate_ocr_result=ocr_res,
        nlp_analysis=nlp_res,
    )
    assert report is not None
    assert "markdown_report" in report
    assert len(report["markdown_report"]) > 100
