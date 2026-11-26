"""Streamlit UI Component for Combined Forensic Incident Report.

Combines visual CV evidence, OCR readings, NLP caller extractions,
SOS location coordinates, and intelligent hospital routing into an executive report.
"""

from typing import Dict, Any, Optional
import streamlit as st

from core.nlp.report_generation import generate_structured_report


def render_report_tab(
    cv_diagnostics: Optional[Dict[str, Any]] = None,
    cv_candidates: Optional[list] = None,
    ocr_result: Optional[Dict[str, Any]] = None,
    nlp_analysis: Optional[Dict[str, Any]] = None,
    sos_location: Optional[Any] = None,
    hospital_data: Optional[Dict[str, Any]] = None,
    accident_assessment: Optional[Dict[str, Any]] = None,
):
    """Render Section 5: Combined Incident Forensic Report."""
    st.subheader("5. Combined Multi-Source Forensic Report")
    st.markdown(
        "Source-separated synthesis of **Visual CV**, **Classical OCR**, **Caller NLP**, **SOS GPS**, and **Road Routing** findings."
    )

    report_data = generate_structured_report(
        cv_diagnostics=cv_diagnostics,
        cv_candidates=cv_candidates,
        plate_ocr_result=ocr_result,
        nlp_analysis=nlp_analysis,
        sos_location=sos_location,
        hospital_routing_result=hospital_data,
        accident_assessment=accident_assessment,
    )

    # Cross-Verification Status Banner
    cross = report_data["cross_check"]
    st.info(f"**Multi-Source Cross Check**: {cross['vehicle_count_comparison']}")

    # Formatted Markdown View
    st.markdown(report_data["markdown_report"])

    # Download Button
    st.download_button(
        label="📥 Download Forensic Incident Report (.md)",
        data=report_data["markdown_report"],
        file_name="third_eye_incident_report.md",
        mime="text/markdown",
    )

    return report_data
