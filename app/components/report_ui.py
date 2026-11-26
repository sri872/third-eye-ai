"""Streamlit UI Component for Combined Forensic Incident Report.

Combines visual CV evidence, OCR readings, and NLP caller extractions
into an executive multi-source forensic summary with explicit provenance.
"""

from typing import Dict, Any, Optional
import streamlit as st

from core.nlp.report_generation import generate_structured_report


def render_report_tab(
    cv_diagnostics: Optional[Dict[str, Any]] = None,
    cv_candidates: Optional[list] = None,
    ocr_result: Optional[Dict[str, Any]] = None,
    nlp_analysis: Optional[Dict[str, Any]] = None,
):
    """Render Section 4: Combined Incident Forensic Report."""
    st.subheader("4. Combined Multi-Source Incident Report")
    st.markdown(
        "Source-separated forensic synthesis of **Visual CV**, **Classical OCR**, and **Caller NLP** findings."
    )

    report_data = generate_structured_report(
        cv_diagnostics=cv_diagnostics,
        cv_candidates=cv_candidates,
        plate_ocr_result=ocr_result,
        nlp_analysis=nlp_analysis,
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
