"""THIRD EYE AI — classical CV, OCR, NLP, SOS location and hospital routing."""
from __future__ import annotations

import os
import streamlit as st

from core.cv.image_processing import load_image
from core.accident.evidence_assessment import assess_accident_evidence
from app.components.cv_ui import render_cv_tab
from app.components.ocr_ui import render_ocr_tab
from app.components.nlp_ui import render_nlp_tab
from app.components.sos_ui import render_sos_location_component
from app.components.routing_ui import render_hospital_routing_tab
from app.components.report_ui import render_report_tab


st.set_page_config(
    page_title="Third Eye AI — CV, OCR, NLP & SOS Routing",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Third Eye AI 🚑")
st.caption(
    "Classical Computer Vision • Indian Number Plate OCR • Rule-Based NLP • "
    "SOS GPS • Nearest Hospital Road Routing"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples", "demo")
SAMPLE_CALLS_DIR = os.path.join(BASE_DIR, "data", "sample_calls")

with st.sidebar:
    st.header("Incident inputs")
    st.subheader("Visual input (optional)")
    demo_images = {
        "Single Car Plate (TS07JA0566)": "single_car_plate.jpg",
        "Two-Car Collision": "two_car_collision.jpg",
        "Heavy Traffic Scene": "heavy_traffic.jpg",
    }
    image_mode = st.radio("Image source", ["Demo sample", "Upload custom image"])
    image_bgr, image_rgb, ground_truth = None, None, None

    if image_mode == "Demo sample":
        chosen_label = st.selectbox("Select sample image", list(demo_images))
        image_path = os.path.join(SAMPLES_DIR, demo_images[chosen_label])
        if not os.path.exists(image_path):
            image_path = os.path.join(BASE_DIR, "data", "samples", demo_images[chosen_label])
        if os.path.exists(image_path):
            orientation_options = {
                "Keep original orientation": 0,
                "Rotate 90° clockwise": 90,
                "Rotate 180°": 180,
                "Rotate 90° counterclockwise": 270,
            }
            default_orientation = (
                "Rotate 90° counterclockwise"
                if "TS07JA0566" in chosen_label
                else "Keep original orientation"
            )
            orientation_label = st.selectbox(
                "Image orientation",
                list(orientation_options),
                index=list(orientation_options).index(default_orientation),
                key=f"demo-orientation-{chosen_label}",
            )
            image_bgr, image_rgb = load_image(
                image_path,
                rotation_degrees=orientation_options[orientation_label],
            )
            if "TS07JA0566" in chosen_label:
                # The demo file has a known registration string for evaluation only.
                ground_truth = "TS07JA0566"
        else:
            st.error(f"Sample image not found: {image_path}")
    else:
        uploaded = st.file_uploader(
            "Upload road image", type=["jpg", "jpeg", "png"], key="incident-image"
        )
        orientation_options = {
            "Keep uploaded orientation": 0,
            "Rotate 90° clockwise": 90,
            "Rotate 180°": 180,
            "Rotate 90° counterclockwise": 270,
        }
        orientation_label = st.selectbox(
            "Uploaded image orientation",
            list(orientation_options),
            key="uploaded-image-orientation",
        )
        if uploaded is not None:
            image_bgr, image_rgb = load_image(
                uploaded,
                rotation_degrees=orientation_options[orientation_label],
            )
            ground_truth = None

    st.divider()
    st.subheader("Caller transcript (NLP)")
    call_options = {
        "Collision on Highway (High Urgency)": "call_demo_01_collision.txt",
        "Single Vehicle Overturned": "call_01.txt",
        "Contradictory Statement Sample": "call_demo_04_contradiction.txt",
        "Traffic Delay / Fender Bender": "call_demo_02_traffic.txt",
    }
    call_mode = st.radio("Transcript source", ["Demo scenario", "Custom transcript"])
    if call_mode == "Demo scenario":
        label = st.selectbox("Select incident scenario", list(call_options))
        call_path = os.path.join(SAMPLE_CALLS_DIR, call_options[label])
        if os.path.exists(call_path):
            with open(call_path, "r", encoding="utf-8") as source_file:
                selected_call_text = source_file.read().strip()
        else:
            selected_call_text = (
                "Two vehicles collided near Phagwara on GT Road. One driver is injured."
            )
    else:
        selected_call_text = st.text_area(
            "Enter caller-reported incident description",
            value="",
            height=120,
            placeholder="Describe the incident, injuries, hazards and known location.",
        )

    st.divider()
    st.caption(
        "CV uses classical OpenCV heuristics; OCR uses character templates; NLP uses "
        "transparent rules. Hospital routing is conventional NetworkX road-graph software. "
        "No ML, DL, RAG or neural OCR model is used."
    )

tab_cv, tab_ocr, tab_nlp, tab_sos, tab_report = st.tabs(
    [
        "📸 Computer Vision",
        "🔤 Plate OCR",
        "📝 NLP & Evidence",
        "📍 SOS & Hospital Route",
        "📊 Combined Report",
    ]
)

cv_data = None
ocr_data = None
nlp_data = None
accident_assessment = {
    "assessment_category": "NOT DETERMINED",
    "confidence_level": "NOT ASSESSED",
    "rationale": "No visual upload was supplied; no image-based assessment was performed.",
    "max_vehicle_iou": None,
    "human_review_required": True,
    "safety_disclaimer": (
        "Caller-reported text is not visual confirmation. Human review is required."
    ),
}

with tab_cv:
    st.subheader("Computer Vision analysis")
    if image_bgr is None:
        st.info("Upload or select an image in the sidebar to run visual analysis.")
    else:
        cv_data = render_cv_tab(image_bgr, image_rgb)

with tab_ocr:
    st.subheader("Indian number-plate OCR")
    if image_bgr is None:
        st.info("Upload or select an image in the sidebar to run plate localization and OCR.")
    else:
        ocr_data = render_ocr_tab(
            image_bgr,
            cv_candidates=cv_data.get("candidates") if cv_data else None,
            ground_truth=ground_truth,
        )

with tab_nlp:
    nlp_data = render_nlp_tab(default_text=selected_call_text)
    if image_bgr is not None:
        st.divider()
        st.subheader("Deterministic accident evidence assessment")
        accident_assessment = assess_accident_evidence(
            cv_candidates=cv_data.get("candidates", []) if cv_data else [],
            cv_diagnostics=cv_data.get("diagnostics", {}) if cv_data else {},
            nlp_analysis=nlp_data,
        )
        tier = accident_assessment["assessment_category"]
        if tier == "ACCIDENT EVIDENCE DETECTED":
            st.error(f"ASSESSMENT: {tier}")
        elif tier == "POSSIBLE ACCIDENT":
            st.warning(f"ASSESSMENT: {tier}")
        elif tier == "INCONCLUSIVE":
            st.info(f"ASSESSMENT: {tier}")
        else:
            st.success(f"ASSESSMENT: {tier}")
        st.markdown(f"**Rationale:** {accident_assessment['rationale']}")
        st.caption(accident_assessment["safety_disclaimer"])
    else:
        st.info(
            "No CV assessment was generated because no image was uploaded. "
            "The NLP output remains caller-reported information."
        )

with tab_sos:
    sos_location = render_sos_location_component()
    st.divider()
    urgency = nlp_data.get("urgency_level", "HIGH") if nlp_data else "HIGH"
    routing_data = render_hospital_routing_tab(
        sos_location,
        urgency_level=urgency,
        incident_text=selected_call_text,
    )

with tab_report:
    render_report_tab(
        cv_diagnostics=cv_data.get("diagnostics") if cv_data else None,
        cv_candidates=cv_data.get("candidates") if cv_data else None,
        ocr_result=ocr_data,
        nlp_analysis=nlp_data,
        sos_location=sos_location,
        hospital_data=routing_data,
        accident_assessment=accident_assessment,
    )
