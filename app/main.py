"""THIRD EYE AI — Classical Computer Vision, Number Plate OCR, NLP, SOS Location & Hospital Routing.

Main Streamlit Application Entrypoint.
Focused on deterministic CV, classical template-matching OCR, rule-based NLP,
and road-network emergency hospital routing without ML, DL, or RAG models.
"""

import os
import streamlit as st
import cv2

from core.cv.image_processing import load_image
from core.accident.evidence_assessment import assess_accident_evidence
from app.components.cv_ui import render_cv_tab
from app.components.ocr_ui import render_ocr_tab
from app.components.nlp_ui import render_nlp_tab
from app.components.sos_ui import render_sos_location_component
from app.components.routing_ui import render_hospital_routing_tab
from app.components.report_ui import render_report_tab


# Page configuration
st.set_page_config(
    page_title="Third Eye AI — CV, OCR, NLP & Emergency Routing",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Third Eye AI 🚑")
st.caption(
    "Classical Computer Vision, Indian Number Plate OCR, Rule-Based NLP & Intelligent Hospital Routing"
)


# Helper to locate project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples", "demo")
SAMPLE_CALLS_DIR = os.path.join(BASE_DIR, "data", "sample_calls")


# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Incident Inputs & Media")

    # 1. Image selection
    st.subheader("1. Visual Input (Image)")
    demo_images = {
        "Single Car Plate (TS07JA0566)": "single_car_plate.jpg",
        "Two-Car Collision": "two_car_collision.jpg",
        "Heavy Traffic Scene": "heavy_traffic.jpg",
    }
    image_mode = st.radio("Image Source", ["Demo Sample", "Upload Custom Image"])

    image_bgr, image_rgb = None, None
    ground_truth = None

    if image_mode == "Demo Sample":
        chosen_label = st.selectbox("Select Sample Image", list(demo_images.keys()))
        chosen_filename = demo_images[chosen_label]
        img_path = os.path.join(SAMPLES_DIR, chosen_filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(BASE_DIR, "data", "samples", chosen_filename)

        if os.path.exists(img_path):
            image_bgr, image_rgb = load_image(img_path)
            if "TS07JA0566" in chosen_label:
                ground_truth = "TS07JA0566"
        else:
            st.error(f"Sample not found: {img_path}")
    else:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_bgr, image_rgb = load_image(uploaded_file)
            custom_gt = st.text_input("Known Ground Truth Plate (Optional)", value="")
            if custom_gt.strip():
                ground_truth = custom_gt.strip()

    st.divider()

    # 2. Caller text selection
    st.subheader("2. Caller Transcript (NLP)")
    call_options = {
        "Collision on Highway (High Urgency)": "call_demo_01_collision.txt",
        "Single Vehicle Overturned": "call_01.txt",
        "Contradictory Statement Sample": "call_demo_04_contradiction.txt",
        "Traffic Delay / Fender Bender": "call_demo_02_traffic.txt",
    }
    call_mode = st.radio("Transcript Source", ["Demo Scenario", "Custom Transcript"])
    selected_call_text = ""

    if call_mode == "Demo Scenario":
        chosen_call_label = st.selectbox("Select Preset Scenario", list(call_options.keys()))
        call_filename = call_options[chosen_call_label]
        call_path = os.path.join(SAMPLE_CALLS_DIR, call_filename)
        if os.path.exists(call_path):
            with open(call_path, "r", encoding="utf-8") as f:
                selected_call_text = f.read().strip()
        else:
            selected_call_text = "Two vehicles involved in an accident near Phagwara on GT Road. One driver is injured."
    else:
        selected_call_text = st.text_area(
            "Enter Emergency Call Text",
            value="Speeding truck collided with white car near Banga Road. Fuel is leaking, driver unconscious.",
            height=120,
        )

    st.divider()
    st.info(
        "💡 **Technology Notice**:\n"
        "- CV: Classical OpenCV (Contours, Sobel, Morphology)\n"
        "- OCR: Template Matching for Indian Plates\n"
        "- NLP: Deterministic Regex & Rule-Based Extraction\n"
        "- Routing: Real Road Graph NetworkX Router\n"
        "*(Zero PyTorch, YOLO, EasyOCR, or LLMs)*"
    )


# -------------------------------------------------------------
# MAIN CONTENT TABS
# -------------------------------------------------------------
if image_bgr is None:
    st.warning("Please select or upload an image to begin analysis.")
    st.stop()

tab_cv, tab_ocr, tab_nlp, tab_sos, tab_report = st.tabs([
    "📸 Computer Vision",
    "🔤 Plate OCR",
    "📝 NLP & Evidence",
    "📍 SOS & Hospital Route",
    "📊 Combined Forensic Report",
])

with tab_cv:
    cv_data = render_cv_tab(image_bgr, image_rgb)

with tab_ocr:
    ocr_data = render_ocr_tab(
        image_bgr,
        cv_candidates=cv_data.get("candidates") if cv_data else None,
        ground_truth=ground_truth,
    )

with tab_nlp:
    nlp_data = render_nlp_tab(default_text=selected_call_text)

    # Render Deterministic Multimodal Accident Evidence Assessment
    st.divider()
    st.subheader("⚖️ Multimodal Accident Evidence Assessment (Deterministic)")
    accident_assessment = assess_accident_evidence(
        cv_candidates=cv_data.get("candidates", []) if cv_data else [],
        cv_diagnostics=cv_data.get("diagnostics", {}) if cv_data else {},
        nlp_analysis=nlp_data,
    )

    tier = accident_assessment["assessment_category"]
    if tier == "ACCIDENT EVIDENCE DETECTED":
        st.error(f"🔴 **ASSESSMENT: {tier}**")
    elif tier == "POSSIBLE ACCIDENT":
        st.warning(f"🟠 **ASSESSMENT: {tier}**")
    elif tier == "INCONCLUSIVE":
        st.info(f"🟡 **ASSESSMENT: {tier}**")
    else:
        st.success(f"🟢 **ASSESSMENT: {tier}**")

    st.markdown(f"**Rationale**: {accident_assessment['rationale']}")
    st.caption(accident_assessment["safety_disclaimer"])

with tab_sos:
    sos_location = render_sos_location_component()
    st.divider()
    urgency = nlp_data.get("urgency_level", "HIGH") if nlp_data else "HIGH"
    routing_data = render_hospital_routing_tab(sos_location, urgency_level=urgency)

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
