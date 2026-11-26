"""THIRD EYE AI — Classical Computer Vision, Number Plate OCR & NLP Incident Analysis.

Main Streamlit Application Entrypoint.
Focused exclusively on classical OpenCV image processing, deterministic template-matching OCR,
and rule-based NLP incident extraction without ML, DL, or RAG models.
"""

import os
from typing import Optional
import streamlit as st
import cv2

from core.cv.image_processing import load_image
from app.components.cv_ui import render_cv_tab
from app.components.ocr_ui import render_ocr_tab
from app.components.nlp_ui import render_nlp_tab
from app.components.report_ui import render_report_tab


# Page configuration
st.set_page_config(
    page_title="Third Eye AI — CV, OCR & NLP",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Third Eye AI 👁️")
st.caption(
    "Classical Computer Vision, Indian Number Plate OCR & Deterministic NLP Incident Analysis"
)


# Helper to locate project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples", "demo")
SAMPLE_CALLS_DIR = os.path.join(BASE_DIR, "data", "sample_calls")


# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.header("📸 Media & Scenario Selection")

    # Image selection
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
            # Fallback to data/samples
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

    # Caller text selection
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
        "*(No PyTorch, YOLO, EasyOCR, or LLMs)*"
    )


# -------------------------------------------------------------
# MAIN CONTENT TABS
# -------------------------------------------------------------
if image_bgr is None:
    st.warning("Please select or upload an image to begin analysis.")
    st.stop()

tab_cv, tab_ocr, tab_nlp, tab_report = st.tabs([
    "📸 Computer Vision",
    "🔤 Plate OCR",
    "📝 NLP Analysis",
    "📊 Combined Incident Report",
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

with tab_report:
    render_report_tab(
        cv_diagnostics=cv_data.get("diagnostics") if cv_data else None,
        cv_candidates=cv_data.get("candidates") if cv_data else None,
        ocr_result=ocr_data,
        nlp_analysis=nlp_data,
    )
