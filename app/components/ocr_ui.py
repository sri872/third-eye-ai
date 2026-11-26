"""Streamlit UI Component for Number Plate OCR.

Provides classical plate localization, character segmentation, template matching,
and Indian registration plate format validation.
"""

from typing import Dict, Any, Optional
import streamlit as st
import cv2
import numpy as np

from core.cv.plate_localization import localize_plate_candidates
from core.ocr.plate_reader import ClassicalPlateReader


@st.cache_resource
def get_plate_reader() -> ClassicalPlateReader:
    return ClassicalPlateReader()


def render_ocr_tab(
    image_bgr: np.ndarray,
    cv_candidates: Optional[list] = None,
    ground_truth: Optional[str] = None,
):
    """Render Section 2: Classical Number Plate OCR."""
    st.subheader("2. Indian Number Plate OCR (Classical Template Matching)")
    st.markdown(
        "Character segmentation & deterministic template matching — **No EasyOCR, Tesseract, or neural OCR engines.**"
    )

    plate_reader = get_plate_reader()

    # 1. Plate Localization
    search_boxes = [c["bbox"] for c in cv_candidates] if cv_candidates else None
    plates = localize_plate_candidates(image_bgr, search_regions=search_boxes)

    if not plates:
        # Fallback to entire image search if vehicle region search found nothing
        plates = localize_plate_candidates(image_bgr, search_regions=None)

    col_l1, col_l2 = st.columns([1, 2])

    with col_l1:
        st.markdown("#### Candidate Plate Detection")
        if not plates:
            st.warning("No candidate plate regions localized by morphological filter.")
            return None

        st.success(f"Localized {len(plates)} plate candidate region(s).")
        plate_idx = st.selectbox(
            "Select Candidate Plate",
            range(len(plates)),
            format_func=lambda i: f"Candidate #{i+1} (Aspect Ratio: {plates[i]['aspect_ratio']}, Score: {plates[i]['score']})",
        )
        selected_plate = plates[plate_idx]
        plate_crop = selected_plate["crop"]
        st.image(
            cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB),
            caption=f"Candidate #{plate_idx+1} Crop",
            use_container_width=True,
        )

    # 2. Run Classical Plate Reading Pipeline
    ocr_result = plate_reader.read_plate(plate_crop, ground_truth=ground_truth)

    with col_l2:
        st.markdown("#### Recognition & Validation")
        status = ocr_result["status"]
        if status == "VERIFIED":
            st.success(f"**PLATE STATUS: VERIFIED** — `{ocr_result['normalized_text']}`")
        elif status == "UNVERIFIED":
            st.warning(f"**PLATE STATUS: UNVERIFIED** — `{ocr_result['normalized_text'] or ocr_result['raw_text']}`")
        else:
            st.error(f"**PLATE STATUS: UNREADABLE / NOT VERIFIED** (Raw text: `{ocr_result['raw_text'] or 'NONE'}`)")

        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Raw OCR String", ocr_result["raw_text"] or "—")
        c_m2.metric("Mean Template Similarity", f"{ocr_result['mean_similarity']:.3f}")
        c_m3.metric("Characters Segmented", f"{ocr_result['character_count']}")

        val = ocr_result["validation"]
        if val["is_valid"]:
            st.info(
                f"**Format Valid**: {val['notes']} | State: **{val['state_name']} ({val['state_code']})** | RTO: **{val['rto_code']}** | Series: **{val['series']}** | Number: **{val['registration_number']}**"
            )
        else:
            st.caption(f"Validation: {val['notes']}")

        if ground_truth:
            if ocr_result["exact_match"]:
                st.success(f"Exact match against ground truth `{ground_truth}`: **PASSED**")
            else:
                st.info(f"Ground truth: `{ground_truth}` | Normalized candidate: `{ocr_result['normalized_text']}`")

    # 3. Character-by-Character Segmentation and Template Matching Details
    if ocr_result.get("characters"):
        st.markdown("#### Character Segmentation & Template Scoring Breakdown")
        char_cols = st.columns(min(10, len(ocr_result["characters"])))

        for i, char_info in enumerate(ocr_result["characters"]):
            col_target = char_cols[i % len(char_cols)]
            with col_target:
                char_img = char_info["char_image"]
                st.image(char_img, caption=f"Pos {i}: {char_info['character']}", width=48)
                st.caption(f"Score: {char_info['similarity']:.2f}")

        # Expandable table of top-3 candidate glyphs
        with st.expander("🔍 Detailed Character Match Scores"):
            detail_rows = []
            for ch in ocr_result["characters"]:
                top3_str = ", ".join([f"{c} ({s:.2f})" for c, s in ch["top_candidates"]])
                detail_rows.append({
                    "Position": ch["position"],
                    "Selected Glyph": ch["character"],
                    "Similarity": ch["similarity"],
                    "Top 3 Template Matches": top3_str,
                    "Bounding Box": str(ch["bbox"]),
                })
            st.table(detail_rows)

    return ocr_result
