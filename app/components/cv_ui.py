"""Streamlit UI Component for Computer Vision Analysis.

Provides interactive controls for classical image processing, enhancement,
edge filtering, and geometric vehicle candidate localization.
"""

import streamlit as st
import cv2
import numpy as np

from core.cv.image_processing import (
    load_image,
    enhance_image,
    to_grayscale,
    detect_edges,
    compute_image_diagnostics,
)
from core.cv.vehicle_localization import localize_vehicle_candidates
from core.cv.plate_localization import localize_plate_candidates
from core.cv.visualization import draw_vehicle_candidates, draw_plate_candidates


def render_cv_tab(image_bgr: np.ndarray, image_rgb: np.ndarray):
    """Render Section 1: Classical Computer Vision Analysis."""
    st.subheader("1. Classical Computer Vision Analysis")
    st.markdown(
        "Deterministic image processing using OpenCV & NumPy — **No neural networks or learned detectors.**"
    )

    # Sidebar / expander controls for image enhancement
    with st.expander("🛠️ Classical Image Enhancement & Edge Controls", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            brightness = st.slider("Brightness Offset", -80, 80, 0, step=5)
            contrast = st.slider("Contrast Factor", 0.5, 2.5, 1.0, step=0.1)
        with c2:
            apply_clahe = st.checkbox("Apply CLAHE (Adaptive Histogram Eq)", value=False)
            blur_type = st.selectbox("Smoothing Filter", ["none", "gaussian", "median"])
            blur_kernel = st.slider("Blur Kernel Size", 3, 9, 3, step=2)
        with c3:
            canny_low = st.slider("Canny Low Threshold", 10, 150, 40, step=5)
            canny_high = st.slider("Canny High Threshold", 50, 250, 130, step=5)

    # Apply enhancements
    enhanced_bgr = enhance_image(
        image_bgr,
        brightness=brightness,
        contrast=contrast,
        apply_clahe=apply_clahe,
        blur_type=blur_type,
        blur_kernel=blur_kernel,
    )
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    gray = to_grayscale(enhanced_bgr)
    edges = detect_edges(gray, low_thresh=canny_low, high_thresh=canny_high)

    # 1. Image Quality Diagnostics
    diagnostics = compute_image_diagnostics(image_bgr)
    enhanced_diagnostics = compute_image_diagnostics(enhanced_bgr)
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    col_d1.metric("Dimensions", f"{diagnostics['width']} × {diagnostics['height']}")
    col_d2.metric("Input Mean Brightness", f"{diagnostics['mean_brightness']}")
    col_d3.metric("Exposure Assessment", diagnostics["exposure_assessment"])
    col_d4.metric("Input Clarity (Laplacian)", f"{diagnostics['sharpness_laplacian_var']:.1f}")
    st.caption(
        f"After current enhancement controls: brightness {enhanced_diagnostics['mean_brightness']:.2f}, "
        f"exposure {enhanced_diagnostics['exposure_assessment']}, "
        f"clarity {enhanced_diagnostics['sharpness_laplacian_var']:.1f}"
    )

    # 2. Side-by-side: Original vs Processed Edge View
    st.markdown("#### Visual Transformations")
    v1, v2, v3 = st.columns(3)
    with v1:
        st.caption("Original Image")
        st.image(image_rgb, use_container_width=True)
    with v2:
        st.caption("Enhanced Image")
        st.image(enhanced_rgb, use_container_width=True)
    with v3:
        st.caption(f"Canny Edges ({canny_low}, {canny_high})")
        st.image(edges, use_container_width=True)

    # 3. Vehicle Candidate Region Localization
    st.markdown("#### Vehicle Candidate Region Localization")
    st.info(
        "Candidate regions are localized using horizontal edge density and morphological contour analysis. "
        "These are geometric candidate proposals, not confirmed neural detections."
    )

    candidates = localize_vehicle_candidates(enhanced_bgr)
    vehicle_search_boxes = [item["bbox"] for item in candidates]
    plate_candidates = localize_plate_candidates(
        enhanced_bgr, search_regions=vehicle_search_boxes or None
    )
    annotated_bgr = draw_vehicle_candidates(enhanced_bgr, candidates)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    st.image(
        annotated_rgb,
        caption=f"Detected {len(candidates)} Geometric Vehicle Candidate Regions",
        use_container_width=True,
    )

    # 4. Plate-shaped regions are separate proposals, not verified plates.
    st.markdown("#### Number-Plate Candidate Localization")
    st.caption(
        "Classical morphology and edge geometry propose plate-like rectangles. "
        "OCR validates each crop separately; a green box alone is not a recognized registration."
    )
    st.metric("Plate-shaped regions proposed", len(plate_candidates))
    if plate_candidates:
        plate_annotated = draw_plate_candidates(enhanced_bgr, plate_candidates)
        st.image(
            cv2.cvtColor(plate_annotated, cv2.COLOR_BGR2RGB),
            caption="Plate-shaped candidate proposals",
            use_container_width=True,
        )
        with st.expander("Plate candidate crop diagnostics"):
            plate_cols = st.columns(min(4, len(plate_candidates)))
            for idx, plate in enumerate(plate_candidates):
                bx, by, bw, bh = plate["bbox"]
                crop = enhanced_rgb[by:by + bh, bx:bx + bw]
                with plate_cols[idx % len(plate_cols)]:
                    st.image(
                        crop,
                        caption=f"Plate #{idx + 1} | score {plate['score']:.3f}",
                        use_container_width=True,
                    )
                    st.caption(
                        f"Box: {plate['bbox']} | aspect ratio: {plate['aspect_ratio']}"
                    )
    else:
        st.warning(
            "No plate-shaped proposal was found in this image. The OCR tab will report "
            "that limitation rather than inventing a plate number."
        )

    # Display candidate crops and metrics
    if candidates:
        st.markdown("##### Candidate Region Diagnostics")
        cols = st.columns(min(4, len(candidates)))
        for idx, cand in enumerate(candidates):
            c_col = cols[idx % len(cols)]
            bx, by, bw, bh = cand["bbox"]
            crop_rgb = enhanced_rgb[by : by + bh, bx : bx + bw]
            if crop_rgb.size > 0:
                with c_col:
                    st.image(crop_rgb, caption=f"Region #{idx+1}: {cand['type_hint']}", use_container_width=True)
                    st.caption(
                        f"AR: {cand['aspect_ratio']} | Area: {cand['area_ratio']*100:.1f}% | Heuristic: {cand['heuristic_confidence']}"
                    )

    return {
        "enhanced_bgr": enhanced_bgr,
        "enhanced_rgb": enhanced_rgb,
        "diagnostics": diagnostics,
        "candidates": candidates,
        "plate_candidates": plate_candidates,
        "edges": edges,
    }
