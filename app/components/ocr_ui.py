"""Streamlit UI for multi-candidate classical number-plate OCR."""
from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import streamlit as st
import numpy as np

from core.cv.plate_localization import localize_plate_candidates
from core.ocr.plate_reader import ClassicalPlateReader


@st.cache_resource
def get_plate_reader() -> ClassicalPlateReader:
    return ClassicalPlateReader()


def _ocr_rank(candidate: Dict[str, Any], result: Dict[str, Any]) -> tuple:
    validation = bool((result.get("validation") or {}).get("is_valid"))
    exact = bool(result.get("exact_match"))
    count = int(result.get("character_count") or 0)
    expected_length = max(0.0, 1.0 - abs(count - 10) / 10.0)
    return (
        exact,
        validation,
        result.get("mean_similarity", 0.0),
        expected_length,
        candidate.get("score", 0.0),
    )


def render_ocr_tab(
    image_bgr: np.ndarray,
    cv_candidates: Optional[list] = None,
    ground_truth: Optional[str] = None,
    plate_candidates: Optional[list] = None,
):
    """Localize multiple plate proposals, OCR each and rank by transparent evidence."""
    st.subheader("2. Indian Number Plate OCR — classical image processing")
    st.markdown(
        "Multi-path morphology and gradient localization, adaptive/Otsu threshold variants, "
        "glyph segmentation and deterministic template matching. **No neural OCR.**"
    )
    st.caption(
        "A candidate is a plate-shaped region proposal, not a confirmed plate. "
        "Template similarity is not a calibrated probability. Without trusted ground truth, "
        "recognized text remains unverified."
    )

    if image_bgr is None or image_bgr.size == 0:
        st.info("Select or upload an image in the sidebar first.")
        return None

    plate_reader = get_plate_reader()

    # Search the full frame and the geometric CV regions together. A weak ROI
    # result must never suppress the full-frame search.
    search_boxes = [c.get("bbox") for c in (cv_candidates or []) if c.get("bbox")]
    plates = (
        plate_candidates
        if plate_candidates is not None
        else localize_plate_candidates(image_bgr, search_regions=search_boxes or None)
    )

    if not plates:
        st.warning(
            "No plate-shaped region was proposed. Try a clear, upright image with the "
            "plate visible and occupying a larger part of the frame. No text has been invented."
        )
        return {
            "status": "NO_PLATE_DETECTED",
            "raw_text": "",
            "normalized_text": "",
            "mean_similarity": 0.0,
            "character_count": 0,
            "characters": [],
            "validation": {"is_valid": False, "notes": "No plate candidate was localized."},
            "is_verified": False,
            "ground_truth": ground_truth,
            "exact_match": False,
            "candidate_count": 0,
        }

    # Evaluate a bounded shortlist to avoid spending excessive time on hundreds
    # of false contour blobs. The localization list is already score-ranked.
    shortlist = plates[:16]
    proposal_indices = {id(candidate): idx + 1 for idx, candidate in enumerate(plates)}
    evaluated = []
    progress = st.progress(0, text="Evaluating plate-shaped candidates...")
    for idx, candidate in enumerate(shortlist):
        result = plate_reader.read_plate(candidate["crop"], ground_truth=ground_truth)
        evaluated.append((candidate, result))
        progress.progress((idx + 1) / len(shortlist), text=f"Evaluating candidate {idx + 1}/{len(shortlist)}")
    progress.empty()

    ranked = sorted(
        evaluated,
        key=lambda pair: _ocr_rank(pair[0], pair[1]),
        reverse=True,
    )
    best_candidate, best_result = ranked[0]

    st.markdown("#### Candidate localization and OCR ranking")
    rows = []
    for idx, (candidate, result) in enumerate(ranked, start=1):
        validation = result.get("validation") or {}
        rows.append({
            "Rank": idx,
            "Candidate": f"#{proposal_indices[id(candidate)]}",
            "BBox (x, y, w, h)": tuple(candidate["bbox"]),
            "Proposal score": candidate["score"],
            "OCR text": result.get("raw_text") or "—",
            "Characters": result.get("character_count", 0),
            "Template similarity": result.get("mean_similarity", 0.0),
            "Format": "Valid pattern" if validation.get("is_valid") else "Not validated",
            "Ground-truth match": "Yes" if result.get("exact_match") else "No / not supplied",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    labels = [
        (
            f"Rank {rank} — proposal #{proposal_indices[id(candidate)]} | "
            f"{result.get('raw_text') or 'NO TEXT'} | "
            f"{result.get('character_count', 0)} chars | "
            f"similarity {result.get('mean_similarity', 0.0):.3f}"
        )
        for rank, (candidate, result) in enumerate(ranked, start=1)
    ]
    selected_label = st.selectbox(
        "Inspect a candidate",
        labels,
        index=0,
        key="ocr-ranked-candidate",
    )
    selected_index = labels.index(selected_label)
    selected_candidate, ocr_result = ranked[selected_index]

    left, right = st.columns([1, 2])
    with left:
        st.markdown("#### Selected plate crop")
        st.image(
            cv2.cvtColor(selected_candidate["crop"], cv2.COLOR_BGR2RGB),
            caption=f"Proposal #{proposal_indices[id(selected_candidate)]} — {selected_candidate['method']}",
            use_container_width=True,
        )
        if ocr_result.get("enhanced_gray") is not None:
            st.image(
                ocr_result["enhanced_gray"],
                caption="CLAHE-enhanced grayscale",
                use_container_width=True,
            )
        if ocr_result.get("binary_plate") is not None:
            st.image(
                ocr_result["binary_plate"],
                caption=f"Selected binary variant: {ocr_result.get('preprocessing_method', 'N/A')}",
                use_container_width=True,
            )

    with right:
        st.markdown("#### Recognition and validation")
        status = ocr_result.get("status", "UNREADABLE")
        if status == "VERIFIED":
            st.success(f"**GROUND-TRUTH MATCH** — `{ocr_result.get('normalized_text')}`")
        elif (ocr_result.get("validation") or {}).get("is_valid"):
            st.warning(
                f"**FORMAT-VALID CANDIDATE — NOT VERIFIED** — "
                f"`{ocr_result.get('normalized_text') or ocr_result.get('raw_text')}`"
            )
        else:
            st.error(
                f"**NOT VERIFIED** — raw candidate: "
                f"`{ocr_result.get('raw_text') or 'NONE'}`"
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Raw OCR string", ocr_result.get("raw_text") or "—")
        c2.metric("Template similarity", f"{ocr_result.get('mean_similarity', 0.0):.3f}")
        c3.metric("Characters segmented", str(ocr_result.get("character_count", 0)))

        validation = ocr_result.get("validation") or {}
        st.caption(validation.get("notes", "Format not validated."))
        if validation.get("is_valid"):
            st.info(
                f"State: **{validation.get('state_name')} ({validation.get('state_code')})** | "
                f"RTO: **{validation.get('rto_code')}** | "
                f"Series: **{validation.get('series')}** | "
                f"Number: **{validation.get('registration_number')}**"
            )
        if ground_truth:
            if ocr_result.get("exact_match"):
                st.success(f"Exact match against supplied ground truth `{ground_truth}`.")
            else:
                st.info(
                    f"Supplied ground truth: `{ground_truth}`; "
                    f"candidate: `{ocr_result.get('normalized_text') or ocr_result.get('raw_text') or 'unreadable'}`."
                )
        else:
            st.info("No ground-truth value was supplied. OCR output is an unverified candidate.")

    attempts = ocr_result.get("variant_attempts") or []
    if attempts:
        with st.expander("Threshold variant comparison"):
            st.dataframe(attempts, use_container_width=True, hide_index=True)

    characters = ocr_result.get("characters") or []
    if characters:
        with st.expander("Character segmentation and template scores"):
            char_cols = st.columns(min(10, len(characters)))
            for idx, char in enumerate(characters):
                with char_cols[idx % len(char_cols)]:
                    st.image(char["char_image"], caption=f"{idx + 1}: {char['character']}", width=48)
                    st.caption(f"{char['similarity']:.2f}")
            st.dataframe(
                [
                    {
                        "Position": item["position"],
                        "Selected glyph": item["character"],
                        "Similarity": item["similarity"],
                        "Top candidates": ", ".join(
                            f"{glyph} ({score:.2f})"
                            for glyph, score in item["top_candidates"]
                        ),
                        "BBox": tuple(item["bbox"]),
                    }
                    for item in characters
                ],
                use_container_width=True,
                hide_index=True,
            )

    # Preserve selected result fields for the combined forensic report.
    returned = dict(ocr_result)
    returned["candidate_count"] = len(plates)
    returned["selected_candidate_bbox"] = selected_candidate["bbox"]
    returned["localization_method"] = selected_candidate["method"]
    returned["ranked_candidate_summaries"] = [
        {
            "bbox": candidate["bbox"],
            "proposal_score": candidate["score"],
            "raw_text": result.get("raw_text", ""),
            "character_count": result.get("character_count", 0),
            "mean_similarity": result.get("mean_similarity", 0.0),
            "format_valid": bool((result.get("validation") or {}).get("is_valid")),
        }
        for candidate, result in ranked
    ]
    return returned
