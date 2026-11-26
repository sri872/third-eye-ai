"""Deterministic Multimodal Accident Evidence Assessment.

Fuses classical CV geometric observations with caller NLP transcript extractions
into a transparent 4-tier evidence category without any machine learning or deep learning models.
"""

from typing import List, Dict, Any, Tuple, Optional


# 4-Tier Assessment Framework
CATEGORY_ACCIDENT_DETECTED = "ACCIDENT EVIDENCE DETECTED"
CATEGORY_POSSIBLE_ACCIDENT = "POSSIBLE ACCIDENT"
CATEGORY_INCONCLUSIVE = "INCONCLUSIVE"
CATEGORY_NO_CLEAR_EVIDENCE = "NO CLEAR ACCIDENT EVIDENCE DETECTED"


def assess_accident_evidence(
    cv_candidates: List[Dict[str, Any]],
    cv_diagnostics: Dict[str, Any],
    nlp_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assess accident evidence deterministically from visual and caller signals.

    Rules:
    - 2D box overlap is a supporting geometric cue, not confirmed physical collision.
    - An uncorroborated single image is out-of-domain for dynamic collision confirmation.
    - Strictly no false safety claims ('SAFE' or 'ROAD SAFE').
    - Human operator review is mandatory.
    """
    num_vehicles = len(cv_candidates) if cv_candidates else 0
    max_iou, num_overlapping_pairs = _compute_candidate_overlaps(cv_candidates)

    has_caller_accident = False
    reported_injuries = []
    reported_hazards = []

    if nlp_analysis:
        entities = nlp_analysis.get("entities", {})
        has_caller_accident = nlp_analysis.get("primary_incident_type") in (
            "VEHICLE_COLLISION", "ROLLOVER_COLLISION", "HAZARDOUS_FIRE_OR_SPILL", "MULTI_VEHICLE_COLLISION"
        ) or nlp_analysis.get("urgency_level") in ("CRITICAL", "HIGH")
        reported_injuries = entities.get("injuries", [])
        reported_hazards = entities.get("hazards", [])

    # Multi-Signal Deterministic Decision Logic
    # 1. Corroborated Accident Evidence: Significant geometric overlap AND positive caller report
    if num_overlapping_pairs >= 1 and has_caller_accident:
        tier = CATEGORY_ACCIDENT_DETECTED
        confidence = "CORROBORATED_EVIDENCE"
        rationale = (
            f"Visual vehicle bounding box overlap (max IoU: {max_iou:.2f}, {num_overlapping_pairs} overlapping pair(s)) "
            "is corroborated by positive caller emergency collision and trauma reports."
        )

    # 2. Possible Accident: Geometric overlap in image alone OR caller reports collision but image has separate boxes
    elif num_overlapping_pairs >= 1 and not has_caller_accident:
        tier = CATEGORY_POSSIBLE_ACCIDENT
        confidence = "UNVERIFIED_VISUAL_CUE"
        rationale = (
            f"Visual candidate bounding box overlap observed (max IoU: {max_iou:.2f}), but no caller emergency transcript "
            "confirms a collision. Still images cannot confirm physical impact vs close vehicle proximity."
        )
    elif num_overlapping_pairs == 0 and has_caller_accident:
        tier = CATEGORY_POSSIBLE_ACCIDENT
        confidence = "UNVERIFIED_CALLER_CUE"
        rationale = (
            "Emergency caller reports an active collision or injuries, but uploaded still image does not show "
            "direct geometric overlap among vehicle candidate proposals."
        )

    # 3. Inconclusive: Poor clarity or marginal proximity
    elif cv_diagnostics.get("clarity_assessment") == "BLURRED_OR_DEFOCUSED" or (0.02 < max_iou < 0.10):
        tier = CATEGORY_INCONCLUSIVE
        confidence = "AMBIGUOUS_CONDITIONS"
        rationale = (
            "Visual conditions or vehicle spacing are ambiguous. Image clarity is reduced or vehicles are in close traffic flow "
            "without structural distortion cues."
        )

    # 4. No Clear Accident Evidence Detected
    else:
        tier = CATEGORY_NO_CLEAR_EVIDENCE
        confidence = "BASELINE_MONITORING"
        rationale = (
            "No geometric collision overlap detected among vehicle candidate proposals, and no caller emergency indicators reported."
        )

    disclaimer = (
        "MANDATORY SAFETY DISCLAIMER: This assessment is an automated classical heuristic. "
        "The absence of detected evidence does NOT confirm that the roadway is safe or clear. "
        "Human dispatcher review is required before initiating real-world emergency actions."
    )

    return {
        "assessment_category": tier,
        "confidence_level": confidence,
        "rationale": rationale,
        "max_vehicle_iou": round(max_iou, 3),
        "overlapping_pairs_count": num_overlapping_pairs,
        "candidate_count": num_vehicles,
        "caller_corroborated": has_caller_accident,
        "human_review_required": True,
        "safety_disclaimer": disclaimer,
    }


def _compute_candidate_overlaps(candidates: List[Dict[str, Any]]) -> Tuple[float, int]:
    """Compute maximum pairwise IoU and count of overlapping candidate pairs."""
    if not candidates or len(candidates) < 2:
        return 0.0, 0

    max_iou = 0.0
    overlapping_count = 0

    boxes = [c["bbox"] for c in candidates]
    n = len(boxes)

    for i in range(n):
        x1, y1, w1, h1 = boxes[i]
        for j in range(i + 1, n):
            x2, y2, w2, h2 = boxes[j]

            # Intersection
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = float(iw * ih)

            # Union
            union = float(w1 * h1 + w2 * h2 - inter)
            iou = inter / union if union > 0 else 0.0

            if iou > max_iou:
                max_iou = iou
            if iou > 0.08:
                overlapping_count += 1

    return max_iou, overlapping_count
