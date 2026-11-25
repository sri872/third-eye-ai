"""Transparent rule-based heuristic scoring for accident analysis in Project Third-Eye.

Computes a deterministic score in [0.0, 1.0] from geometric detection features.
Identifies which forensic rules triggered.
Strict rule: Output is labeled method="RULE_BASED_HEURISTIC" and is NOT a model probability.
"""

from typing import Any, Dict, List


def evaluate_accident_rules(features: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate geometric features against forensic collision rules.

    Args:
        features: Feature dictionary computed by core/ml/accident_features.py.

    Returns:
        Dict with keys:
            - method: 'RULE_BASED_HEURISTIC'
            - score: float in [0.0, 1.0] (deterministic heuristic score)
            - rules_fired: List of descriptive rule strings
            - is_accident: bool (score >= 0.45)
            - severity: 'none', 'minor', 'moderate', 'critical'
    """
    score = 0.0
    rules_fired: List[str] = []

    v_count = features.get("vehicle_count", 0)
    p_count = features.get("person_count", 0)
    max_iou = features.get("max_vehicle_iou", 0.0)
    max_overlap_min = features.get("max_overlap_min", 0.0)
    heavy_pairs = features.get("heavily_overlapping_pairs", 0)
    v_p_overlap = features.get("vehicle_person_overlap_count", 0)
    ar_outliers = features.get("aspect_ratio_outliers", 0)
    crowd_density = features.get("crowd_density", 0.0)

    # Rule 1: High vehicle overlap (Direct structural collision / T-bone / Rear-end)
    if max_iou >= 0.25 or max_overlap_min >= 0.35:
        score += 0.50
        rules_fired.append(
            f"RULE_HIGH_VEHICLE_OVERLAP: Severe vehicle interpenetration detected (IoU={max_iou:.2f}, overlap_ratio={max_overlap_min:.2f})"
        )
    elif max_iou >= 0.15 or max_overlap_min >= 0.20:
        score += 0.30
        rules_fired.append(
            f"RULE_MODERATE_VEHICLE_OVERLAP: Noticeable vehicle bounding box overlap (IoU={max_iou:.2f})"
        )

    # Rule 2: Multi-vehicle pileup (multiple overlapping pairs)
    if heavy_pairs >= 2:
        score += 0.25
        rules_fired.append(
            f"RULE_MULTI_VEHICLE_PILEUP: Multiple vehicle pairs heavily overlapping ({heavy_pairs} pairs)"
        )

    # Rule 3: Vehicle-person intersection (Pedestrian struck or trapped)
    if v_p_overlap >= 1:
        score += 0.35
        rules_fired.append(
            f"RULE_PEDESTRIAN_IMPACT: Person-vehicle bounding box intersection detected ({v_p_overlap} overlaps)"
        )

    # Rule 4: Vehicle deformation or rollover (aspect ratio outlier with multiple vehicles)
    if ar_outliers >= 1 and v_count >= 2 and (max_iou > 0.05 or heavy_pairs > 0):
        score += 0.15
        rules_fired.append(
            f"RULE_STRUCTURAL_DEFORMATION: Vehicle aspect ratio distortion ({ar_outliers} outlier(s)) in traffic vicinity"
        )

    # Rule 5: Crowd gathering / Scene distress
    if p_count >= 4 and v_count >= 1 and (max_iou > 0.05 or v_p_overlap > 0):
        score += 0.10
        rules_fired.append(
            f"RULE_INCIDENT_CROWD_GATHERING: High bystander/casualty density ({p_count} persons) near scene"
        )

    # Cap score strictly to [0.0, 1.0]
    final_score = round(min(1.0, max(0.0, score)), 4)

    # Classify severity based on rule-based thresholds
    if final_score >= 0.70:
        severity = "critical"
    elif final_score >= 0.45:
        severity = "moderate"
    elif final_score >= 0.20:
        severity = "minor"
    else:
        severity = "none"

    is_accident = final_score >= 0.45

    return {
        "method": "RULE_BASED_HEURISTIC",
        "score": final_score,
        "rules_fired": rules_fired,
        "is_accident": is_accident,
        "severity": severity,
    }
