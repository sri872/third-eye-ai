"""Conservative four-tier accident evidence fusion.

This module is the single decision layer for ``EvidenceAssessment``. It never
runs a model itself: it combines already-computed classifier output, detector
output, derived geometry, and (only when genuinely produced) sampled-frame
temporal evidence into one stable contract consumed by Accident Check, the
incident report and Dispatch.

Decision rules (documented, deterministic and intentionally conservative):

* Bounding-box overlap, overlap ratio, overlap counts, vehicle counts, aspect
  ratio outliers and person-vehicle proximity are *one correlated geometric
  observation*. They are never independent collision confirmations and never
  establish a physical collision on their own.
* An uncalibrated classifier score is diagnostic output. A high score without
  independent supporting evidence cannot produce ``ACCIDENT EVIDENCE DETECTED``.
* Still photographs are outside the dashcam-video training domain: they are
  limited to ``INCONCLUSIVE`` or ``POSSIBLE ACCIDENT`` and their scene status is
  always ``NOT DETERMINED (out-of-domain still image)``.
* Repeated high predictions on adjacent sampled video frames are one correlated
  temporal observation, not independent confirmations. No impact, trajectory
  change, debris or damage is inferred unless an implemented tracker supplies
  it (no such tracker is available in this project).
* ``ACCIDENT EVIDENCE DETECTED`` is only emitted for an explicit human-verified
  impact record, or for a validated temporal impact sequence produced by an
  implemented detector. Human review remains mandatory in every case.
* ``NO CLEAR ACCIDENT EVIDENCE DETECTED`` only states that this sample contained
  insufficient positive evidence. It never means the road is safe or no accident
  occurred.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from core.schemas import (
    STILL_IMAGE_SCENE_STATUS,
    EvidenceAssessment,
)

STILL_DOMAIN_NOTE = (
    "OUT-OF-DOMAIN: classifier trained on forward-facing dashcam video frames; "
    "an arbitrary still image is not a supported input domain."
)
VIDEO_DOMAIN_NOTE = (
    "IN-DOMAIN (video frames), but only evenly sampled frames were classified; "
    "no full-clip inference, no validated impact tracker and no calibrated score."
)
UNKNOWN_DOMAIN_NOTE = "NOT DETERMINED: media type unavailable."

SOURCE_MODEL = "MODEL (PREDICTION)"
SOURCE_VISION = "REAL (VISION)"
SOURCE_GEOMETRY = "GEOMETRIC SPATIAL OBSERVATION"
SOURCE_TEMPORAL = "TEMPORAL (SAMPLED FRAMES)"
SOURCE_HUMAN = "HUMAN-VERIFIED EVIDENCE"
SOURCE_SIMULATED = "SIMULATED (CALLER)"

# Only these explicit, documented methods may supply machine-detected impact
# evidence. No such tracker exists in this project yet, so the list is empty by
# design: machine-only confirmation is therefore impossible.
IMPLEMENTED_IMPACT_METHODS: tuple[str, ...] = ()


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _feature(features: Optional[Mapping[str, Any]], key: str, default: float = 0.0) -> float:
    value = _as_float((features or {}).get(key))
    return default if value is None else value


def evaluate_temporal_impact(
    temporal: Optional[Mapping[str, Any]],
    features: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return validated temporal impact evidence, or None when unavailable.

    A sustained run of high classifier scores is *not* an impact sequence. An
    impact record must name an implemented detection method, cite at least two
    distinct frame references, and must already have been produced by that
    method upstream. Because this project has no tracker or impact detector,
    no caller can legitimately construct such a record; the function exists so
    the rule is explicit and testable when a real detector is added.
    """
    if not isinstance(temporal, Mapping):
        return None
    record = temporal.get("validated_impact_sequence")
    if not isinstance(record, Mapping):
        return None
    method = str(record.get("method") or "").strip()
    frames = record.get("frame_references")
    if method not in IMPLEMENTED_IMPACT_METHODS:
        return None
    if not isinstance(frames, (list, tuple)) or len(frames) < 2:
        return None
    if len({str(frame) for frame in frames}) < 2:
        return None
    if record.get("impact_detected") is not True:
        return None
    return dict(record)


def _human_impact_record(human_verified_impact: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Accept only an explicit human verification record, never a bare boolean."""
    if not isinstance(human_verified_impact, Mapping):
        return None
    if human_verified_impact.get("impact_verified") is not True:
        return None
    if not str(human_verified_impact.get("verified_by") or "").strip():
        return None
    if not str(human_verified_impact.get("description") or "").strip():
        return None
    return dict(human_verified_impact)


def build_accident_assessment(
    *,
    model_verdict: Optional[str],
    model_score: Optional[float],
    media_type: str,
    features: Optional[Mapping[str, Any]] = None,
    temporal: Optional[Mapping[str, Any]] = None,
    human_verified_impact: Optional[Mapping[str, Any]] = None,
    caller_report: Optional[Mapping[str, Any]] = None,
) -> EvidenceAssessment:
    """Fuse available evidence into the shared four-tier assessment.

    Args:
        model_verdict: Raw classifier verdict, copied verbatim.
        model_score: Raw uncalibrated classifier score, copied verbatim.
        media_type: ``image`` or ``video`` (anything else is unknown).
        features: Geometric features from ``extract_accident_features``.
        temporal: Sampled-frame summary from ``analyze_frame_scores``; may also
            contain a machine-validated impact sequence produced by an implemented
            tracker (none exist in this project yet).
        human_verified_impact: Explicit external human verification record. It
            is a different evidence type from a machine prediction.
        caller_report: Optional simulated caller evidence; kept separate and
            never counted as real vision confirmation.
    """
    media = str(media_type or "").strip().lower()
    if media not in {"image", "video"}:
        media = "unknown"

    score = _as_float(model_score)
    score_text = f"{score:.4f}" if score is not None else "not determined"
    verdict_text = str(model_verdict or "not determined")

    evidence: List[str] = []
    contradictions: List[str] = []
    sources: List[str] = [SOURCE_MODEL]

    evidence.append(
        f"{SOURCE_MODEL}: raw verdict '{verdict_text}', uncalibrated score {score_text}. "
        "Diagnostic only; not calibrated confidence and not collision proof."
    )

    v_count = int(_feature(features, "vehicle_count"))
    p_count = int(_feature(features, "person_count"))
    max_iou = _feature(features, "max_vehicle_iou")
    heavy_pairs = int(_feature(features, "heavily_overlapping_pairs"))
    max_overlap_min = _feature(features, "max_overlap_min")
    person_overlaps = int(_feature(features, "vehicle_person_overlap_count"))

    evidence.append(
        f"{SOURCE_VISION}: {v_count} detected vehicle box(es) and {p_count} detected "
        "person box(es); counts and boxes are detector output, not identities or injuries."
    )
    sources.append(SOURCE_VISION)

    if max_iou > 0.0 or heavy_pairs > 0 or person_overlaps > 0:
        # All geometry measurements describe the same underlying boxes: they are
        # recorded once as a single correlated observation.
        evidence.append(
            f"{SOURCE_GEOMETRY}: max vehicle box IoU {max_iou:.2f}, overlap of smaller "
            f"box {max_overlap_min:.2f}, {heavy_pairs} heavily overlapping pair(s), "
            f"{person_overlaps} person-box/vehicle-box intersection(s). "
            "Perspective, occlusion, close traffic and duplicate boxes can produce "
            "the same pattern; this is not proof of physical contact."
        )
        sources.append(SOURCE_GEOMETRY)
        visual_cue = True
    else:
        evidence.append(
            f"{SOURCE_GEOMETRY}: no substantial vehicle-box overlap or person-box "
            "intersection in the supplied detections."
        )
        sources.append(SOURCE_GEOMETRY)
        visual_cue = False

    if person_overlaps > 0:
        contradictions.append(
            "Person-vehicle box intersection is proximity only; it does not establish "
            "an impact, injury or casualty."
        )

    temporal_available = False
    temporal_signal = "NOT DETERMINED"
    high_frames = 0
    sustained_high = 0
    if isinstance(temporal, Mapping) and media == "video":
        temporal_available = True
        temporal_signal = str(temporal.get("temporal_signal") or "NOT DETERMINED")
        high_frames = int(_feature(temporal, "high_frame_count"))
        sustained_high = int(_feature(temporal, "sustained_high_run"))
        evidence.append(
            f"{SOURCE_TEMPORAL}: sampled-frame signal '{temporal_signal}'; "
            f"{high_frames} high-score frame(s), longest consecutive high run "
            f"{sustained_high}. Adjacent-frame predictions are correlated, not "
            "independent confirmations."
        )
        sources.append(SOURCE_TEMPORAL)
        if high_frames >= 1 and score is not None and score >= 0.60:
            contradictions.append(
                "High scores on neighbouring sampled frames may be the same event "
                "measured repeatedly; no independent impact event was detected."
            )

    impact = evaluate_temporal_impact(temporal, features)
    human_record = _human_impact_record(human_verified_impact)
    if impact is not None:
        evidence.append(
            f"{SOURCE_TEMPORAL}: validated impact sequence from method "
            f"'{impact.get('method')}' on {len(impact.get('frame_references', []))} distinct "
            "frame references."
        )
    if human_record is not None:
        evidence.append(
            f"{SOURCE_HUMAN}: impact verified by '{human_record.get('verified_by')}': "
            f"{human_record.get('description')}. Human verification is itself a distinct "
            "evidence type and remains subject to review."
        )
        sources.append(SOURCE_HUMAN)

    if isinstance(caller_report, Mapping) and caller_report:
        evidence.append(
            f"{SOURCE_SIMULATED}: caller-reported information is preserved separately "
            "and never counted as real vision evidence or confirmation."
        )
        sources.append(SOURCE_SIMULATED)

    high_score = score is not None and score >= 0.60
    low_score = score is not None and score < 0.40
    frame_min = _as_float(temporal.get("min_score")) if isinstance(temporal, Mapping) else None
    frame_max = _as_float(temporal.get("max_score")) if isinstance(temporal, Mapping) else None
    conflicting_frames = (
        frame_min is not None and frame_max is not None and frame_min < 0.40 and frame_max >= 0.60
    )
    if high_score and not visual_cue and impact is None and human_record is None:
        contradictions.append(
            "The classifier score is high but there is no independent visual or "
            "temporal impact evidence in this sample."
        )
    if low_score and visual_cue:
        contradictions.append(
            "Geometric overlap is present while the classifier score is low; these "
            "signals conflict and the overlap is not collision proof."
        )
    if conflicting_frames:
        contradictions.append(
            "Conflicting classifier signals across sampled frames: at least one "
            "frame scored below 0.40 and one scored at or above 0.60."
        )

    if media == "image":
        category = "POSSIBLE ACCIDENT" if high_score and visual_cue else "INCONCLUSIVE"
        scene_status = STILL_IMAGE_SCENE_STATUS
        domain = STILL_DOMAIN_NOTE
        uncertainty = "VERY HIGH"
        reasoning = (
            "A still photograph cannot supply temporal impact evidence. The raw "
            "classifier score and any box overlap are retained as diagnostic cues "
            "only; a possible accident-related cue is not a confirmed collision, "
            "and absence of a cue is not a road-safety conclusion. Human review required."
        )
        contradictions.append(
            "No temporal impact evidence is available from a single still image; "
            "normal traffic and road safety cannot be concluded."
        )
    elif human_record is not None or impact is not None:
        category = "ACCIDENT EVIDENCE DETECTED"
        scene_status = "HUMAN-VERIFIED IMPACT" if human_record is not None else "VALIDATED TEMPORAL IMPACT"
        domain = VIDEO_DOMAIN_NOTE
        uncertainty = "LOW (evidence-specific, not a calibrated probability)"
        reasoning = (
            "A distinct, explicitly verified impact record exists. The raw classifier "
            "score is preserved separately and is not used as confirmation. Human "
            "review remains required before operational action."
        )
    elif media == "video":
        if sustained_high >= 2 and visual_cue:
            category = "POSSIBLE ACCIDENT"
        elif high_score and visual_cue:
            category = "POSSIBLE ACCIDENT"
        elif low_score and not visual_cue and high_frames == 0:
            category = "NO CLEAR ACCIDENT EVIDENCE DETECTED"
        else:
            category = "INCONCLUSIVE"
        scene_status = "VIDEO SAMPLE REVIEWED (NOT DETERMINED)"
        domain = VIDEO_DOMAIN_NOTE
        uncertainty = "HIGH"
        reasoning = (
            "Only sampled frames were classified. A possible cue from correlated "
            "classifier and geometry observations is not impact proof; the absence "
            "of positive evidence in this sample is not a claim that no accident "
            "happened or that the road is safe. Human review required."
        )
        contradictions.append(
            "No validated temporal impact sequence was produced; no impact, "
            "trajectory change, debris or damage is inferred."
        )
    else:
        category = "INCONCLUSIVE"
        scene_status = "NOT DETERMINED (media type unavailable)"
        domain = UNKNOWN_DOMAIN_NOTE
        uncertainty = "VERY HIGH"
        reasoning = "Media type is unknown, so no media-specific conclusion is available."

    if not contradictions:
        contradictions.append("No contradictory evidence was recorded in the supplied fields.")

    return EvidenceAssessment(
        assessment_category=category,
        model_verdict=verdict_text,
        model_score=score,
        media_type=media,
        domain_compatibility=domain,
        scene_status=scene_status,
        evidence=evidence,
        contradictory_evidence=contradictions,
        evidence_sources=sources,
        uncertainty=uncertainty,
        human_review_required=True,
        reasoning=reasoning,
        temporal_evidence_available=temporal_available,
        data_source="REAL",
    )
