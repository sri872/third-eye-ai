"""Deterministic template-based incident text summary generator for Project Third-Eye (Phase 4).

Generates human-readable, auditable incident narratives from:
- AccidentEvent (visual classifier verdict, score, severity)
- VehicleReports (vehicle type, color, plate, damage)
- OccupantReports (visible pedestrian count, vehicle proximity)

Rules:
- Hedged wording matching verdict (LIKELY ACCIDENT / UNCERTAIN / NO ACCIDENT SIGNAL).
- A low/no-signal result is insufficient positive evidence, never normal traffic,
  road safety, or proof that no accident occurred.
- Plates mentioned ONLY if readable; tagged 'unverified' when OCR confidence is below 0.70.
- Never invents facts or details. Unknowns stay 'not determined'.
- Explicitly lists data sources and physical limitations.
"""

from typing import List, Optional

from core.schemas import AccidentEvent, OccupantReport, VehicleReport


def generate_incident_summary(
    accident_event: AccidentEvent,
    vehicle_reports: Optional[List[VehicleReport]] = None,
    occupant_reports: Optional[List[OccupantReport]] = None,
    media_type: str = "video",
) -> str:
    """Generate a deterministic, transparent incident summary text.

    Args:
        accident_event: The AccidentEvent record.
        vehicle_reports: List of VehicleReport records.
        occupant_reports: List of OccupantReport records.
        media_type: "image" or "video". A still image is outside the dashcam
            classifier's training domain, so a low score must not be described
            as confirmed normal traffic.

    Returns:
        Formatted multi-paragraph incident text summary.
    """
    v_reports = vehicle_reports or []
    o_reports = occupant_reports or []

    # 1. Headline & Hedged Verdict
    verdict = accident_event.verdict or "UNCERTAIN"
    score_str = f"{accident_event.score:.4f}" if accident_event.score is not None else "N/A"

    if verdict == "LIKELY ACCIDENT":
        verdict_text = (
            f"[INCIDENT ASSESSMENT: LIKELY ACCIDENT] (Severity: {accident_event.severity.upper()})\n"
            f"Visual and model evidence indicates a probable vehicle collision "
            f"(classifier confidence score: {score_str}, method: {accident_event.method}). "
            f"Emergency dispatch recommended."
        )
    elif verdict == "UNCERTAIN":
        verdict_text = (
            f"[INCIDENT ASSESSMENT: UNCERTAIN] (Severity: {accident_event.severity.upper()})\n"
            f"Visual collision cues are inconclusive; anomaly detected but scene geometry "
            f"indicates on-scene or human verification required (classifier score: {score_str})."
        )
    elif media_type == "image":
        # Out-of-domain still: the score is a ranking signal, not proof of safety.
        verdict_text = (
            "[INCIDENT ASSESSMENT: NOT DETERMINED (out-of-domain still image)] "
            f"(Severity: {accident_event.severity.upper()})\n"
            f"Raw classifier verdict (separate diagnostic field): NO ACCIDENT SIGNAL; "
            f"uncalibrated score: {score_str}. This uploaded still image is outside "
            "the model's training domain, so normal traffic cannot be confirmed; "
            "human review is required."
        )
    else:  # NO ACCIDENT SIGNAL on in-domain video frames
        verdict_text = (
            "[INCIDENT ASSESSMENT: NO CLEAR ACCIDENT EVIDENCE DETECTED] "
            f"(Severity: {accident_event.severity.upper()})\n"
            f"Raw classifier verdict (separate diagnostic field): NO ACCIDENT SIGNAL; "
            f"uncalibrated score: {score_str}. Insufficient positive accident evidence "
            "was found in the analyzed frames. This is not a finding of normal traffic "
            "flow, road safety, or no accident; human review remains required."
        )

    # 2. Vehicle Evidence
    vehicle_lines = []
    if v_reports:
        vehicle_lines.append(f"Vehicles Detected ({len(v_reports)} total):")
        for idx, veh in enumerate(v_reports, 1):
            color_desc = f"{veh.color} ({veh.color_share * 100:.1f}% share)" if veh.color else "unknown color"
            type_desc = veh.vehicle_type or "vehicle"

            # Plate formatting
            if veh.license_plate:
                if veh.plate_confidence is not None and veh.plate_confidence < 0.70:
                    plate_desc = f"Plate: {veh.license_plate} (unverified, OCR conf: {veh.plate_confidence:.2f})"
                elif veh.plate_confidence is not None:
                    plate_desc = f"Plate: {veh.license_plate} (OCR conf: {veh.plate_confidence:.2f})"
                else:
                    plate_desc = f"Plate: {veh.license_plate} (unverified)"
            else:
                plate_desc = "Plate: unreadable / none visible"

            vehicle_lines.append(
                f"  - Vehicle #{idx} ({veh.vehicle_id}): {color_desc} {type_desc} | {plate_desc}"
            )
    else:
        vehicle_lines.append(f"Vehicles Detected: {accident_event.vehicles_involved} (details not individually indexed)")

    vehicles_text = "\n".join(vehicle_lines)

    # 3. People & Occupants Assessment
    visible_count = len(o_reports)
    near_count = sum(1 for r in o_reports if r.vehicle_id is not None)

    people_lines = [
        f"Pedestrian & Occupant Triage ({visible_count} visible person(s) detected):",
        f"  - Visible individuals touching/near vehicles: {near_count}",
        f"  - Visible individuals disjoint from vehicles: {visible_count - near_count}",
        "  - Injury / Health State: Not determined (single-angle camera imagery cannot infer medical condition).",
    ]
    people_text = "\n".join(people_lines)

    # 4. Context & Forensic Reasons
    reasons_text = f"Forensic Context: {accident_event.reasons or 'None logged'}"

    # 5. Data Sources & Limitations
    limitations_text = (
        "Evidence Sources & Operational Constraints:\n"
        "  - Data Source: VISION (Real image analysis via Faster R-CNN, ResNet-18, and EasyOCR)\n"
        "  - Limitation: Visible exterior only. Occupants inside vehicles cannot be reliably counted "
        "from a single image. Unknown facts are strictly marked 'not determined'."
    )
    if media_type == "image":
        limitations_text += (
            "\n  - Out-of-domain still image: the classifier was trained on dashcam video "
            "frames; its score is an uncalibrated ranking signal, not proof of an accident "
            "or of normal traffic."
        )

    summary_blocks = [
        verdict_text,
        vehicles_text,
        people_text,
        reasons_text,
        limitations_text,
    ]

    return "\n\n".join(summary_blocks)
