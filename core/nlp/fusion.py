"""Multimodal evidence fusion module for Project Third-Eye (Phase 4).

Merges:
1. Real vision analysis: Faster R-CNN detections, ResNet-18 classifier verdict, EasyOCR plates, OccupantReports.
2. Simulated emergency caller text: extracted locations, vehicle counts, injury keywords, hazard keywords.

Rules:
- Never silently overwrites or chooses one source over another.
- Tags every field with its exact source: VISION, CALLER_SIMULATED, or MISSING.
- Explicitly flags discrepancies (e.g. caller reports 2 vehicles, vision detects 6) and shows both.
- Generates a human-readable, auditable fused incident summary.
- Persists IncidentReport records to SQLite database.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import List, Literal, Optional
import uuid

from core.nlp.incident_text import generate_incident_summary
from core.schemas import AccidentEvent, CallerExtraction, IncidentReport, OccupantReport, VehicleReport
from db.database import save_incident_report

logger = logging.getLogger("third_eye.nlp.fusion")

# Placeholders are metadata about the pipeline, never a location seen in an image.
_PLACEHOLDER_LOCATIONS = {"monitored roadway", "not determined", "unknown", "none"}


def _is_placeholder_location(value: Optional[str]) -> bool:
    """Return True when a string is not a usable source location value."""
    if value is None:
        return True
    cleaned = str(value).strip()
    return (
        not cleaned
        or cleaned.casefold() in _PLACEHOLDER_LOCATIONS
        or "uploaded media" in cleaned.casefold()
    )


def _normalise_location(value: Optional[str]) -> str:
    """Compare locations without silently rewriting the stored value."""
    return " ".join(str(value or "").split()).casefold()


def fuse_incident_evidence(
    accident_event: AccidentEvent,
    vehicle_reports: Optional[List[VehicleReport]] = None,
    occupant_reports: Optional[List[OccupantReport]] = None,
    caller_extraction: Optional[CallerExtraction] = None,
    image_name: Optional[str] = None,
    save_to_db: bool = True,
    db_path: Optional[Path | str] = None,
    media_type: str = "video",
    image_location: Optional[str] = None,
    image_location_source: Literal["VISION", "SIMULATED", "MISSING"] = "VISION",
    video_evidence: Optional[dict] = None,
) -> IncidentReport:
    """Fuse vision analysis and emergency call extraction into an IncidentReport.

    Args:
        accident_event: Visual incident event from Phase 3b.
        vehicle_reports: Vehicle reports from Phase 2.
        occupant_reports: Occupant reports from Phase 4.
        caller_extraction: Structured caller extraction from Phase 4.
        image_name: Optional image file name.
        save_to_db: If True, saves to SQLite incident_reports table.
        db_path: Optional SQLite database path.
        media_type: "image" or "video"; forwarded to the incident narrative so an
            out-of-domain still is never described as confirmed normal traffic.
        image_location: Location associated with the uploaded media by the caller
            of this function (e.g. the SIMULATED demo selection used for hospital
            lookup). It is stored separately from the caller's own location.
        image_location_source: Provenance of ``image_location``: VISION,
            SIMULATED or MISSING. Never inferred from image pixels.
        video_evidence: Optional Phase 14 real sampled-frame/temporal evidence.
            It is retained verbatim and is explanatory only; it never overrides
            the authoritative vision verdict.

    Returns:
        IncidentReport schema instance.
    """
    v_reports = vehicle_reports or []
    o_reports = occupant_reports or []

    report_id = f"inc-{uuid.uuid4().hex[:8]}"
    now_utc = datetime.now(timezone.utc)
    img_label = image_name or accident_event.event_id

    # 1. Location separation. Keep every original source value; the legacy
    # location field stays untouched for compatibility with stored reports, but
    # the two distinct locations are now also available as separate fields.
    supplied_image_location = image_location if not _is_placeholder_location(image_location) else None
    event_location = (
        accident_event.location_name
        if not _is_placeholder_location(accident_event.location_name)
        else None
    )
    image_loc = supplied_image_location or event_location
    image_loc_source: Literal["VISION", "SIMULATED", "MISSING"] = (
        image_location_source
        if supplied_image_location
        else ("VISION" if event_location else "MISSING")
    )
    caller_loc = caller_extraction.location if caller_extraction and caller_extraction.location else None
    location_mismatch = bool(
        image_loc
        and caller_loc
        and _normalise_location(image_loc) != _normalise_location(caller_loc)
    )

    if caller_extraction and caller_extraction.location:
        location = caller_extraction.location
        location_source = "CALLER_SIMULATED"
    elif accident_event.location_name and accident_event.location_name != "Monitored Roadway":
        location = accident_event.location_name
        location_source = "VISION"
    else:
        location = caller_extraction.location if caller_extraction else None
        location_source = "CALLER_SIMULATED" if location else "MISSING"

    # 2. Vehicle count & discrepancy check
    vision_vc = len(v_reports) if v_reports else accident_event.vehicles_involved
    caller_vc = caller_extraction.vehicle_count if caller_extraction else None
    vc_disagreement = False
    disagreements: List[str] = []

    if caller_vc is not None and vision_vc is not None:
        if caller_vc != vision_vc:
            vc_disagreement = True
            disagreements.append(
                f"VEHICLE COUNT DISAGREEMENT: Caller reported {caller_vc} vehicle(s), "
                f"while Vision detected {vision_vc} vehicle(s)."
            )

    # 3. People count & discrepancy check
    vision_pc = len(o_reports)
    caller_pc = caller_extraction.people_involved if caller_extraction else None
    pc_disagreement = False

    if caller_pc is not None and vision_pc is not None:
        if caller_pc != vision_pc:
            pc_disagreement = True
            disagreements.append(
                f"PEOPLE COUNT DISAGREEMENT: Caller mentioned {caller_pc} person(s), "
                f"while Vision detected {vision_pc} visible person(s)."
            )

    # 4. Hazards, Injuries, and Urgency
    if caller_extraction and caller_extraction.hazard_keywords:
        hazards = caller_extraction.hazard_keywords
        hazards_source = "CALLER_SIMULATED"
    else:
        hazards = []
        hazards_source = "MISSING"

    if caller_extraction and caller_extraction.injury_keywords:
        injuries = caller_extraction.injury_keywords
        injuries_source = "CALLER_SIMULATED"
    else:
        injuries = []
        injuries_source = "MISSING"

    if caller_extraction and caller_extraction.urgency_keywords:
        urgency = caller_extraction.urgency_keywords[0]
        urgency_source = "CALLER_SIMULATED"
    else:
        urgency = None
        urgency_source = "MISSING"

    # Cross-source conflicts are recorded separately from the count disagreements
    # so the existing `disagreements` list keeps its meaning and consumers.
    source_disagreements: List[str] = []
    score_text = (
        f"{accident_event.score:.4f}" if accident_event.score is not None else "not determined"
    )
    conflict_details: List[str] = []
    # A caller's claim of an accident is itself evidence from a different source;
    # it must not be dropped just because no urgency/injury/hazard keyword was found.
    if caller_extraction and caller_extraction.accident_status == "reported":
        conflict_details.append("SIMULATED caller-reported accident/collision")
    if urgency:
        conflict_details.append(f"SIMULATED caller urgency '{urgency}'")
    if caller_extraction and caller_extraction.injury_keywords:
        conflict_details.append(
            "SIMULATED caller-reported injuries ("
            + ", ".join(caller_extraction.injury_keywords)
            + ")"
        )
    if caller_extraction and caller_extraction.hazard_keywords:
        conflict_details.append(
            "SIMULATED caller-reported hazards ("
            + ", ".join(caller_extraction.hazard_keywords)
            + ")"
        )
    vision_unconfirmed = accident_event.verdict in (
        "NO ACCIDENT SIGNAL",
        "UNCERTAIN",
        "NOT DETERMINED (out-of-domain still image)",
    )
    if vision_unconfirmed and conflict_details:
        source_disagreements.append(
            "SOURCE DISAGREEMENT: SIMULATED CALLER-REPORTED ACCIDENT — REAL VISION "
            "ASSESSMENT DOES NOT CONFIRM THE INCIDENT. "
            f"REAL VISION assessment [{accident_event.verdict}] "
            f"(uncalibrated score {score_text}) conflicts with "
            + "; ".join(conflict_details)
            + ". Dispatcher verification required; SIMULATED caller evidence is "
            "retained separately and does not establish the visual outcome."
        )
    if location_mismatch:
        source_disagreements.append(
            "LOCATION MISMATCH: image-associated location "
            f"'{image_loc}' (Source: {image_loc_source}) differs from caller-reported "
            f"location '{caller_loc}' (Source: CALLER_SIMULATED). Both values are retained "
            "verbatim; no silent merge. Dispatcher verification required."
        )
    if caller_extraction and caller_extraction.contradictions:
        for c in caller_extraction.contradictions:
            source_disagreements.append(f"CALLER CONTRADICTION: {c}")

    # 5. Build comprehensive narrative
    vision_narrative = generate_incident_summary(
        accident_event, v_reports, o_reports, media_type=media_type
    )

    fused_blocks = [
        "==================================================================",
        f"FUSED INCIDENT AUDIT REPORT: {report_id}",
        f"Timestamp: {now_utc.isoformat()}",
        f"Image Reference: {img_label} | Call Reference: {caller_extraction.call_id if caller_extraction else 'None'}",
        "Data Source: SIMULATED (Vision data REAL, paired with SIMULATED emergency call)",
        "==================================================================",
        "\n1. MULTIMODAL EXECUTIVE VERDICT:",
        f"  - REAL VISION ASSESSMENT (uploaded-media result): [{accident_event.verdict}] (Source: VISION, raw uncalibrated score: {score_text})",
        f"  - Severity Level: {accident_event.severity.upper()} (Source: VISION)",
        f"  - Image-associated location: {image_loc or 'Unknown'} (Source: {image_loc_source})",
        f"  - Caller-reported location: {caller_loc or 'Unknown'} (Source: {'CALLER_SIMULATED' if caller_loc else 'MISSING'})",
        f"  - LOCATION MISMATCH: {'YES' if location_mismatch else 'NO'} (both source values retained verbatim)",
        f"  - Reported Location (legacy field): {location or 'Unknown'} (Source: {location_source})",
        f"  - SIMULATED CALLER URGENCY (not a verified priority): {urgency.upper() if urgency else 'NOT SPECIFIED'} (Source: {urgency_source})",
        "  - Dispatcher verification required before acting on any simulated caller claim.",
    ]

    # Disagreements section (count-level; kept verbatim for existing consumers)
    if disagreements:
        fused_blocks.append("\n2. [!] DETECTED CROSS-MODAL DISAGREEMENTS [!]:")
        for d in disagreements:
            fused_blocks.append(f"  - {d}")
        fused_blocks.append("  * Protocol Notice: Both signals are retained. Dispatcher verification required.")
    else:
        fused_blocks.append("\n2. CROSS-MODAL ALIGNMENT:")
        fused_blocks.append("  - No direct contradictions between caller description and camera observations.")

    if source_disagreements:
        fused_blocks.append("\n2b. SOURCE DISAGREEMENT (REAL VISION vs SIMULATED CALLER):")
        for item in source_disagreements:
            fused_blocks.append(f"  - {item}")

    # Simulated caller evidence summary
    if caller_extraction:
        fused_blocks.append("\n3. SIMULATED CALLER EVIDENCE (Source: CALLER_SIMULATED; not verified by the uploaded image):")
        fused_blocks.append(f"  - Raw Caller Transcript: \"{caller_extraction.raw_text}\"")
        fused_blocks.append(f"  - Extracted Vehicles: {caller_extraction.vehicle_count if caller_extraction.vehicle_count is not None else 'not determined'} ({', '.join(caller_extraction.vehicle_types) if caller_extraction.vehicle_types else 'unspecified types'})")
        fused_blocks.append(f"  - Reported Casualties: {caller_extraction.people_involved if caller_extraction.people_involved is not None else 'not determined'}")
        fused_blocks.append(f"  - Caller Accident Status: {caller_extraction.accident_status}")
        fused_blocks.append(f"  - Caller Injury Status: {caller_extraction.injury_status} ({', '.join(injuries) if injuries else 'no specific keywords'})")
        fused_blocks.append(f"  - Ambulance Requested: {'YES' if caller_extraction.ambulance_requested is True else ('NO (explicitly absent)' if caller_extraction.ambulance_requested is False else 'not determined')}")
        fused_blocks.append(f"  - Emergency Assistance Requested: {'YES' if caller_extraction.emergency_assistance_requested is True else ('NO (explicitly absent)' if caller_extraction.emergency_assistance_requested is False else 'not determined')}")
        fused_blocks.append(f"  - Environmental Hazards: {', '.join(hazards) if hazards else 'None reported'}")
        if caller_extraction.contradictions:
            fused_blocks.append(f"  - Caller Contradictions: {'; '.join(caller_extraction.contradictions)}")

    # Real vision evidence summary
    fused_blocks.append("\n4. REAL VISION EVIDENCE (Source: VISION; uploaded media only):")
    fused_blocks.append(vision_narrative)

    video_assessment_text: Optional[str] = None
    video_temporal_signal: Optional[str] = None
    if video_evidence:
        temporal = video_evidence.get("temporal") or {}
        video_temporal_signal = str(temporal.get("temporal_signal", "NOT DETERMINED"))
        video_assessment_text = str(
            video_evidence.get("assessment_text", "Video evidence not determined.")
        )
        metadata = video_evidence.get("metadata") or {}
        frame_rows = video_evidence.get("frame_rows") or []
        fused_blocks.append(
            "\n5. REAL VIDEO FRAME/TEMPORAL EVIDENCE (Source: REAL; sampled frames only):"
        )
        fused_blocks.append(
            f"  - Sampled frames: {len(frame_rows)} of {metadata.get('frame_count', 'not determined')} "
            f"source frames; duration {metadata.get('duration_s', 'not determined')} s; "
            f"FPS {metadata.get('fps', 'not determined')}."
        )
        fused_blocks.append(
            f"  - Classifier score range: {temporal.get('min_score', 'not determined')}–"
            f"{temporal.get('max_score', 'not determined')} "
            f"(mean {temporal.get('mean_score', 'not determined')})."
        )
        fused_blocks.append(
            f"  - Score bands: >=0.60 {temporal.get('high_frame_count', 0)}; "
            f"0.40–0.60 {temporal.get('uncertain_frame_count', 0)}; "
            f"<0.40 {temporal.get('no_signal_frame_count', 0)}."
        )
        fused_blocks.append(
            f"  - Longest consecutive runs: >=0.60 "
            f"{temporal.get('sustained_high_run', 0)} sampled frame(s); >=0.40 "
            f"{temporal.get('sustained_uncertain_or_high_run', 0)} sampled frame(s)."
        )
        fused_blocks.append(f"  - Temporal signal: {video_temporal_signal}.")
        fused_blocks.append(f"  - Assessment: {video_assessment_text}")
        fused_blocks.append(
            "  - Limitation: evenly sampled frames only; no full-clip inference, no "
            "temporal ground truth, and no accident confirmation. This evidence never "
            "overrides the REAL VISION verdict."
        )

    summary_text = "\n".join(fused_blocks)

    report = IncidentReport(
        report_id=report_id,
        image_name=img_label,
        call_id=caller_extraction.call_id if caller_extraction else None,
        timestamp=now_utc,
        verdict=accident_event.verdict or "UNCERTAIN",
        verdict_source="VISION",
        location=location,
        location_source=location_source,
        vision_vehicle_count=vision_vc,
        caller_vehicle_count=caller_vc,
        vehicle_count_disagreement=vc_disagreement,
        vision_people_count=vision_pc,
        caller_people_count=caller_pc,
        people_count_disagreement=pc_disagreement,
        hazards=hazards,
        hazards_source=hazards_source,
        injuries=injuries,
        injuries_source=injuries_source,
        urgency=urgency,
        urgency_source=urgency_source,
        summary_text=summary_text,
        disagreements=disagreements,
        data_source="SIMULATED",
        image_location=image_loc,
        image_location_source=image_loc_source,
        caller_location=caller_loc,
        location_mismatch=location_mismatch,
        vision_score=accident_event.score,
        source_disagreement=bool(source_disagreements),
        source_disagreements=source_disagreements,
        video_evidence=video_evidence,
        video_assessment=video_assessment_text,
        video_temporal_signal=video_temporal_signal,
        injury_status=caller_extraction.injury_status if caller_extraction else "not determined",
        accident_status=caller_extraction.accident_status if caller_extraction else "not determined",
        ambulance_requested=caller_extraction.ambulance_requested if caller_extraction else None,
        emergency_assistance_requested=caller_extraction.emergency_assistance_requested if caller_extraction else None,
        pedestrians_mentioned=caller_extraction.pedestrians_mentioned if caller_extraction else False,
        passengers_mentioned=caller_extraction.passengers_mentioned if caller_extraction else False,
        caller_contradictions=caller_extraction.contradictions if caller_extraction else [],
    )

    if save_to_db:
        save_incident_report(report, db_path=db_path)

    return report
