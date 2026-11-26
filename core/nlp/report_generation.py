"""Structured incident report with explicit evidence and location provenance."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def generate_structured_report(
    cv_diagnostics: Optional[Dict[str, Any]] = None,
    cv_candidates: Optional[list] = None,
    plate_ocr_result: Optional[Dict[str, Any]] = None,
    nlp_analysis: Optional[Dict[str, Any]] = None,
    sos_location: Optional[Any] = None,
    hospital_routing_result: Optional[Dict[str, Any]] = None,
    accident_assessment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a source-labelled report; missing evidence stays NOT DETERMINED."""
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    cv_diagnostics = cv_diagnostics or {}
    cv_candidates = cv_candidates or []
    ocr = plate_ocr_result or {}
    nlp = nlp_analysis or {}
    entities = nlp.get("entities") or {}
    assessment = accident_assessment or {}
    sos = (
        sos_location.to_dict()
        if hasattr(sos_location, "to_dict")
        else (sos_location or {})
    )
    routing = hospital_routing_result or {}
    primary = routing.get("primary_recommendation")
    visual_count = len(cv_candidates)
    visual_status = visual_count if cv_diagnostics else "NOT RUN"

    caller_count = entities.get("total_vehicles_reported", 0)
    if caller_count and cv_diagnostics:
        comparison = "MATCH" if caller_count == visual_count else "DIFFERENCE"
        count_comparison = (
            f"{comparison}: {visual_count} geometric CV candidate region(s) versus "
            f"{caller_count} caller-reported vehicle(s). Candidate regions are not confirmed vehicles."
        )
    else:
        count_comparison = (
            "NOT DETERMINED: visual or caller-reported vehicle count is missing."
        )

    lines = [
        "# THIRD EYE AI — INCIDENT AND FORENSIC REPORT",
        f"Generated: {generated_at}",
        "",
        "## 1. Computer Vision evidence [REAL CV] [CLASSICAL HEURISTICS]",
        f"- Geometric vehicle candidate regions: {visual_status}",
        f"- Image dimensions: {cv_diagnostics.get('width', 'NOT DETERMINED')} × {cv_diagnostics.get('height', 'NOT DETERMINED')}",
        f"- Lighting: {cv_diagnostics.get('lighting_condition', 'NOT DETERMINED')}",
        f"- Sharpness diagnostic: {cv_diagnostics.get('sharpness_laplacian_var', 'NOT DETERMINED')}",
        "- Limitation: rectangular candidate proposals are not proof of a vehicle or collision.",
        "",
        "## 2. Number-plate OCR [CLASSICAL OCR]",
        f"- Status: {ocr.get('status', 'NOT RUN')}",
        f"- Raw recognized text: {ocr.get('raw_text') or 'NOT READABLE'}",
        f"- Normalized candidate: {ocr.get('normalized_text') or 'NOT DETERMINED'}",
        f"- Mean template similarity: {ocr.get('mean_similarity', 'NOT DETERMINED')}",
        f"- Format validation: {(ocr.get('validation') or {}).get('notes', 'NOT DETERMINED')}",
        "- Similarity is not a calibrated probability; uncertain characters are not guessed.",
        "",
        "## 3. Caller NLP [CALLER REPORTED]",
        f"- Incident type: {nlp.get('primary_incident_type', 'NOT DETERMINED')}",
        f"- Rule-based urgency: {nlp.get('urgency_level', 'NOT DETERMINED')}",
        f"- Urgency rationale: {nlp.get('urgency_reason', 'NOT DETERMINED')}",
        f"- Vehicles: {entities.get('vehicles') or 'None extracted'}",
        f"- Locations: {entities.get('locations') or 'None extracted'}",
        f"- Injuries: {entities.get('injuries') or 'None extracted'}",
        f"- Hazards: {entities.get('hazards') or 'None extracted'}",
        f"- Transcript contradictions: {nlp.get('contradictions_detected') or 'None detected by current rules'}",
        "",
        "## 4. Accident evidence assessment [DETERMINISTIC HEURISTIC]",
        f"- Category: {assessment.get('assessment_category', 'NOT DETERMINED')}",
        f"- Rationale: {assessment.get('rationale', 'No image-based assessment was performed.')}",
        "- Human review required: YES",
        "",
        "## 5. Incident location [SOS GPS]",
        f"- Coordinates: {sos.get('latitude', 'NOT SUPPLIED')}, {sos.get('longitude', 'NOT SUPPLIED')}",
        f"- Source: {sos.get('source', 'NOT SUPPLIED')}",
        f"- Accuracy: {sos.get('accuracy_meters', 'NOT SUPPLIED')} metres",
        f"- Timestamp: {sos.get('timestamp', 'NOT SUPPLIED')}",
        f"- Stale: {sos.get('is_stale', 'NOT DETERMINED')}",
        "",
        "## 6. Hospital recommendation and route [HOSPITAL ROUTE] [OPENSTREETMAP / NETWORKX]",
    ]

    if primary:
        facility = primary["facility"]
        route = primary["route"]
        emergency_text = (
            "OSM emergency=yes (hours/readiness unverified)"
            if facility.get("has_documented_emergency")
            else "Undocumented in OSM"
        )
        lines.extend([
            f"- Recommended hospital: {facility['name']}",
            f"- Address: {facility['address']}",
            f"- Category: {facility['category']}",
            f"- OSM emergency tag: {emergency_text}",
            f"- Contact: {facility.get('phone') or 'NOT LISTED'}",
            f"- Road distance: {route['distance_km']:.2f} km",
            f"- Static estimated journey time: {route['estimated_time_minutes']:.1f} min",
            f"- Route source: {route.get('route_source', 'NOT DETERMINED')}",
            f"- Ranking explanation: {primary['ranking_reason']}",
            f"- Navigation link: {route['google_maps_url']}",
            "- No live traffic, bed availability, staffing, hospital acceptance or ambulance assignment is confirmed.",
        ])
        if routing.get("alternatives"):
            lines.append(
                "- Alternatives: " + "; ".join(
                    item["facility"]["name"] for item in routing["alternatives"]
                )
            )
    else:
        lines.append(
            f"- No hospital route prepared: {routing.get('error', 'SOS location or a connected road route is unavailable.')}"
        )

    lines.extend([
        "",
        "## 7. Cross-source comparison [COMPUTED COMPARISON]",
        f"- Visual candidate count vs caller-reported count: {count_comparison}",
        "",
        "## 8. Mandatory safety notice",
        "- Human review is required before any emergency action.",
        "- This application does not place 112/108 calls, contact or assign an ambulance, or confirm hospital acceptance.",
        "- Road times are estimates from static OpenStreetMap graph attributes and endpoint connectors; no live traffic data is used.",
        "- Hospital records are from a dated OpenStreetMap snapshot and may be incomplete or outdated.",
    ])

    return {
        "timestamp": generated_at,
        "cv_evidence": {
            "source": "REAL CV (CLASSICAL IMAGE PROCESSING)",
            "candidate_vehicle_count": visual_count,
            "candidates": cv_candidates,
            "diagnostics": cv_diagnostics,
        },
        "ocr_evidence": ocr,
        "nlp_evidence": nlp,
        "sos_location": sos,
        "hospital_routing": routing,
        "accident_assessment": assessment,
        "cross_check": {
            "vehicle_count_comparison": count_comparison,
            "human_review_required": True,
        },
        "markdown_report": "\n".join(lines),
    }
