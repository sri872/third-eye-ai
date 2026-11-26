"""Structured Incident Report Generation.

Fuses classical CV detections, OCR readings, rule-based NLP extractions,
SOS GPS coordinates, and deterministic hospital routing into a unified,
source-separated forensic report.
"""

from typing import Dict, Any, Optional
from datetime import datetime


def generate_structured_report(
    cv_diagnostics: Optional[Dict[str, Any]] = None,
    cv_candidates: Optional[list] = None,
    plate_ocr_result: Optional[Dict[str, Any]] = None,
    nlp_analysis: Optional[Dict[str, Any]] = None,
    sos_location: Optional[Any] = None,
    hospital_routing_result: Optional[Dict[str, Any]] = None,
    accident_assessment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a combined, source-separated incident report.

    Every field is tagged with an explicit provenance label:
    [REAL CV], [CLASSICAL OCR], [CALLER REPORTED], [SOS GPS], [HOSPITAL ROUTE], [COMPUTED COMPARISON].
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. CV Evidence
    cv_boxes = cv_candidates if cv_candidates is not None else []
    cv_summary = {
        "source": "REAL CV (CLASSICAL IMAGE PROCESSING)",
        "candidate_vehicle_count": len(cv_boxes),
        "candidates": cv_boxes,
        "diagnostics": cv_diagnostics or {},
        "notes": "Candidate regions localized via classical contour & edge morphology. Not a learned detector.",
    }

    # 2. OCR Evidence
    ocr_res = plate_ocr_result or {}
    ocr_summary = {
        "source": "CLASSICAL OCR (TEMPLATE MATCHING)",
        "plate_status": ocr_res.get("status", "NOT_PROCESSED"),
        "raw_text": ocr_res.get("raw_text", ""),
        "normalized_plate": ocr_res.get("normalized_text", ""),
        "mean_similarity": ocr_res.get("mean_similarity", 0.0),
        "is_verified": ocr_res.get("is_verified", False),
        "validation_details": ocr_res.get("validation", {}),
    }

    # 3. NLP Evidence
    nlp = nlp_analysis or {}
    entities = nlp.get("entities", {})
    nlp_summary = {
        "source": "CALLER REPORTED (DETERMINISTIC NLP RULES)",
        "reported_vehicles": entities.get("vehicles", []),
        "total_vehicles_reported": entities.get("total_vehicles_reported", 0),
        "reported_locations": entities.get("locations", []),
        "reported_injuries": entities.get("injuries", []),
        "reported_hazards": entities.get("hazards", []),
        "urgency_level": nlp.get("urgency_level", "UNKNOWN"),
        "urgency_reason": nlp.get("urgency_reason", ""),
        "incident_type": nlp.get("primary_incident_type", "UNKNOWN"),
        "contradictions": nlp.get("contradictions_detected", []),
    }

    # 4. SOS Location & Hospital Routing
    sos_dict = sos_location.to_dict() if hasattr(sos_location, "to_dict") else (sos_location or {})
    primary_hosp = hospital_routing_result.get("primary_recommendation") if hospital_routing_result else None

    # 5. Multimodal Accident Evidence Assessment
    assessment = accident_assessment or {}

    # 6. Computed Cross-Verification
    visual_count = len(cv_boxes)
    caller_count = entities.get("total_vehicles_reported", 0)

    if caller_count > 0 and visual_count > 0:
        if visual_count == caller_count:
            count_consistency = f"AGREED: Visual candidate count ({visual_count}) matches caller report ({caller_count})."
        else:
            count_consistency = f"DISCREPANCY: Visual candidate count ({visual_count}) differs from caller report ({caller_count})."
    else:
        count_consistency = "UNVERIFIED: Insufficient visual or caller vehicle count to perform comparison."

    cross_check = {
        "source": "COMPUTED COMPARISON",
        "vehicle_count_comparison": count_consistency,
        "human_review_required": True,
        "human_review_reason": "Automated classical screening requires human operator verification before emergency dispatch.",
    }

    # 7. Build Markdown summary text
    md_lines = [
        "# THIRD EYE AI — STRUCTURED INCIDENT & FORENSIC REPORT",
        f"**Generated**: {timestamp_str} | **Architecture**: Classical CV + Template OCR + Rule NLP + Road Routing",
        "",
        "---",
        "",
        "## 1. Visual Computer Vision Evidence `[REAL CV]`",
        f"- **Candidate Vehicle Regions Localized**: {len(cv_boxes)}",
        f"- **Image Dimensions**: {cv_diagnostics.get('width', 'N/A')}x{cv_diagnostics.get('height', 'N/A')}",
        f"- **Lighting Condition**: {cv_diagnostics.get('lighting_condition', 'N/A')}",
        f"- **Clarity / Sharpness Metric**: {cv_diagnostics.get('clarity_assessment', 'N/A')} (Laplacian: {cv_diagnostics.get('sharpness_laplacian_var', 'N/A')})",
        "",
        "## 2. Number Plate OCR Findings `[CLASSICAL OCR]`",
        f"- **Plate Verification Status**: `{ocr_summary['plate_status']}`",
        f"- **Raw OCR Candidates**: `{ocr_summary['raw_text'] or 'NONE'}`",
        f"- **Normalized Registration Number**: `{ocr_summary['normalized_plate'] or 'NONE'}`",
        f"- **Template Mean Similarity**: `{ocr_summary['mean_similarity']}` (Threshold: 0.70)",
        f"- **State Jurisdiction**: {ocr_summary['validation_details'].get('state_name', 'Not Determined')}",
        "",
        "## 3. Caller-Reported Incident Analysis `[CALLER REPORTED]`",
        f"- **Primary Incident Classification**: `{nlp_summary['incident_type']}`",
        f"- **Determined Urgency**: `{nlp_summary['urgency_level']}`",
        f"- **Urgency Rationale**: {nlp_summary['urgency_reason']}",
        f"- **Reported Locations**: {', '.join(nlp_summary['reported_locations']) if nlp_summary['reported_locations'] else 'None stated'}",
        f"- **Reported Vehicles**: {len(nlp_summary['reported_vehicles'])} ({nlp_summary['total_vehicles_reported']} total units)",
        f"- **Reported Injuries**: {len(nlp_summary['reported_injuries'])} items ({', '.join(i['keyword'] for i in nlp_summary['reported_injuries']) if nlp_summary['reported_injuries'] else 'None reported'})",
        f"- **Reported Hazards**: {len(nlp_summary['reported_hazards'])} items ({', '.join(h['keyword'] for h in nlp_summary['reported_hazards']) if nlp_summary['reported_hazards'] else 'None reported'})",
        "",
        "## 4. Accident Evidence Assessment `[DETERMINISTIC FUSION]`",
        f"- **Assessment Category**: `{assessment.get('assessment_category', 'NOT_EVALUATED')}`",
        f"- **Confidence Level**: `{assessment.get('confidence_level', 'N/A')}`",
        f"- **Assessment Rationale**: {assessment.get('rationale', 'N/A')}",
        f"- **Maximum Candidate Box IoU**: {assessment.get('max_vehicle_iou', 0.0)}",
        "",
        "## 5. SOS Incident Location `[SOS GPS]`",
        f"- **Incident Coordinates**: {sos_dict.get('latitude', 'N/A')}°N, {sos_dict.get('longitude', 'N/A')}°E",
        f"- **Location Source**: `{sos_dict.get('source', 'UNSPECIFIED')}`",
        f"- **GPS Accuracy**: ±{sos_dict.get('accuracy_meters', 'N/A')} meters",
        f"- **Location Timestamp**: {sos_dict.get('timestamp', 'N/A')}",
        "",
        "## 6. Nearest Suitable Hospital & Emergency Route `[HOSPITAL ROUTE]`",
    ]

    if primary_hosp:
        fac = primary_hosp["facility"]
        route = primary_hosp["route"]
        md_lines.extend([
            f"- **Primary Recommended Facility**: **{fac['name']}**",
            f"- **Facility Category**: {fac['category'].title()} | Emergency Tag: {'YES (24/7)' if fac['has_documented_emergency'] else 'Undocumented'}",
            f"- **Road Network Distance**: **{route['distance_km']} km** (Est. travel time: ~{route['estimated_time_minutes']} min)",
            f"- **Facility Address**: {fac['address']}",
            f"- **Contact Number**: {fac.get('phone') or 'Not listed in OpenStreetMap cache'}",
            f"- **Ranking Rationale**: {primary_hosp['ranking_reason']}",
            f"- **Navigation Route**: [Google Maps Directions]({route['google_maps_url']})",
        ])
    else:
        md_lines.append("- *No hospital route computed (requires validated SOS coordinates).*")

    md_lines.extend([
        "",
        "## 7. Multi-Source Cross-Verification `[COMPUTED COMPARISON]`",
        f"- **Vehicle Count Comparison**: {count_consistency}",
        f"- **Contradictions in Transcript**: {' | '.join(nlp_summary['contradictions']) if nlp_summary['contradictions'] else 'No internal contradictions detected'}",
        "",
        "## 8. Mandatory Verification & Limitations",
        "- **Human Review Required**: YES (Mandatory before emergency action)",
        "- **Prototype Safety Notice**: Ambulance routing and hospital recommendations are simulated for evaluation. The system does NOT place real 112/108 emergency calls or confirm bed availability.",
    ])

    return {
        "timestamp": timestamp_str,
        "cv_evidence": cv_summary,
        "ocr_evidence": ocr_summary,
        "nlp_evidence": nlp_summary,
        "sos_location": sos_dict,
        "hospital_routing": hospital_routing_result,
        "accident_assessment": assessment,
        "cross_check": cross_check,
        "markdown_report": "\n".join(md_lines),
    }
