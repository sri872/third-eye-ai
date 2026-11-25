"""Streamlit-free helpers for Tab 4 (Incident report).

The report is presented as source-separated sections: REAL evidence from the
uploaded media is never mixed with SIMULATED evidence from a sample call. Every
displayed value carries a source label in its own table column, and all original
report fields are retained in a final inventory section.

This module runs no models, imports no Streamlit symbols, writes no files and
does not touch the database. It only formats values already present in a report.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from core.ocr.plate_reader import PLATE_TRUST_THRESHOLD

NOT_DETERMINED = "not determined"
NOT_READABLE = f"{NOT_DETERMINED} (unreadable or unverified)"

SOURCE_REAL_VISION = "REAL (VISION)"
SOURCE_SIM_CALLER = "SIMULATED (CALLER)"
SOURCE_SIM_DEMO = "SIMULATED (DEMO)"
SOURCE_MISSING = "MISSING"
SOURCE_COMPUTED = "COMPUTED (COMPARISON)"
SOURCE_PROVENANCE = "PROVENANCE METADATA"
SOURCE_RULE = "OPERATIONAL RULE"
SOURCE_MODEL = "MODEL (PREDICTION)"
SOURCE_OCR = "OCR (CANDIDATE)"
SOURCE_UNKNOWN = "UNKNOWN"

# OCR candidates are never registrations: verification uses the one 0.70
# threshold from core.ocr.plate_reader, and format validity is not verification.
PLATE_UNVERIFIED = "PLATE IDENTIFICATION: UNVERIFIED"
PLATE_VERIFIED = "PLATE IDENTIFICATION: VERIFIED"

MAX_DISPLAY_LENGTH = 120

_DISAGREEMENT_KEYS = {
    "vehicle_count_disagreement",
    "people_count_disagreement",
    "location_mismatch",
    "source_disagreement",
}

_CALLER_DATA_KEYS = (
    "caller_location",
    "caller_vehicle_count",
    "caller_people_count",
    "urgency",
    "injuries",
    "hazards",
    "caller_contradictions",
)


@dataclass(frozen=True)
class ReportSection:
    """One readable table in the incident report tab."""

    heading: str
    rows: tuple[dict[str, str], ...]

    def as_table(self) -> list[dict[str, str]]:
        """Return plain dictionaries suitable for st.table / DataFrame tests."""
        return [dict(row) for row in self.rows]


def display_value(value: Any, limit: int | None = None) -> str:
    """Render a stored value without inventing a replacement.

    Every returned string fits within ``MAX_DISPLAY_LENGTH``. The truncation
    marker is included in that budget so the suffix never pushes a cell over
    the width limit; the original text length is still reported verbatim.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return NOT_DETERMINED
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, (list, tuple)):
        text = ", ".join(display_value(item) for item in value) if value else "none recorded"
    elif isinstance(value, float):
        text = f"{value:.4f}"
    else:
        text = str(value)

    cap = MAX_DISPLAY_LENGTH if limit is None else min(int(limit), MAX_DISPLAY_LENGTH)
    if cap <= 0:
        return ""
    if len(text) <= cap:
        return text
    suffix = f" … (full text below, {len(text)} chars)"
    if len(suffix) >= cap:
        return text[:cap]
    return text[: cap - len(suffix)].rstrip() + suffix


def flag(value: Any) -> str:
    """Render a disagreement flag in the same wording as the previous report UI."""
    return "FLAG" if bool(value) else "NO FLAG"


def _row(
    field: str, value: Any, source: str, as_flag: bool = False, limit: int | None = None
) -> dict[str, str]:
    return {
        "Field": display_value(field),
        "Value": flag(value) if as_flag else display_value(value, limit=limit),
        "Source": display_value(source),
    }


def _has_value(value: Any) -> bool:
    """Return True for a stored value that is actually present."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _caller_data_present(
    report: dict[str, Any], keys: tuple[str, ...] = _CALLER_DATA_KEYS
) -> bool:
    """A call reference alone is not enough; require at least one caller value."""
    if _call_present(report):
        return True
    return any(_has_value(report.get(key)) for key in keys)


def format_comparison_state(
    disagreement: Any = None,
    caller_present: bool | None = None,
    *,
    caller_data_present: bool | None = None,
    report: dict[str, Any] | None = None,
    kind: str = "mismatch",
) -> str:
    """Describe a cross-source comparison without inventing a caller value.

    A missing caller side means there is nothing to compare. When both sides are
    present the caller-facing wording distinguishes location mismatches from
    count/source disagreements.
    """
    present = caller_present
    if present is None:
        present = caller_data_present
    if present is None:
        present = _caller_data_present(report) if report is not None else True
    if not present:
        return "NOT COMPARABLE — CALLER DATA MISSING"

    if kind.strip().casefold() in {"disagreement", "disagreements", "count", "counts", "source"}:
        return "DISAGREEMENT" if bool(disagreement) else "NO DISAGREEMENT"
    return "MISMATCH" if bool(disagreement) else "NO MISMATCH"


def _comparison_row(
    field: str,
    value: Any,
    has_call: bool,
    kind: str = "mismatch",
    source: str = SOURCE_COMPUTED,
) -> dict[str, str]:
    """Keep the legacy FLAG wording for paired calls; explain missing callers."""
    if has_call:
        return _row(field, flag(value), source)
    return _row(field, format_comparison_state(value, caller_present=False, kind=kind), source)


def _call_present(report: dict[str, Any]) -> bool:
    return bool(report.get("call_id"))


def _image_source(report: dict[str, Any]) -> str:
    source = str(report.get("image_location_source") or "MISSING").upper()
    if source == "SIMULATED":
        return SOURCE_SIM_DEMO
    if source == "VISION":
        return SOURCE_REAL_VISION
    return SOURCE_MISSING


def _source_for_key(key: str, report: dict[str, Any]) -> str:
    """Best-effort provenance label for the raw field inventory."""
    if key in _DISAGREEMENT_KEYS or key in {"disagreements", "source_disagreements", "summary_text"}:
        return SOURCE_COMPUTED
    if key.endswith("_source") or key == "data_source":
        return SOURCE_PROVENANCE
    if key in {"verdict", "vision_vehicle_count", "vision_people_count", "vision_score"}:
        return SOURCE_REAL_VISION
    if key == "evidence_assessment":
        return SOURCE_REAL_VISION if _has_value(report.get(key)) else SOURCE_MISSING
    if key in {"image_location", "image_location_source"}:
        return _image_source(report)
    if key in {"call_id", "caller_location", "caller_vehicle_count", "caller_people_count",
               "hazards", "hazards_source", "injuries", "injuries_source",
               "urgency", "urgency_source"}:
        return SOURCE_SIM_CALLER if _call_present(report) else SOURCE_MISSING
    if key == "location":
        return SOURCE_SIM_CALLER if report.get("location_source") == "CALLER_SIMULATED" else SOURCE_REAL_VISION
    if key in {"report_id", "timestamp"}:
        return SOURCE_PROVENANCE
    return SOURCE_COMPUTED


_MISSING_ASSESSMENT_FIELDS = (
    "Assessment category",
    "Model verdict",
    "Uncalibrated model score",
    "Media type",
    "Domain compatibility",
    "Scene status",
    "Evidence",
    "Contradictory evidence",
    "Evidence sources",
    "Uncertainty",
    "Human review required",
    "Reasoning",
    "Temporal evidence available",
    "Data source",
)


def _assessment_evidence_source(assessment: dict[str, Any]) -> str:
    """Use the supplied evidence_sources verbatim; fall back to REAL vision."""
    sources = assessment.get("evidence_sources")
    if isinstance(sources, (list, tuple)):
        labels = [str(item).strip() for item in sources if _has_value(item)]
    elif isinstance(sources, str) and sources.strip():
        labels = [sources.strip()]
    else:
        labels = []
    return ", ".join(labels) if labels else SOURCE_REAL_VISION


def _items(value: Any) -> list[Any]:
    """Treat a supplied list as a list without turning missing data into data."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if _has_value(value):
        return [value]
    return []


def _assessment_section(report: dict[str, Any]) -> ReportSection:
    """Display the supplied four-tier assessment verbatim, never re-tier it."""
    heading = "8. Four-tier accident evidence assessment"
    assessment = report.get("evidence_assessment")
    if not isinstance(assessment, dict) or not assessment:
        return ReportSection(
            heading,
            tuple(_row(field, NOT_DETERMINED, SOURCE_PROVENANCE) for field in _MISSING_ASSESSMENT_FIELDS),
        )

    evidence_source = _assessment_evidence_source(assessment)
    evidence_rows = tuple(
        _row("Evidence", item, evidence_source, limit=96)
        for item in _items(assessment.get("evidence"))
    ) or (_row("Evidence", "none recorded", evidence_source),)
    contradictory_rows = tuple(
        _row("Contradictory evidence", item, evidence_source, limit=96)
        for item in _items(assessment.get("contradictory_evidence"))
    ) or (_row("Contradictory evidence", "none recorded", evidence_source),)

    rows = [
        _row("Assessment category", assessment.get("assessment_category"), SOURCE_PROVENANCE),
        _row("Model verdict", assessment.get("model_verdict"), SOURCE_MODEL),
        _row("Uncalibrated model score", assessment.get("model_score"), SOURCE_MODEL),
        _row("Media type", assessment.get("media_type"), SOURCE_REAL_VISION),
        _row("Domain compatibility", assessment.get("domain_compatibility"), SOURCE_REAL_VISION),
        _row("Scene status", assessment.get("scene_status"), SOURCE_REAL_VISION),
        *evidence_rows,
        *contradictory_rows,
        _row("Evidence sources", assessment.get("evidence_sources"), SOURCE_PROVENANCE),
        _row("Uncertainty", assessment.get("uncertainty"), SOURCE_PROVENANCE),
        _row("Human review required", "MANDATORY", SOURCE_RULE),
        _row("Reasoning", assessment.get("reasoning"), SOURCE_PROVENANCE, limit=96),
        _row(
            "Temporal evidence available",
            assessment.get("temporal_evidence_available"),
            SOURCE_REAL_VISION,
        ),
        _row("Data source", assessment.get("data_source"), SOURCE_PROVENANCE),
    ]
    return ReportSection(heading, tuple(rows))


def _model_prediction_section(report: dict[str, Any]) -> ReportSection:
    """Separate the raw classifier output from the evidence assessment."""
    return ReportSection(
        "9. Model prediction (raw output, not a determination)",
        (
            _row("Raw model verdict", report.get("verdict"), SOURCE_MODEL),
            _row("Uncalibrated model score", report.get("vision_score"), SOURCE_MODEL),
            _row("Verdict source", report.get("verdict_source"), SOURCE_PROVENANCE),
            _row("Video temporal signal", report.get("video_temporal_signal"), SOURCE_MODEL),
            _row("Video-level assessment", report.get("video_assessment"), SOURCE_MODEL, limit=96),
            _row(
                "Prediction limitation",
                "Raw model output is a prediction, not an accident determination.",
                SOURCE_RULE,
            ),
        ),
    )


_CANDIDATE_KEYS = (
    "raw_ocr_text",
    "raw_text",
    "recognized_text",
    "plate_candidate",
    "ocr_candidate",
    "license_plate",
    "plate",
    "text",
)
_NORMALIZED_KEYS = ("license_plate", "plate", "plate_candidate", "ocr_candidate")
_CONFIDENCE_KEYS = (
    "plate_confidence",
    "ocr_confidence",
    "candidate_confidence",
    "raw_confidence",
    "confidence",
)
_STATUS_KEYS = (
    "ocr_status",
    "plate_detection_status",
    "plate_status",
    "detection_status",
    "status",
)


def _lookup(mapping: dict[str, Any]) -> dict[str, Any]:
    return {str(key).casefold(): value for key, value in mapping.items()}


def _first_value(lookup: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = lookup.get(key)
        if _has_value(value):
            return value
    return None


def _plate_entries(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect plate evidence from report keys that already exist."""
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(entry: dict[str, Any]) -> None:
        lookup = _lookup(entry)
        candidate = _first_value(lookup, _CANDIDATE_KEYS)
        confidence = _first_value(lookup, _CONFIDENCE_KEYS)
        marker = (str(candidate or ""), str(confidence or ""))
        if marker in seen:
            return
        seen.add(marker)
        entries.append(entry)

    flat = {
        key: value
        for key, value in report.items()
        if key != "ocr_evidence"
        and key != "vehicle_reports"
        and any(token in str(key).casefold() for token in ("plate", "ocr"))
    }
    if flat:
        add(flat)

    ocr_evidence = report.get("ocr_evidence")
    if isinstance(ocr_evidence, dict):
        evidence_lookup = _lookup(ocr_evidence)
        if any(
            key in evidence_lookup for key in _CANDIDATE_KEYS + _CONFIDENCE_KEYS + _STATUS_KEYS
        ):
            add(dict(ocr_evidence))
        else:
            for vehicle_id, evidence in ocr_evidence.items():
                if isinstance(evidence, dict):
                    entry = dict(evidence)
                elif hasattr(evidence, "__dict__"):
                    entry = dict(vars(evidence))
                else:
                    entry = {"raw_text": evidence}
                entry.setdefault("vehicle_id", vehicle_id)
                add(entry)
    elif isinstance(ocr_evidence, (list, tuple)):
        for evidence in ocr_evidence:
            if isinstance(evidence, dict):
                add(evidence)

    for item in report.get("vehicle_reports") or []:
        if isinstance(item, dict):
            entry = item
        elif hasattr(item, "__dict__"):
            entry = dict(vars(item))
        else:
            continue
        if any(token in str(key).casefold() for key in entry for token in ("plate", "ocr")):
            add(entry)

    for line in str(report.get("summary_text") or "").splitlines():
        match = re.search(r"Plate:\s*(.*?)(?:\s*\(([^)]*)\))?\s*$", line, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = (match.group(1) or "").strip()
        if not candidate or candidate.casefold() in {
            "unreadable / none visible",
            "unreadable",
            "none",
            "not determined",
        }:
            continue
        detail = match.group(2) or ""
        entry: dict[str, Any] = {"license_plate": candidate, "ocr_status": detail or "OK"}
        conf_match = re.search(r"OCR conf:\s*([0-9]*\.?[0-9]+)", detail, flags=re.IGNORECASE)
        if conf_match:
            entry["plate_confidence"] = float(conf_match.group(1))
        add(entry)

    return entries


def _ocr_section(report: dict[str, Any]) -> ReportSection:
    """Show OCR text as a candidate; only the threshold can verify a plate."""
    heading = "10. OCR and registration evidence (candidate, not verified)"
    entries = _plate_entries(report)
    if not entries:
        return ReportSection(
            heading,
            (
                _row("Plate candidate", NOT_READABLE, SOURCE_OCR),
                _row("Plate identification", NOT_READABLE, SOURCE_OCR),
                _row("OCR confidence", NOT_DETERMINED, SOURCE_OCR),
                _row("Registration source", SOURCE_UNKNOWN, SOURCE_UNKNOWN),
                _row(
                    "Format validity rule",
                    "Format-valid OCR text is a candidate, never a verified registration.",
                    SOURCE_RULE,
                ),
                _row(
                    "Verification rule",
                    f"Candidates below {PLATE_TRUST_THRESHOLD:.2f} OCR confidence are unverified.",
                    SOURCE_RULE,
                ),
            ),
        )

    rows: list[dict[str, str]] = []
    for entry in entries:
        lookup = _lookup(entry)
        candidate = _first_value(lookup, _CANDIDATE_KEYS)
        normalized = _first_value(lookup, _NORMALIZED_KEYS)
        confidence = _first_value(lookup, _CONFIDENCE_KEYS)
        ocr_status = _first_value(lookup, _STATUS_KEYS)
        verified = False
        if isinstance(confidence, str) and confidence.strip():
            try:
                confidence = float(confidence.strip())
            except ValueError:
                pass
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            verified = float(confidence) >= PLATE_TRUST_THRESHOLD
        vehicle_id = entry.get("vehicle_id")
        prefix = f"{display_value(vehicle_id)} — " if _has_value(vehicle_id) else ""
        entry_rows = [
            _row(
                f"{prefix}Plate candidate",
                candidate if _has_value(candidate) else NOT_READABLE,
                SOURCE_OCR,
                limit=96,
            ),
        ]
        if _has_value(normalized) and str(normalized) != str(candidate):
            entry_rows.append(
                _row(
                    f"{prefix}Normalized plate candidate",
                    normalized,
                    SOURCE_OCR,
                    limit=96,
                )
            )
        entry_rows.extend(
            (
                _row(
                    f"{prefix}Plate identification",
                    PLATE_VERIFIED if verified else PLATE_UNVERIFIED,
                    SOURCE_OCR,
                ),
                _row(
                    f"{prefix}OCR confidence",
                    confidence if _has_value(confidence) else NOT_DETERMINED,
                    SOURCE_OCR,
                ),
                _row(
                    f"{prefix}OCR status",
                    ocr_status if _has_value(ocr_status) else NOT_DETERMINED,
                    SOURCE_OCR,
                ),
            )
        )
        rows.extend(entry_rows)
    rows.extend(
        (
            _row("Registration source", SOURCE_UNKNOWN, SOURCE_UNKNOWN),
            _row(
                "Format validity rule",
                "Format-valid OCR text is a candidate, never a verified registration.",
                SOURCE_RULE,
            ),
            _row(
                "Verification rule",
                f"Candidates below {PLATE_TRUST_THRESHOLD:.2f} OCR confidence are unverified.",
                SOURCE_RULE,
            ),
        )
    )
    return ReportSection(heading, tuple(rows))


def build_report_sections(report: dict[str, Any]) -> list[ReportSection]:
    """Build all source-labelled sections for one report dictionary.

    Unknown or absent keys render as ``not determined`` so an older report
    dictionary can still be displayed without fabricating any value.
    """
    has_call = _call_present(report)
    caller_source = SOURCE_SIM_CALLER if has_call else SOURCE_MISSING
    image_source = _image_source(report)
    location_caller_present = _caller_data_present(report, ("caller_location",))
    counts_caller_present = _caller_data_present(
        report, ("caller_vehicle_count", "caller_people_count")
    )
    verdict_caller_present = _caller_data_present(
        report, ("urgency", "caller_location", "caller_vehicle_count", "caller_people_count")
    )

    identity = ReportSection(
        "1. Report identity (source-labelled)",
        (
            _row("Report ID", report.get("report_id"), SOURCE_PROVENANCE),
            _row("Timestamp", report.get("timestamp"), SOURCE_PROVENANCE),
            _row("Media reference", report.get("image_name"), SOURCE_REAL_VISION),
            _row("Call reference", report.get("call_id"), caller_source),
            _row(
                "Pairing",
                "ARBITRARY DEMO PAIRING" if has_call else "VISION ONLY",
                SOURCE_SIM_DEMO if has_call else SOURCE_REAL_VISION,
            ),
            _row("Report data source", report.get("data_source"), SOURCE_PROVENANCE),
        ),
    )

    vision = ReportSection(
        "2. REAL VISION EVIDENCE (uploaded media only)",
        (
            _row("Vision verdict", report.get("verdict"), SOURCE_REAL_VISION),
            _row("Uncalibrated vision score", report.get("vision_score"), SOURCE_REAL_VISION),
            _row("Verdict source", report.get("verdict_source"), SOURCE_PROVENANCE),
            _row("Vision vehicle count (detected)", report.get("vision_vehicle_count"), SOURCE_REAL_VISION),
            _row("Vision people count (visible)", report.get("vision_people_count"), SOURCE_REAL_VISION),
            _row("Image-associated location", report.get("image_location"), image_source),
            _row(
                "Injuries inferred from image",
                NOT_DETERMINED + " (no medical inference from pixels)",
                SOURCE_REAL_VISION,
            ),
            _row("Occupants inside vehicles", NOT_DETERMINED, SOURCE_REAL_VISION),
            _row("Vehicle damage inferred from image", NOT_DETERMINED, SOURCE_REAL_VISION),
        ),
    )

    caller = ReportSection(
        "3. SIMULATED CALLER EVIDENCE (sample call, not verified by image)",
        (
            _row("Call reference", report.get("call_id"), caller_source),
            _row("Caller-reported location", report.get("caller_location"), caller_source),
            _row("Caller vehicle count (reported)", report.get("caller_vehicle_count"), caller_source),
            _row("Caller people count (reported)", report.get("caller_people_count"), caller_source),
            _row("Caller accident status", report.get("accident_status"), caller_source),
            _row("Caller injury status", report.get("injury_status"), caller_source),
            _row("Ambulance requested", report.get("ambulance_requested"), caller_source),
            _row("Emergency assistance requested", report.get("emergency_assistance_requested"), caller_source),
            _row("Pedestrians mentioned", report.get("pedestrians_mentioned"), caller_source),
            _row("Passengers mentioned", report.get("passengers_mentioned"), caller_source),
            _row("Caller-reported injuries", report.get("injuries"), caller_source),
            _row("Caller-reported hazards", report.get("hazards"), caller_source),
            _row("Caller-reported urgency (SIMULATED)", report.get("urgency"), caller_source),
            _row(
                "Emergency priority verified?",
                "NO — human dispatcher confirmation required",
                SOURCE_RULE,
            ),
        ),
    )

    location = ReportSection(
        "4. Location comparison (no silent merge)",
        (
            _row("Image-associated location", report.get("image_location"), image_source),
            _row("Caller-reported location", report.get("caller_location"), caller_source),
            _comparison_row(
                "Location mismatch",
                report.get("location_mismatch"),
                location_caller_present,
                kind="mismatch",
            ),
            _row(
                "Comparison availability",
                format_comparison_state(
                    report.get("location_mismatch"),
                    caller_present=location_caller_present,
                    kind="mismatch",
                ),
                SOURCE_COMPUTED,
            ),
            _row(
                "Legacy location field",
                report.get("location"),
                _source_for_key("location", report),
            ),
            _row("Legacy location_source", report.get("location_source"), SOURCE_PROVENANCE),
            _row(
                "Merge policy",
                "Both original source values retained verbatim; no silent merge.",
                SOURCE_RULE,
            ),
        ),
    )

    counts = ReportSection(
        "5. Count comparison (REAL detections vs SIMULATED report)",
        (
            _row("Vision vehicles", report.get("vision_vehicle_count"), SOURCE_REAL_VISION),
            _row("Caller vehicles", report.get("caller_vehicle_count"), caller_source),
            _comparison_row(
                "Vehicle-count disagreement",
                report.get("vehicle_count_disagreement"),
                counts_caller_present,
                kind="disagreement",
            ),
            _row("Vision people (visible)", report.get("vision_people_count"), SOURCE_REAL_VISION),
            _row("Caller people", report.get("caller_people_count"), caller_source),
            _comparison_row(
                "People-count disagreement",
                report.get("people_count_disagreement"),
                counts_caller_present,
                kind="disagreement",
            ),
            _row(
                "Comparison availability",
                format_comparison_state(
                    report.get("vehicle_count_disagreement"),
                    caller_present=counts_caller_present,
                    kind="disagreement",
                ),
                SOURCE_COMPUTED,
            ),
            _row(
                "Vehicle-count comparison availability",
                format_comparison_state(
                    report.get("vehicle_count_disagreement"),
                    caller_present=counts_caller_present,
                    kind="disagreement",
                ),
                SOURCE_COMPUTED,
            ),
            _row(
                "People-count comparison availability",
                format_comparison_state(
                    report.get("people_count_disagreement"),
                    caller_present=counts_caller_present,
                    kind="disagreement",
                ),
                SOURCE_COMPUTED,
            ),
        ),
    )

    verdict = ReportSection(
        "6. Multimodal verdict and triage",
        (
            _row("Vision verdict (independent)", report.get("verdict"), SOURCE_REAL_VISION),
            _row("Caller urgency (independent, SIMULATED)", report.get("urgency"), caller_source),
            _row("Urgency source", report.get("urgency_source"), SOURCE_PROVENANCE),
            _comparison_row(
                "Source disagreement",
                report.get("source_disagreement"),
                verdict_caller_present,
                kind="disagreement",
            ),
            _row(
                "Comparison availability",
                format_comparison_state(
                    report.get("source_disagreement"),
                    caller_present=verdict_caller_present,
                    kind="disagreement",
                ),
                SOURCE_COMPUTED,
            ),
            _row(
                "Dispatcher verification",
                "REQUIRED before any simulated claim is treated as an emergency priority",
                SOURCE_RULE,
            ),
            _row(
                "Override rule",
                "SIMULATED caller evidence never overrides the REAL vision verdict.",
                SOURCE_RULE,
            ),
        ),
    )

    flag_rows = [
        _row("Cross-modal disagreement", item, SOURCE_COMPUTED, limit=72)
        for item in (report.get("disagreements") or [])
    ]
    flag_rows.extend(
        _row("Source disagreement", item, SOURCE_COMPUTED, limit=72)
        for item in (report.get("source_disagreements") or [])
    )
    if not flag_rows:
        flag_rows.append(_row("Disagreements", "none recorded", SOURCE_COMPUTED))
    disagreements = ReportSection("7. Disagreement flags (preserved)", tuple(flag_rows))

    assessment = _assessment_section(report)
    model_prediction = _model_prediction_section(report)
    ocr_evidence = _ocr_section(report)

    inventory = ReportSection(
        "11. Field inventory (all report fields retained)",
        tuple(
            {
                "Field": display_value(key),
                "Value": display_value(value),
                "Source": display_value(_source_for_key(key, report)),
            }
            for key, value in report.items()
        ),
    )

    return [
        identity,
        vision,
        caller,
        location,
        counts,
        verdict,
        disagreements,
        assessment,
        model_prediction,
        ocr_evidence,
        inventory,
    ]


def narrative_markdown(summary_text: Any) -> str:
    """Render the preserved narrative as wrapped Markdown, never as st.code."""
    if summary_text is None or (isinstance(summary_text, str) and not summary_text.strip()):
        return NOT_DETERMINED
    lines = [line.rstrip() for line in str(summary_text).splitlines()]
    return "\n\n".join(line for line in lines if line.strip())
