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
from typing import Any

NOT_DETERMINED = "not determined"

SOURCE_REAL_VISION = "REAL (VISION)"
SOURCE_SIM_CALLER = "SIMULATED (CALLER)"
SOURCE_SIM_DEMO = "SIMULATED (DEMO)"
SOURCE_MISSING = "MISSING"
SOURCE_COMPUTED = "COMPUTED (COMPARISON)"
SOURCE_PROVENANCE = "PROVENANCE METADATA"
SOURCE_RULE = "OPERATIONAL RULE"

_DISAGREEMENT_KEYS = {
    "vehicle_count_disagreement",
    "people_count_disagreement",
    "location_mismatch",
    "source_disagreement",
}


@dataclass(frozen=True)
class ReportSection:
    """One readable table in the incident report tab."""

    heading: str
    rows: tuple[dict[str, str], ...]

    def as_table(self) -> list[dict[str, str]]:
        """Return plain dictionaries suitable for st.table / DataFrame tests."""
        return [dict(row) for row in self.rows]


def display_value(value: Any, limit: int | None = None) -> str:
    """Render a stored value without inventing a replacement."""
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
    if limit is not None and len(text) > limit:
        return text[:limit].rstrip() + f" … (full text below, {len(text)} chars)"
    return text


def flag(value: Any) -> str:
    """Render a disagreement flag in the same wording as the previous report UI."""
    return "FLAG" if bool(value) else "NO FLAG"


def _row(
    field: str, value: Any, source: str, as_flag: bool = False, limit: int | None = None
) -> dict[str, str]:
    return {
        "Field": field,
        "Value": flag(value) if as_flag else display_value(value, limit=limit),
        "Source": source,
    }


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


def build_report_sections(report: dict[str, Any]) -> list[ReportSection]:
    """Build all source-labelled sections for one report dictionary.

    Unknown or absent keys render as ``not determined`` so an older report
    dictionary can still be displayed without fabricating any value.
    """
    has_call = _call_present(report)
    caller_source = SOURCE_SIM_CALLER if has_call else SOURCE_MISSING
    image_source = _image_source(report)

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
            _row("Location mismatch", report.get("location_mismatch"), SOURCE_COMPUTED, as_flag=True),
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
            _row(
                "Vehicle-count disagreement",
                report.get("vehicle_count_disagreement"),
                SOURCE_COMPUTED,
                as_flag=True,
            ),
            _row("Vision people (visible)", report.get("vision_people_count"), SOURCE_REAL_VISION),
            _row("Caller people", report.get("caller_people_count"), caller_source),
            _row(
                "People-count disagreement",
                report.get("people_count_disagreement"),
                SOURCE_COMPUTED,
                as_flag=True,
            ),
        ),
    )

    verdict = ReportSection(
        "6. Multimodal verdict and triage",
        (
            _row("Vision verdict (independent)", report.get("verdict"), SOURCE_REAL_VISION),
            _row("Caller urgency (independent, SIMULATED)", report.get("urgency"), caller_source),
            _row("Urgency source", report.get("urgency_source"), SOURCE_PROVENANCE),
            _row("Source disagreement", report.get("source_disagreement"), SOURCE_COMPUTED, as_flag=True),
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

    inventory = ReportSection(
        "8. Field inventory (all report fields retained)",
        tuple(
            {
                "Field": key,
                "Value": display_value(value, limit=72),
                "Source": _source_for_key(key, report),
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
        inventory,
    ]


def narrative_markdown(summary_text: Any) -> str:
    """Render the preserved narrative as wrapped Markdown, never as st.code."""
    if summary_text is None or (isinstance(summary_text, str) and not summary_text.strip()):
        return NOT_DETERMINED
    lines = [line.rstrip() for line in str(summary_text).splitlines()]
    return "\n\n".join(line for line in lines if line.strip())
