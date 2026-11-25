"""Regression tests for the Tab 4 incident-report presentation upgrade.

The report layer only formats values that already exist in the report. These
tests pin that behaviour: the four-tier category is passed through verbatim,
raw model output stays separate from the assessment, OCR candidates are never
promoted to verified plates, and missing caller data is never compared.
"""

from __future__ import annotations

import pytest

from app.incident_report import (
    MAX_DISPLAY_LENGTH,
    NOT_DETERMINED,
    PLATE_UNVERIFIED,
    SOURCE_MODEL,
    SOURCE_OCR,
    SOURCE_RULE,
    SOURCE_UNKNOWN,
    build_report_sections,
    display_value,
    format_comparison_state,
)

CATEGORIES = (
    "ACCIDENT EVIDENCE DETECTED",
    "POSSIBLE ACCIDENT",
    "INCONCLUSIVE",
    "NO CLEAR ACCIDENT EVIDENCE DETECTED",
)

NOT_COMPARABLE = "NOT COMPARABLE — CALLER DATA MISSING"


def _report(**overrides: object) -> dict:
    """A minimal report dictionary shaped like ``IncidentReport.model_dump``."""
    report: dict = {
        "report_id": "inc-test-001",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "image_name": "road_1.jpg",
        "call_id": "call_01",
        "caller_location": "MG Road near Trinity Metro station",
        "caller_vehicle_count": 2,
        "caller_people_count": 1,
        "verdict": "NO ACCIDENT SIGNAL",
        "verdict_source": "VISION",
        "vision_score": 0.2166,
        "location_mismatch": True,
        "vehicle_count_disagreement": True,
        "people_count_disagreement": False,
        "source_disagreement": True,
        "urgency": "hurry",
        "injuries": ["bleeding"],
        "hazards": [],
        "caller_contradictions": [],
        "summary_text": "INCIDENT ASSESSMENT\nDispatcher verification required.",
    }
    report.update(overrides)
    return report


def _assessment(**overrides: object) -> dict:
    assessment: dict = {
        "assessment_category": "INCONCLUSIVE",
        "model_verdict": "NO ACCIDENT SIGNAL",
        "model_score": 0.2166,
        "media_type": "image",
        "domain_compatibility": "NOT DETERMINED (out-of-domain still image)",
        "scene_status": "NOT DETERMINED (out-of-domain still image)",
        "evidence": ["vehicle-shaped regions present in the still image"],
        "contradictory_evidence": ["caller reports an injury"],
        "evidence_sources": ["REAL (VISION)"],
        "uncertainty": "HIGH",
        "human_review_required": True,
        "reasoning": "Still image is outside the classifier training domain.",
        "temporal_evidence_available": False,
        "data_source": "REAL",
    }
    assessment.update(overrides)
    return assessment


def _assessment_rows(report: dict) -> dict[str, dict[str, str]]:
    section = build_report_sections(report)[7]
    assert section.heading == "8. Four-tier accident evidence assessment"
    return {row["Field"]: row for row in section.as_table()}


@pytest.mark.parametrize("category", CATEGORIES)
def test_assessment_category_is_displayed_exactly_as_supplied(category: str) -> None:
    report = _report(evidence_assessment=_assessment(assessment_category=category))
    rows = _assessment_rows(report)

    assert rows["Assessment category"]["Value"] == category
    assert rows["Assessment category"]["Source"]


def test_human_review_is_mandatory_and_prediction_rows_are_labelled() -> None:
    rows = _assessment_rows(_report(evidence_assessment=_assessment()))

    assert rows["Human review required"]["Value"] == "MANDATORY"
    assert rows["Human review required"]["Source"] == SOURCE_RULE
    assert rows["Model verdict"]["Source"] == SOURCE_MODEL
    assert rows["Uncalibrated model score"]["Source"] == SOURCE_MODEL
    assert rows["Model verdict"]["Value"] == "NO ACCIDENT SIGNAL"


def test_raw_model_verdict_is_separate_from_category_and_score_is_uncalibrated() -> None:
    category = "NO CLEAR ACCIDENT EVIDENCE DETECTED"
    raw_verdict = "NO ACCIDENT SIGNAL"
    report = _report(
        evidence_assessment=_assessment(
            assessment_category=category,
            model_verdict=raw_verdict,
            model_score=0.2166,
        )
    )
    rows = _assessment_rows(report)

    assert rows["Assessment category"]["Value"] == category
    assert rows["Model verdict"]["Value"] == raw_verdict
    assert rows["Assessment category"]["Value"] != rows["Model verdict"]["Value"]
    assert rows["Uncalibrated model score"]["Value"] == "0.2166"
    assert "Uncalibrated" in rows["Uncalibrated model score"]["Field"]

    prediction_rows = {
        row["Field"]: row for row in build_report_sections(report)[8].as_table()
    }
    assert prediction_rows["Raw model verdict"]["Value"] == raw_verdict
    assert prediction_rows["Raw model verdict"]["Source"] == SOURCE_MODEL


def test_missing_assessment_is_not_determined_without_inventing_a_category() -> None:
    rows = _assessment_rows(_report())

    assert rows["Assessment category"]["Value"] == NOT_DETERMINED
    assert all(row["Value"] == NOT_DETERMINED for row in rows.values())
    assert not any(category in row["Value"] for row in rows.values() for category in CATEGORIES)


def test_ocr_below_threshold_is_unverified_and_raw_text_is_retained() -> None:
    raw_text = "PB01 AB 1234"
    report = _report(
        license_plate="PB01AB1234",
        plate_confidence=0.42,
        plate_detection_status="FORMAT VALID",
        raw_ocr_text=raw_text,
    )
    ocr_rows = build_report_sections(report)[9].as_table()
    values = [row["Value"] for row in ocr_rows]

    assert PLATE_UNVERIFIED in values
    assert "PLATE IDENTIFICATION: VERIFIED" not in values
    assert any(raw_text in value for value in values)
    assert any("PB01AB1234" in value for value in values)
    assert any(row["Source"] == SOURCE_OCR for row in ocr_rows)
    assert any(row["Source"] == SOURCE_UNKNOWN for row in ocr_rows)


def test_ocr_without_plate_fields_is_not_readable() -> None:
    ocr_rows = build_report_sections(_report())[9].as_table()

    assert any("not determined (unreadable or unverified)" in row["Value"] for row in ocr_rows)


def test_caller_missing_comparison_state_is_explicit() -> None:
    report = _report(
        call_id=None,
        caller_location=None,
        caller_vehicle_count=None,
        caller_people_count=None,
        urgency=None,
        injuries=[],
        hazards=[],
        caller_contradictions=[],
        location_mismatch=False,
        vehicle_count_disagreement=False,
        people_count_disagreement=False,
        source_disagreement=False,
    )
    sections = build_report_sections(report)

    for index in (3, 4, 5):
        assert any(
            row["Value"] == NOT_COMPARABLE for row in sections[index].as_table()
        )

    assert format_comparison_state(None, caller_present=False) == NOT_COMPARABLE
    assert format_comparison_state(True, caller_present=True) == "MISMATCH"
    assert format_comparison_state(True, caller_present=True, kind="disagreement") == "DISAGREEMENT"
    assert format_comparison_state(False, caller_present=True, kind="source") == "NO DISAGREEMENT"


def test_section_order_inventory_and_cell_widths() -> None:
    report = _report(evidence_assessment=_assessment(reasoning="R" * 500))
    sections = build_report_sections(report)

    assert [section.heading[:2] for section in sections[:7]] == [
        "1.", "2.", "3.", "4.", "5.", "6.", "7."
    ]
    assert sections[7].heading == "8. Four-tier accident evidence assessment"
    assert sections[-1].heading.startswith("11. Field inventory")
    assert {row["Field"] for row in sections[-1].as_table()} == set(report)

    for section in sections:
        for row in section.as_table():
            for cell in row.values():
                assert len(cell) <= MAX_DISPLAY_LENGTH


def test_display_value_truncation_includes_suffix_in_budget() -> None:
    rendered = display_value("x" * 400, limit=120)

    assert len(rendered) <= MAX_DISPLAY_LENGTH
    assert "full text below, 400 chars" in rendered
