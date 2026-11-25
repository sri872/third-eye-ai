"""Regression tests for Tab 4 source separation (REAL image + SIMULATED call).

Covers the observed scenario: a real uploaded image paired with call_01.txt,
which is explicitly a simulated sample call. The caller reports MG Road near
Trinity Metro station while a different demo location is associated with the
upload; vision reports 7 vehicles / 2 visible people and NO ACCIDENT SIGNAL,
while the simulated caller reports 2 vehicles / 1 person, injuries and HURRY.
No model is retrained or changed here: these are deterministic evidence
fixtures used to verify separation, provenance and readability.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.incident_report import (
    SOURCE_COMPUTED,
    SOURCE_REAL_VISION,
    SOURCE_SIM_CALLER,
    SOURCE_SIM_DEMO,
    build_report_sections,
    narrative_markdown,
)
from core.nlp.caller_extraction import extract_caller_info
from core.nlp.fusion import fuse_incident_evidence
from core.schemas import AccidentEvent, CallerExtraction, OccupantReport, VehicleReport

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_FILE = REPO_ROOT / "app" / "main.py"
REAL_IMAGE = REPO_ROOT / "data" / "samples" / "road_1.jpg"
CALL_01 = REPO_ROOT / "data" / "sample_calls" / "call_01.txt"
LOCATIONS = REPO_ROOT / "config" / "demo_incident_locations.json"

IMAGE_LOCATION = json.loads(LOCATIONS.read_text(encoding="utf-8"))[0]["name"]
CALLER_LOCATION = "MG Road near Trinity Metro station"


def _vision_evidence() -> tuple[AccidentEvent, list[VehicleReport], list[OccupantReport]]:
    """The reported REAL vision evidence: 7 vehicles, 2 visible people, no signal."""
    event = AccidentEvent(
        event_id="acc-real-image",
        timestamp=datetime.now(timezone.utc),
        location_name="Uploaded media (simulated demo location used only for hospital lookup)",
        latitude=31.2142,
        longitude=75.7785,
        severity="none",
        vehicles_involved=7,
        casualties_estimated=0,
        status="not determined (out-of-domain still image)",
        confidence=0.2166,
        method="TRAINED_MODEL",
        score=0.2166,
        verdict="NO ACCIDENT SIGNAL",
        persons_near_vehicles_count=0,
        reasons="Classifier score: 0.2166 -> Verdict: NO ACCIDENT SIGNAL",
        data_source="REAL",
    )
    vehicles = [
        VehicleReport(vehicle_id=f"veh-real-{i:02d}", vehicle_type="car", data_source="REAL")
        for i in range(1, 8)
    ]
    people = [
        OccupantReport(occupant_id=f"occ-real-{i:02d}", vehicle_id=None, data_source="REAL")
        for i in range(1, 3)
    ]
    return event, vehicles, people


def _caller_evidence() -> CallerExtraction:
    """Use the verbatim simulated call rather than invented caller values."""
    return extract_caller_info(CALL_01.read_text(encoding="utf-8"), call_id="call_01")


@pytest.fixture
def paired_report():
    event, vehicles, people = _vision_evidence()
    return fuse_incident_evidence(
        accident_event=event,
        vehicle_reports=vehicles,
        occupant_reports=people,
        caller_extraction=_caller_evidence(),
        image_name="img1.jpeg",
        image_location=IMAGE_LOCATION,
        image_location_source="SIMULATED",
        save_to_db=False,
    )


class TestSourceSeparatedFusion:
    """Fusion must keep evidence from the two sources independent."""

    def test_real_image_plus_simulated_call_keeps_sources_separate(self, paired_report) -> None:
        assert paired_report.image_name == "img1.jpeg"
        assert paired_report.call_id == "call_01"
        assert paired_report.data_source == "SIMULATED"

        # Vision verdict stays independent and authoritative for the upload.
        assert paired_report.verdict == "NO ACCIDENT SIGNAL"
        assert paired_report.verdict_source == "VISION"
        assert paired_report.vision_score == pytest.approx(0.2166)

        # Caller urgency stays independent and explicitly simulated.
        assert paired_report.urgency == "hurry"
        assert paired_report.urgency_source == "CALLER_SIMULATED"

    def test_location_disagreement_preserves_both_original_values(self, paired_report) -> None:
        assert paired_report.image_location == IMAGE_LOCATION
        assert paired_report.image_location_source == "SIMULATED"
        assert paired_report.caller_location == CALLER_LOCATION
        assert paired_report.location_mismatch is True

        # Legacy fields are unchanged for existing consumers / DB records.
        assert paired_report.location == CALLER_LOCATION
        assert paired_report.location_source == "CALLER_SIMULATED"

        assert any("LOCATION MISMATCH" in item for item in paired_report.source_disagreements)
        assert IMAGE_LOCATION in paired_report.summary_text
        assert CALLER_LOCATION in paired_report.summary_text

    def test_vision_caller_count_disagreement_is_preserved(self, paired_report) -> None:
        assert paired_report.vision_vehicle_count == 7
        assert paired_report.caller_vehicle_count == 2
        assert paired_report.vehicle_count_disagreement is True
        assert paired_report.vision_people_count == 2
        assert paired_report.caller_people_count == 1
        assert paired_report.people_count_disagreement is True
        assert any("VEHICLE COUNT DISAGREEMENT" in item for item in paired_report.disagreements)
        assert any("PEOPLE COUNT DISAGREEMENT" in item for item in paired_report.disagreements)

    def test_conflict_does_not_let_simulated_urgency_override_vision(self, paired_report) -> None:
        assert paired_report.source_disagreement is True
        conflict = " ".join(paired_report.source_disagreements)
        assert "SOURCE DISAGREEMENT" in conflict
        assert "NO ACCIDENT SIGNAL" in conflict
        assert "Dispatcher verification required" in conflict
        assert "never overrides the REAL vision verdict" in conflict
        assert paired_report.verdict == "NO ACCIDENT SIGNAL"
        assert paired_report.urgency_source == "CALLER_SIMULATED"

    def test_injuries_come_only_from_simulated_caller(self, paired_report) -> None:
        assert paired_report.injuries_source == "CALLER_SIMULATED"
        assert paired_report.hazards_source == "CALLER_SIMULATED"
        assert paired_report.injuries == ["bleeding", "unconscious"]
        assert "Injury / Health State: Not determined" in paired_report.summary_text


class TestIncidentReportPresentation:
    """Tab 4 helpers must label provenance and retain every field."""

    def test_every_row_has_a_source_and_sources_are_separated(self, paired_report) -> None:
        report = paired_report.model_dump(mode="json")
        sections = build_report_sections(report)
        assert len(sections) >= 8

        for section in sections:
            assert section.rows
            for row in section.as_table():
                assert set(row) == {"Field", "Value", "Source"}
                assert row["Source"]

        joined = " ".join(
            [section.heading for section in sections]
            + [
                f"{row['Field']} {row['Value']}"
                for section in sections
                for row in section.as_table()
            ]
        )
        assert "REAL VISION EVIDENCE" in joined
        assert "SIMULATED CALLER EVIDENCE" in joined
        assert "no silent merge" in joined.lower()
        assert "Dispatcher verification" in joined

        vision_rows = sections[1].as_table()
        caller_rows = sections[2].as_table()
        assert any(row["Source"] == SOURCE_REAL_VISION for row in vision_rows)
        assert any(row["Source"] == SOURCE_SIM_CALLER for row in caller_rows)
        assert any(row["Source"] == SOURCE_SIM_DEMO for row in sections[3].as_table())
        assert any(row["Source"] == SOURCE_COMPUTED for row in sections[6].as_table())

    def test_all_report_fields_are_retained(self, paired_report) -> None:
        report = paired_report.model_dump(mode="json")
        sections = build_report_sections(report)
        inventory = sections[-1].as_table()
        assert {row["Field"] for row in inventory} == set(report)
        assert len(inventory) == len(report)

    def test_narrative_is_wrapped_markdown_not_a_code_block(self, paired_report) -> None:
        markdown = narrative_markdown(paired_report.summary_text)
        assert "INCIDENT ASSESSMENT" in markdown
        assert "SOURCE DISAGREEMENT" in markdown
        assert "\n\n" in markdown
        assert "```" not in markdown


def _run_app() -> AppTest:
    return AppTest.from_file(str(APP_FILE), default_timeout=300).run()


def _all_report_rows(tab: object) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for node in tab.table:  # type: ignore[attr-defined]
        frame = node.value
        assert list(frame.columns) == ["Field", "Value", "Source"], list(frame.columns)
        rows.extend(frame.to_dict("records"))
    return rows


def test_tab4_renders_separate_sources_for_real_image_plus_simulated_call() -> None:
    """AppTest pairs a real upload with call_01 and checks readable Tab 4 output."""
    assert REAL_IMAGE.exists() and CALL_01.exists()
    at = _run_app()
    at.sidebar.file_uploader[0].upload(
        REAL_IMAGE.name, REAL_IMAGE.read_bytes(), "image/jpeg"
    ).run()
    at.sidebar.selectbox[0].set_value("call_01.txt | SIMULATED SAMPLE CALL").run()

    assert not at.exception, [item.value for item in at.exception]
    assert not at.error, [item.value for item in at.error]
    report_tab = next(tab for tab in at.tabs if tab.label.startswith("4."))

    assert not report_tab.get("code"), "Report must not be rendered as a monospaced block"
    headings = " ".join(node.value for node in report_tab.get("markdown"))
    assert "REAL VISION EVIDENCE" in headings
    assert "SIMULATED CALLER EVIDENCE" in headings
    assert "Location comparison (no silent merge)" in headings
    assert "Dispatcher verification" in headings

    rows = _all_report_rows(report_tab)
    assert rows
    by_field: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_field.setdefault(row["Field"], []).append(row)

    assert any(r["Source"] == SOURCE_REAL_VISION for r in by_field["Vision verdict"])
    assert any(r["Source"] == SOURCE_SIM_CALLER for r in by_field["Caller-reported urgency (SIMULATED)"])
    assert any(r["Value"] == CALLER_LOCATION for r in by_field["Caller-reported location"])
    assert any(r["Value"] == "FLAG" for r in by_field["Location mismatch"])
    assert any(r["Value"] == "FLAG" for r in by_field["Source disagreement"])

    # Structured tables stay narrow enough to avoid horizontal clipping.
    assert max(len(row["Value"]) for row in rows) <= 120
