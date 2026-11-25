"""Comprehensive NLP audit test suite for Project Third-Eye.

Tests the 6 mandatory NLP test cases and edge cases:
- Test 1: Collision ("Two cars collided near NH-44. One person may be injured. Please send emergency assistance.")
- Test 2: Heavy traffic ("There are many cars and motorcycles near the main road. Several pedestrians are crossing. No collision has been reported.")
- Test 3: Uncertain incident ("I saw a vehicle stopped on the road. I am not sure whether an accident happened.")
- Test 4: Confirmed caller-reported injury ("Two vehicles crashed near the bus stand. Three people were injured. An ambulance is needed.")
- Test 5: No injury ("One car stopped near NH-44. Nobody was injured and no ambulance was requested.")
- Test 6: Contradiction ("The caller first reports two cars, then says only one car was involved.")
- Additional: empty text, whitespace, punctuation/capitalization, long text, and Tab 4 integration.
"""

from datetime import datetime, timezone
import pytest

from core.nlp.caller_extraction import (
    extract_caller_info,
    evaluate_caller_extraction,
)
from core.nlp.fusion import fuse_incident_evidence
from core.schemas import AccidentEvent
from app.incident_report import build_report_sections


class TestNLPAuditMandatoryCases:
    """Verify the 6 required NLP demonstration test cases."""

    def test_case_1_collision(self):
        """Test 1 — Collision:
        'Two cars collided near NH-44. One person may be injured. Please send emergency assistance.'
        Expected:
        - Vehicle count: 2
        - Location: NH-44
        - Injury: possible
        - Emergency request: present
        """
        text = "Two cars collided near NH-44. One person may be injured. Please send emergency assistance."
        res = extract_caller_info(text, call_id="test_case_1")

        assert res.vehicle_count == 2
        assert res.location == "NH-44"
        assert res.injury_status == "possible"
        assert "injured" in res.injury_keywords
        assert res.emergency_assistance_requested is True
        assert res.accident_status == "reported"
        assert res.uncertainty_detected is True

    def test_case_2_heavy_traffic(self):
        """Test 2 — Heavy traffic:
        'There are many cars and motorcycles near the main road. Several pedestrians are crossing. No collision has been reported.'
        Expected:
        - Cars: mentioned
        - Motorcycles: mentioned
        - Pedestrians: mentioned
        - Exact counts: not determined
        - Collision: explicitly not reported
        """
        text = "There are many cars and motorcycles near the main road. Several pedestrians are crossing. No collision has been reported."
        res = extract_caller_info(text, call_id="test_case_2")

        assert "car" in res.vehicle_types
        assert "motorcycle" in res.vehicle_types
        assert res.pedestrians_mentioned is True
        # Exact counts must stay None (not determined) from indefinite words
        assert res.vehicle_count is None
        assert res.people_involved is None
        assert res.accident_status == "explicitly_not_reported"
        assert res.negation_detected is True

    def test_case_3_uncertain_incident(self):
        """Test 3 — Uncertain incident:
        'I saw a vehicle stopped on the road. I am not sure whether an accident happened.'
        Expected:
        - Vehicle: mentioned
        - Accident: uncertain
        - Injury: not determined
        """
        text = "I saw a vehicle stopped on the road. I am not sure whether an accident happened."
        res = extract_caller_info(text, call_id="test_case_3")

        assert "vehicle" in res.vehicle_types or res.vehicle_count == 1
        assert res.accident_status == "uncertain"
        assert res.injury_status == "not determined"
        assert res.uncertainty_detected is True

    def test_case_4_confirmed_injury(self):
        """Test 4 — Confirmed caller-reported injury:
        'Two vehicles crashed near the bus stand. Three people were injured. An ambulance is needed.'
        Expected:
        - Vehicle count: 2
        - Person count: 3
        - Injury: caller-reported
        - Ambulance request: present
        - Location: bus stand, without invented coordinates
        """
        text = "Two vehicles crashed near the bus stand. Three people were injured. An ambulance is needed."
        res = extract_caller_info(text, call_id="test_case_4")

        assert res.vehicle_count == 2
        assert res.people_involved == 3
        assert res.injury_status == "confirmed_caller_reported"
        assert "injured" in res.injury_keywords
        assert res.ambulance_requested is True
        assert res.location is not None
        assert "bus stand" in res.location.lower()
        assert res.accident_status == "reported"

    def test_case_5_no_injury(self):
        """Test 5 — No injury:
        'One car stopped near NH-44. Nobody was injured and no ambulance was requested.'
        Expected:
        - Vehicle count: 1
        - Location: NH-44
        - Injury: explicitly reported absent
        - Ambulance request: explicitly absent
        """
        text = "One car stopped near NH-44. Nobody was injured and no ambulance was requested."
        res = extract_caller_info(text, call_id="test_case_5")

        assert res.vehicle_count == 1
        assert res.location == "NH-44"
        assert res.injury_status == "explicitly_absent"
        assert len(res.injury_keywords) == 0
        assert res.ambulance_requested is False
        assert res.negation_detected is True

    def test_case_6_contradiction(self):
        """Test 6 — Contradiction:
        'The caller first reports two cars, then says only one car was involved.'
        Expected:
        - Both statements preserved
        - Count discrepancy flagged
        - No silent resolution
        """
        text = "The caller first reports two cars, then says only one car was involved."
        res = extract_caller_info(text, call_id="test_case_6")

        assert len(res.contradictions) > 0
        assert any("vehicle count contradiction" in c.lower() for c in res.contradictions)
        # Vehicle count must not be silently resolved to one number
        assert res.vehicle_count is None
        # Raw text preserves full transcript
        assert "first reports two cars" in res.raw_text
        assert "only one car was involved" in res.raw_text


class TestNLPAuditEdgeCases:
    """Verify edge cases: empty strings, capitalization, punctuation, long texts."""

    def test_empty_and_whitespace_input(self):
        """Empty or whitespace input must return a clean, unpopulated CallerExtraction without crashing."""
        for empty_val in ["", "   ", "\n\t\n"]:
            res = extract_caller_info(empty_val, call_id="test_empty")
            assert res.location is None
            assert res.vehicle_count is None
            assert res.people_involved is None
            assert res.injury_status == "not determined"
            assert res.accident_status == "not determined"
            assert res.ambulance_requested is None
            assert res.emergency_assistance_requested is None
            assert len(res.contradictions) == 0

    def test_capitalization_and_punctuation_tolerance(self):
        """Should handle ALL CAPS, lowercase highways, and diverse punctuation."""
        text = "COLLISION ON nh-44! TWO CARS INVOLVED. ONE PERSON IS INJURED."
        res = extract_caller_info(text, call_id="test_caps")
        assert res.location == "NH-44"
        assert res.vehicle_count == 2
        assert res.injury_status == "confirmed_caller_reported"

    def test_long_caller_description(self):
        """Very long transcript should process cleanly without timeouts or exceptions."""
        long_text = (
            "Caller: Hello operator, I am driving along GT Road near Phagwara bypass. "
            "Traffic is moving slowly and there are two cars stopped ahead. One car has a damaged bumper. "
            "I saw the driver getting out, nobody was injured according to him, but traffic is backing up. "
            "No ambulance was requested, just need a traffic patrol officer to clear the lane. "
            + "Repeating again for clarity, everything is calm. " * 10
        )
        res = extract_caller_info(long_text, call_id="test_long")
        assert res.location is not None
        assert res.injury_status == "explicitly_absent"
        assert res.ambulance_requested is False

    def test_tab4_fusion_integration(self):
        """Verify that extracted NLP fields flow cleanly through fusion into Tab 4 report sections."""
        text = "Two cars collided near NH-44. One person may be injured. Please send emergency assistance."
        caller = extract_caller_info(text, call_id="test_fusion_tab4")

        accident_event = AccidentEvent(
            event_id="evt-demo",
            timestamp=datetime.now(timezone.utc),
            location_name="NH-44",
            latitude=31.22,
            longitude=75.77,
            severity="medium",
            vehicles_involved=2,
            score=0.35,
            verdict="UNCERTAIN",
            data_source="REAL",
        )

        fused = fuse_incident_evidence(
            accident_event=accident_event,
            caller_extraction=caller,
            image_name="demo.jpg",
            save_to_db=False,
            media_type="image",
        )

        assert fused.injury_status == "possible"
        assert fused.accident_status == "reported"
        assert fused.emergency_assistance_requested is True

        # Render report sections for Tab 4
        sections = build_report_sections(fused.model_dump(mode="json"))
        section_titles = [s.heading for s in sections]

        assert any("SIMULATED CALLER EVIDENCE" in t for t in section_titles)
        assert any("Disagreement flags" in t for t in section_titles)

        # Inspect caller rows in Section 3
        caller_sec = next(s for s in sections if "SIMULATED CALLER EVIDENCE" in s.heading)
        rows_dict = {r["Field"]: r["Value"] for r in caller_sec.as_table()}

        assert rows_dict["Caller accident status"] == "reported"
        assert rows_dict["Caller injury status"] == "possible"
        assert rows_dict["Emergency assistance requested"] == "YES"

    def test_benchmark_answer_key_still_passes(self):
        """Ensure all 8 sample calls in expected.json still exceed 0.80 benchmark threshold."""
        eval_result = evaluate_caller_extraction()
        assert eval_result["sample_count"] == 8
        metrics = eval_result["metrics"]
        for field in ["location", "vehicle_count", "vehicle_types", "people_involved", "injury_keywords"]:
            assert metrics[field]["precision"] >= 0.80, f"Precision fell below 0.80 on {field}"
            assert metrics[field]["recall"] >= 0.80, f"Recall fell below 0.80 on {field}"
