"""Unit tests for emergency caller entity and keyword extraction (Phase 4).

Validates:
1. spaCy + Rule Matcher extraction against data/sample_calls/expected.json.
2. Precision and recall metrics across all 7 target fields.
3. Negation handling ('no fire or fuel leak', 'did not see any blood', 'nobody is hurt').
4. Honest handling of vague calls (missing fields stay None).
"""

from pathlib import Path
import pytest

from core.nlp.caller_extraction import evaluate_caller_extraction, extract_caller_info
from core.schemas import CallerExtraction

CALLS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_calls"


class TestCallerExtraction:
    """Test caller NLP extraction and answer key evaluation."""

    def test_evaluate_against_expected_json(self):
        """Evaluate extraction on all 8 sample calls against expected.json answer key."""
        res = evaluate_caller_extraction()
        assert res["sample_count"] == 8
        metrics = res["metrics"]

        # Ensure all 7 fields were evaluated
        expected_fields = [
            "location",
            "vehicle_count",
            "vehicle_types",
            "people_involved",
            "injury_keywords",
            "hazard_keywords",
            "urgency_keywords",
        ]
        for field in expected_fields:
            assert field in metrics
            assert "precision" in metrics[field]
            assert "recall" in metrics[field]
            assert "f1" in metrics[field]
            # Verify high performance
            assert metrics[field]["precision"] >= 0.80
            assert metrics[field]["recall"] >= 0.80

    def test_negation_awareness(self):
        """Verify negated symptoms or hazards are excluded from keywords."""
        sample_text = (
            "Minor bump on 100 Feet Road. Two cars involved. "
            "There is no fire and no fuel leak. Nobody is bleeding and no one is hurt. "
            "Do not rush, just normal dispatch."
        )
        ext = extract_caller_info(sample_text, call_id="test_neg")
        assert isinstance(ext, CallerExtraction)
        assert ext.data_source == "SIMULATED"

        # Fire and fuel leak are negated
        assert "fire" not in ext.hazard_keywords
        assert "fuel leak" not in ext.hazard_keywords

        # Bleeding and hurt are negated
        assert "bleeding" not in ext.injury_keywords
        assert "hurt" not in ext.injury_keywords

    def test_vague_call_returns_none(self):
        """Verify vague descriptions do not fabricate locations or counts."""
        sample_text = "I saw some traffic backed up somewhere. Looks like an incident happened."
        ext = extract_caller_info(sample_text, call_id="test_vague")
        assert ext.location is None
        assert ext.vehicle_count is None
        assert ext.people_involved is None
        assert len(ext.injury_keywords) == 0
        assert len(ext.hazard_keywords) == 0
