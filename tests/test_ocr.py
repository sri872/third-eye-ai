"""Tests for Classical OCR and Indian Number Plate Recognition.

Verifies character segmentation, template matching, Indian format validation,
and end-to-end plate reading without neural models.
"""

import os
import pytest
import numpy as np
import cv2

from core.ocr.preprocessing import preprocess_plate, preprocess_plate_variants
from core.ocr.character_segmentation import segment_characters
from core.ocr.template_recognition import TemplateRecognizer, LETTERS, DIGITS
from core.ocr.plate_validation import validate_indian_plate
from core.ocr.plate_reader import ClassicalPlateReader


@pytest.fixture
def recognizer():
    return TemplateRecognizer()


@pytest.fixture
def synthetic_plate_image():
    """Create a high-contrast synthetic plate image with 'TS07JA0566'."""
    plate = np.full((60, 240, 3), 250, dtype=np.uint8)
    # Draw border
    cv2.rectangle(plate, (2, 2), (237, 57), (10, 10, 10), 2)
    # Put text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(plate, "TS07JA0566", (15, 42), font, 0.95, (10, 10, 10), 2, cv2.LINE_AA)
    return plate


def test_preprocess_plate(synthetic_plate_image):
    """Test plate preprocessing returns enhanced and binarized images."""
    enhanced, binary = preprocess_plate(synthetic_plate_image, target_height=64)
    assert enhanced.shape[0] == 64
    assert binary.shape[0] == 64
    assert binary.dtype == np.uint8
    # Characters should be white (255)
    assert np.count_nonzero(binary == 255) > 100


def test_character_segmentation(synthetic_plate_image):
    """Test character segmentation extracts individual glyphs."""
    _, binary = preprocess_plate(synthetic_plate_image, target_height=64)
    segmented = segment_characters(binary, target_char_w=24, target_char_h=32)
    assert len(segmented) >= 8  # Should segment characters from TS07JA0566
    for char_img, bbox in segmented:
        assert char_img.shape == (32, 24)
        assert bbox[2] > 0 and bbox[3] > 0


def test_template_self_matching(recognizer):
    """Test that every letter template matches itself with similarity >= 0.85."""
    for letter in LETTERS:
        tmpl = recognizer.letter_templates[letter]
        best_char, score, _ = recognizer.match_character(tmpl, prior="letter")
        assert best_char == letter, f"Letter {letter} misclassified as {best_char}"
        assert score >= 0.85, f"Letter {letter} self-score too low: {score}"

    for digit in DIGITS:
        tmpl = recognizer.digit_templates[digit]
        best_char, score, _ = recognizer.match_character(tmpl, prior="digit")
        assert best_char == digit, f"Digit {digit} misclassified as {best_char}"
        assert score >= 0.85, f"Digit {digit} self-score too low: {score}"


def test_template_positional_priors(recognizer):
    """Test that positional priors enforce letter vs digit evaluation."""
    # When testing template '0' with letter prior, it should pick a letter (like 'O' or 'D')
    zero_tmpl = recognizer.digit_templates["0"]
    best_char, _, _ = recognizer.match_character(zero_tmpl, prior="letter")
    assert best_char in LETTERS

    # When testing template 'O' with digit prior, it should pick a digit (like '0')
    o_tmpl = recognizer.letter_templates["O"]
    best_char, _, _ = recognizer.match_character(o_tmpl, prior="digit")
    assert best_char in DIGITS


def test_indian_plate_format_validation():
    """Test validation against Indian registration plate standards."""
    # Valid standard registrations
    v1 = validate_indian_plate("TS07JA0566")
    assert v1["is_valid"] is True
    assert v1["state_code"] == "TS"
    assert v1["state_name"] == "Telangana"
    assert v1["rto_code"] == "07"
    assert v1["series"] == "JA"
    assert v1["registration_number"] == "0566"

    v2 = validate_indian_plate("KA01AB1234")
    assert v2["is_valid"] is True
    assert v2["state_code"] == "KA"
    assert v2["state_name"] == "Karnataka"

    v3 = validate_indian_plate("DL01AB1234")
    assert v3["is_valid"] is True
    assert v3["state_name"] == "Delhi"

    # Valid Bharat series
    v_bh = validate_indian_plate("22BH1234AA")
    assert v_bh["is_valid"] is True
    assert v_bh["format_type"] == "BHARAT_SERIES"

    # Invalid strings
    assert validate_indian_plate("INVALID")["is_valid"] is False
    assert validate_indian_plate("")["is_valid"] is False
    assert validate_indian_plate("1234567890")["is_valid"] is False


def test_end_to_end_plate_reader(synthetic_plate_image):
    """Test complete plate reader pipeline on synthetic plate."""
    reader = ClassicalPlateReader()
    result = reader.read_plate(synthetic_plate_image, ground_truth="TS07JA0566")
    assert result["character_count"] >= 8
    assert result["mean_similarity"] > 0.50
    assert result["status"] in ("VERIFIED", "UNVERIFIED")
    assert result["validation"]["is_valid"] is True


def test_plate_reader_empty():
    """Test plate reader handles empty inputs gracefully."""
    reader = ClassicalPlateReader()
    res = reader.read_plate(np.zeros((0, 0, 3), dtype=np.uint8))
    assert res["status"] == "NO_PLATE_DETECTED"
    assert res["is_verified"] is False


def test_preprocess_plate_variants_produce_valid_binary_masks(synthetic_plate_image):
    variants = preprocess_plate_variants(synthetic_plate_image)
    assert variants
    for method, enhanced, binary in variants:
        assert method
        assert enhanced.ndim == 2
        assert binary.shape == enhanced.shape
        assert binary.dtype == np.uint8
        assert set(np.unique(binary)).issubset({0, 255})


def test_ocr_without_ground_truth_is_never_marked_verified(synthetic_plate_image):
    reader = ClassicalPlateReader()
    result = reader.read_plate(synthetic_plate_image)
    assert result["status"] != "VERIFIED"
    assert result["is_verified"] is False
    assert result["similarity_is_probability"] is False
    assert result.get("variant_attempts") is not None


def test_template_recognizer_uses_multiple_classical_font_variants(recognizer):
    assert all(len(variants) > 1 for variants in recognizer.letter_template_bank.values())
    assert all(len(variants) > 1 for variants in recognizer.digit_template_bank.values())
