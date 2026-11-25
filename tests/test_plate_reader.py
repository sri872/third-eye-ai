"""Unit and integration tests for license plate reader (core/ocr/plate_reader.py).

Tests:
- Plate text normalization and sanitization (MOCKED / unit).
- Real confidence extraction and unreadable handling (MOCKED).
- One live EasyOCR test on real sample photos in data/samples/.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

import core.ocr.plate_reader as plate_reader
from core.ocr.plate_reader import (
    PLATE_TRUST_THRESHOLD,
    PlateOCREvidence,
    generate_plate_preprocessing_variants,
    is_valid_indian_plate_format,
    is_verified_plate_confidence,
    normalize_plate_text,
    preprocess_plate_region,
    read_plate,
    read_plate_evidence,
)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


class TestPlateReaderMocked:
    """Tests using explicitly MOCKED OCR output for logic and normalization."""

    def test_normalize_plate_text(self):
        # Basic lowercase and spacing normalization
        assert normalize_plate_text("ka 05 mh 1234") == "KA05MH1234"
        # Special characters and punctuation stripped
        assert normalize_plate_text("DL-01-AB-9999!") == "DL01AB9999"
        # Too short (< 3 characters) returns None
        assert normalize_plate_text("A1") is None
        assert normalize_plate_text("") is None
        assert normalize_plate_text(None) is None

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_read_plate_with_mocked_success(self, mock_get_reader):
        """MOCKED test: Valid OCR detection returning normalized plate and real confidence."""
        mock_reader = MagicMock()
        # EasyOCR output structure: [(bbox, text, prob)]
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "KA01AB1234", 0.9412)
        ]
        mock_get_reader.return_value = (mock_reader, "cpu")

        dummy_crop = np.zeros((50, 100, 3), dtype=np.uint8)
        plate, conf = read_plate(dummy_crop)

        assert plate == "KA01AB1234"
        assert conf == 0.9412

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_read_plate_with_mocked_unreadable(self, mock_get_reader):
        """MOCKED test: OCR finds nothing or only symbols, returns (None, None)."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mock_get_reader.return_value = (mock_reader, "cpu")

        dummy_crop = np.zeros((50, 100, 3), dtype=np.uint8)
        plate, conf = read_plate(dummy_crop)

        assert plate is None
        assert conf is None

    @patch("core.ocr.plate_reader.get_ocr_reader", side_effect=RuntimeError("models unavailable"))
    def test_read_plate_reader_failure_is_unreadable(self, mock_get_reader):
        """OCR/reader failure never fabricates a plate: it stays unreadable."""
        dummy_crop = np.zeros((50, 100, 3), dtype=np.uint8)
        assert read_plate(dummy_crop) == (None, None)

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_read_plate_selects_highest_confidence(self, mock_get_reader):
        """MOCKED test: Selects highest confidence when multiple candidate texts exist."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "LOWCONF1", 0.4500),
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "HIGHCONF2", 0.9650),
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "MIDCONF3", 0.7200),
        ]
        mock_get_reader.return_value = (mock_reader, "cpu")

        dummy_crop = np.zeros((50, 100, 3), dtype=np.uint8)
        plate, conf = read_plate(dummy_crop)

        assert plate == "HIGHCONF2"
        assert conf == 0.9650

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_evidence_keeps_raw_text_verbatim_below_verification_threshold(self, mock_get_reader):
        """Low-confidence text stays raw and unverified; normalization never silently corrects it."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "car0.99", 0.6990),
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "??", 0.3100),
        ]
        mock_get_reader.return_value = (mock_reader, "cpu")

        evidence = read_plate_evidence(np.zeros((50, 100, 3), dtype=np.uint8))
        assert evidence.plate == "CAR099"
        assert evidence.plate_confidence == 0.6990
        assert evidence.raw_text == "car0.99"
        assert evidence.raw_confidence == 0.6990
        assert evidence.candidate_count == 2
        assert evidence.ocr_status == "OK"

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_evidence_no_valid_plate_still_keeps_raw_text(self, mock_get_reader):
        """When no candidate normalizes to a plate, raw text is still shown, unverified."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "??", 0.8100),
        ]
        mock_get_reader.return_value = (mock_reader, "cpu")

        evidence = read_plate_evidence(np.zeros((50, 100, 3), dtype=np.uint8))
        assert evidence.plate is None
        assert evidence.plate_confidence is None
        assert evidence.raw_text == "??"
        assert evidence.raw_confidence == 0.8100

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_evidence_records_no_results_and_reader_failure(self, mock_get_reader):
        """No text and reader failure both produce empty evidence, not invented text."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mock_get_reader.return_value = (mock_reader, "cpu")
        crop = np.zeros((50, 100, 3), dtype=np.uint8)

        empty = read_plate_evidence(crop)
        assert empty.raw_text is None and empty.raw_confidence is None
        assert empty.candidate_count == 0
        assert empty.ocr_status == "NO OCR RESULTS"
        assert read_plate(crop) == (None, None)

        mock_get_reader.side_effect = RuntimeError("models unavailable")
        failed = read_plate_evidence(crop)
        assert failed.raw_text is None and failed.raw_confidence is None
        assert failed.candidate_count == 0
        assert failed.ocr_status == "OCR INITIALIZATION FAILURE"

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_evidence_small_crop_skips_ocr(self, mock_get_reader):
        """A crop below the 15px minimum must not call EasyOCR or invent text."""
        evidence = read_plate_evidence(np.zeros((10, 100, 3), dtype=np.uint8))
        assert evidence.raw_text is None and evidence.candidate_count == 0
        assert evidence.ocr_status == "CROP TOO SMALL"
        mock_get_reader.assert_not_called()

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_evidence_records_inference_failure_separately(self, mock_get_reader):
        """OCR inference failure is a distinct status and never yields a plate."""
        reader = MagicMock()
        reader.readtext.side_effect = RuntimeError("decode failed")
        mock_get_reader.return_value = (reader, "cpu")

        evidence = read_plate_evidence(np.zeros((50, 100, 3), dtype=np.uint8))
        assert evidence.plate is None
        assert evidence.ocr_status == "OCR INFERENCE FAILURE"

    def test_preprocess_plate_region_upscales_without_binarising(self):
        """Preprocessing gently upscales a detected region without creating text."""
        region = np.full((24, 80, 3), 180, dtype=np.uint8)
        processed = preprocess_plate_region(region)
        assert processed.ndim == 3
        assert processed.shape[0] >= 48
        assert processed.shape[1] >= 160
        assert set(np.unique(processed)) != {0, 255}, "Must not binarise plate pixels"


class TestPlateFormatVariantAndGateRegression:
    """Format hints, deterministic variants and the fixed 0.70 verification gate."""

    def test_format_valid_standard_and_bharat_patterns(self):
        assert is_valid_indian_plate_format("TS07JA0566")
        assert is_valid_indian_plate_format(normalize_plate_text("ka-05-mh-1234"))
        assert is_valid_indian_plate_format("21BH1234AA")
        # Partial/alphanumeric reads are not registration formats.
        assert not is_valid_indian_plate_format("IND")
        assert not is_valid_indian_plate_format("CAR099")
        assert not is_valid_indian_plate_format("??")
        assert not is_valid_indian_plate_format(None)

    def test_evidence_defaults_are_backward_compatible(self):
        evidence = PlateOCREvidence("ABC123", 0.8, "abc-123", 0.8, 1)
        assert evidence.format_valid is False
        assert evidence.preprocessing_variant is None

    def test_variants_are_deterministic_and_non_fabricating(self):
        crop = np.full((30, 90, 3), 150, dtype=np.uint8)
        variants = generate_plate_preprocessing_variants(crop)
        assert list(variants)[:4] == ["raw", "padded", "contrast_enhanced", "sharpened"]
        assert variants["raw"] is crop
        assert variants["padded"].shape[0] > crop.shape[0]
        for image in variants.values():
            assert isinstance(image, np.ndarray) and image.size > 0
        assert generate_plate_preprocessing_variants(crop).keys() == variants.keys()

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_format_valid_wins_over_higher_confidence_partial(self, mock_get_reader, monkeypatch):
        """Registration-format tier outranks alphanumeric/partial, regardless of confidence."""
        crop = np.full((40, 120, 3), 160, dtype=np.uint8)
        monkeypatch.setattr(
            plate_reader,
            "generate_plate_preprocessing_variants",
            lambda image: {"raw": image, "contrast_enhanced": image + 1},
        )
        reader = MagicMock()
        reader.readtext.side_effect = [
            [([[0, 0], [10, 0], [10, 10], [0, 10]], "IND", 0.99)],
            [([[0, 0], [10, 0], [10, 10], [0, 10]], "KA05MH1234", 0.71)],
        ]
        mock_get_reader.return_value = (reader, "cpu")

        evidence = read_plate_evidence(crop)
        assert evidence.plate == "KA05MH1234"
        assert evidence.plate_confidence == 0.71
        assert evidence.format_valid is True
        assert evidence.preprocessing_variant == "contrast_enhanced"
        # Raw text still reflects what EasyOCR actually emitted, verbatim.
        assert evidence.raw_text == "IND"
        assert evidence.raw_confidence == 0.99

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_higher_real_confidence_wins_within_same_tier(self, mock_get_reader, monkeypatch):
        crop = np.full((40, 120, 3), 160, dtype=np.uint8)
        monkeypatch.setattr(
            plate_reader,
            "generate_plate_preprocessing_variants",
            lambda image: {"raw": image, "padded": image + 1},
        )
        reader = MagicMock()
        reader.readtext.side_effect = [
            [([[0, 0], [10, 0], [10, 10], [0, 10]], "ABC123", 0.81)],
            [([[0, 0], [10, 0], [10, 10], [0, 10]], "XYZ789", 0.92)],
        ]
        mock_get_reader.return_value = (reader, "cpu")

        evidence = read_plate_evidence(crop)
        assert evidence.plate == "XYZ789"
        assert evidence.plate_confidence == 0.92
        assert evidence.preprocessing_variant == "padded"
        assert evidence.format_valid is False

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_raw_text_and_confidence_are_never_rounded_or_corrected(self, mock_get_reader, monkeypatch):
        crop = np.full((40, 120, 3), 160, dtype=np.uint8)
        monkeypatch.setattr(
            plate_reader,
            "generate_plate_preprocessing_variants",
            lambda image: {"raw": image},
        )
        reader = MagicMock()
        reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "ka-05 mh 1234", 0.7123456)
        ]
        mock_get_reader.return_value = (reader, "cpu")

        evidence = read_plate_evidence(crop)
        assert evidence.raw_text == "ka-05 mh 1234"
        assert evidence.raw_confidence == 0.7123456
        assert evidence.plate == "KA05MH1234"
        assert evidence.plate_confidence == 0.7123456

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_identical_crops_do_not_repeat_ocr(self, mock_get_reader, monkeypatch):
        """Variant drill-down is memoized, so app/main.py's two calls share OCR."""
        crop = np.full((40, 120, 3), 170, dtype=np.uint8)
        monkeypatch.setattr(
            plate_reader,
            "generate_plate_preprocessing_variants",
            lambda image: {"raw": image, "padded": image},
        )
        reader = MagicMock()
        reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "KA05MH1234", 0.88)
        ]
        mock_get_reader.return_value = (reader, "cpu")

        first = read_plate_evidence(crop)
        assert reader.readtext.call_count == 1  # raw and padded are the same pixels
        second = read_plate_evidence(crop)
        third = read_plate_evidence(crop, evaluate_variants=False)
        assert reader.readtext.call_count == 1
        assert (first.plate, second.plate, third.plate) == ("KA05MH1234",) * 3

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_processed_then_raw_crop_shares_ocr(self, mock_get_reader, monkeypatch):
        """The app's processed/raw pair must not OCR the same pixels twice."""
        crop = np.full((40, 120, 3), 180, dtype=np.uint8)
        processed = preprocess_plate_region(crop)

        def variants(image):
            if image is crop:
                return {"raw": crop, "contrast_enhanced": processed}
            return {"raw": image}

        monkeypatch.setattr(plate_reader, "generate_plate_preprocessing_variants", variants)
        reader = MagicMock()
        reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "KA05MH1234", 0.88)
        ]
        mock_get_reader.return_value = (reader, "cpu")

        read_plate_evidence(processed)
        assert reader.readtext.call_count == 1
        evidence = read_plate_evidence(crop)
        assert reader.readtext.call_count == 2
        assert evidence.preprocessing_variant in {"raw", "contrast_enhanced"}

    @patch("core.ocr.plate_reader.get_ocr_reader")
    def test_evaluate_variants_false_reads_raw_only(self, mock_get_reader, monkeypatch):
        crop = np.full((40, 120, 3), 175, dtype=np.uint8)
        monkeypatch.setattr(
            plate_reader,
            "generate_plate_preprocessing_variants",
            lambda image: pytest.fail("variants must not be generated"),
        )
        reader = MagicMock()
        reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "KA05MH1234", 0.93)
        ]
        mock_get_reader.return_value = (reader, "cpu")

        evidence = read_plate_evidence(crop, evaluate_variants=False)
        assert reader.readtext.call_count == 1
        assert evidence.preprocessing_variant == "raw"

    def test_verification_gate_is_confidence_only(self):
        """Format validity never verifies a plate; only >= 0.70 OCR confidence does."""
        assert PLATE_TRUST_THRESHOLD == 0.7
        formatted_but_low = PlateOCREvidence(
            plate="KA05MH1234", plate_confidence=0.699, format_valid=True
        )
        assert not is_verified_plate_confidence(formatted_but_low.plate_confidence)
        assert is_verified_plate_confidence(0.70)
        assert is_verified_plate_confidence(0.9412)
        assert not is_verified_plate_confidence(None)


class TestPlateReaderLive:
    """Live OCR test on real sample photos."""

    def test_live_ocr_on_real_sample_photo(self):
        """Runs live EasyOCR on a real photo crop from data/samples/.

        Skips cleanly if no sample photos exist.
        """
        sample_files = list(SAMPLES_DIR.glob("*.jpg"))
        if not sample_files:
            pytest.skip("No real sample images found in data/samples/. Add a real photo to run live OCR verification.")

        img = cv2.imread(str(sample_files[0]))
        assert img is not None

        # Take a real crop from the image
        h, w = img.shape[:2]
        crop = img[int(h * 0.4) : int(h * 0.8), int(w * 0.1) : int(w * 0.6)]

        plate, conf = read_plate(crop)
        # Verify valid return types (either None or valid string and float in [0.0, 1.0])
        if plate is not None:
            assert isinstance(plate, str)
            assert len(plate) >= 3
            assert isinstance(conf, float)
            assert 0.0 <= conf <= 1.0
        else:
            assert conf is None
