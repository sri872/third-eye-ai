"""License plate reading module for Project Third-Eye.

Extracts text candidates from plate regions or legacy whole-vehicle crops using
EasyOCR with lazy loading. Normalizes plate text (uppercase, alphanumeric only)
without fabricating characters. Returns real OCR confidence or (None, None) when
unreadable, while preserving the exact raw OCR text and its status.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Optional, Tuple

import cv2
import easyocr
import numpy as np

logger = logging.getLogger("third_eye.ocr.plate_reader")

# Project root models directory for isolated local storage
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "easyocr"
USER_NETWORK_DIR = MODEL_DIR / "user_network"

_READER: Optional[easyocr.Reader] = None
_DEVICE: str = "cpu"


@dataclass(frozen=True)
class PlateOCREvidence:
    """Real OCR output for one vehicle crop, with no inferred characters.

    ``plate`` / ``plate_confidence`` are the same values returned by ``read_plate``.
    ``raw_text`` / ``raw_confidence`` keep the highest-confidence text that EasyOCR
    actually emitted, even when normalization rejects it. This lets the UI show
    recognized text honestly without presenting it as a verified registration.
    Verification itself stays in the UI at PLATE_TRUST_THRESHOLD = 0.7.
    """

    plate: Optional[str] = None
    plate_confidence: Optional[float] = None
    raw_text: Optional[str] = None
    raw_confidence: Optional[float] = None
    candidate_count: int = 0
    # Dedicated-detector provenance. Defaults keep the Phase 11 interface intact.
    plate_detection_status: str = "NOT RUN (LEGACY WHOLE-CROP OCR)"
    plate_bbox: Optional[Tuple[int, int, int, int]] = None
    plate_detector: str = "NONE"
    ocr_status: str = "NOT RUN"


def get_ocr_reader() -> Tuple[easyocr.Reader, str]:
    """Lazy-load the EasyOCR English reader once and cache it."""
    global _READER, _DEVICE

    if _READER is None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        USER_NETWORK_DIR.mkdir(parents=True, exist_ok=True)

        _DEVICE = "cpu"
        print(f"[Third-Eye OCR] Initializing EasyOCR Reader on device: {_DEVICE.upper()}")
        
        _READER = easyocr.Reader(
            ["en"],
            gpu=False,
            model_storage_directory=str(MODEL_DIR),
            user_network_directory=str(USER_NETWORK_DIR),
            download_enabled=False,
            verbose=False,
        )

    return _READER, _DEVICE


def normalize_plate_text(raw_text: str) -> Optional[str]:
    """Normalize plate text: uppercase, alphanumeric characters only.

    Strict rule: Never invent, guess, or auto-correct characters.
    Returns None if result is empty or too short (< 3 characters).
    """
    if not raw_text:
        return None
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if len(cleaned) < 3:
        return None
    return cleaned


def preprocess_plate_region(region: np.ndarray) -> np.ndarray:
    """Mild, non-fabricating preprocessing for a detected plate region.

    Conversions are limited to grayscale, CLAHE, light denoising and aspect-
    preserving upscaling. No binarisation or character synthesis is performed.
    The caller keeps the exact raw OCR text emitted from this image.
    """
    if region is None or not isinstance(region, np.ndarray) or region.size == 0:
        raise ValueError("Plate region is empty")

    gray = (
        cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        if region.ndim == 3 and region.shape[2] >= 3
        else region
    )
    equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.fastNlMeansDenoising(equalised, None, 5, 7, 21)
    height, width = denoised.shape[:2]
    scale = max(1.0, 48.0 / max(height, 1), 160.0 / max(width, 1))
    scale = min(scale, 4.0)
    if scale > 1.0:
        resized = cv2.resize(
            denoised,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        resized = denoised
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def read_plate_evidence(crop: np.ndarray) -> PlateOCREvidence:
    """Collect real EasyOCR evidence from a plate region without guessing text.

    No confidence threshold is applied here: the goal is to keep the exact
    recognized text and its real confidence so the UI can decide what is verified.
    Character normalization is unchanged (``normalize_plate_text``).
    """
    if crop is None or crop.size == 0:
        return PlateOCREvidence(ocr_status="EMPTY CROP")

    if min(crop.shape[:2]) < 15:
        return PlateOCREvidence(ocr_status="CROP TOO SMALL")

    try:
        reader, _ = get_ocr_reader()
    except Exception as exc:
        logger.warning("OCR initialization failed: %s", exc)
        return PlateOCREvidence(ocr_status="OCR INITIALIZATION FAILURE")

    try:
        results = reader.readtext(crop)
    except Exception as exc:
        logger.warning("OCR inference failed on crop: %s", exc)
        return PlateOCREvidence(ocr_status="OCR INFERENCE FAILURE")

    if not results:
        return PlateOCREvidence(ocr_status="NO OCR RESULTS")

    best_plate: Optional[str] = None
    best_plate_conf: Optional[float] = None
    best_raw_text: Optional[str] = None
    best_raw_conf: Optional[float] = None
    candidate_count = 0

    # EasyOCR results format: [(bbox, text, prob), ...]
    for _, text, prob in results:
        if not text:
            continue
        candidate_count += 1
        conf = float(prob)

        # Keep the highest-confidence raw text exactly as EasyOCR emitted it.
        if best_raw_conf is None or conf > best_raw_conf:
            best_raw_text = str(text)
            best_raw_conf = round(conf, 4)

        norm = normalize_plate_text(str(text))
        if norm is not None:
            # Pick highest confidence valid plate candidate; never auto-correct it.
            # When multiple candidates exist, prefer full registration numbers (having both
            # letters and digits) over partial 3-letter country prefixes like 'IND'.
            norm_has_ld = bool(re.search(r"[A-Z]", norm) and re.search(r"[0-9]", norm))
            best_has_ld = bool(best_plate and re.search(r"[A-Z]", best_plate) and re.search(r"[0-9]", best_plate))
            if (norm_has_ld and not best_has_ld) or (
                norm_has_ld == best_has_ld and (best_plate_conf is None or conf > best_plate_conf)
            ):
                best_plate = norm
                best_plate_conf = round(conf, 4)

    return PlateOCREvidence(
        plate=best_plate,
        plate_confidence=best_plate_conf,
        raw_text=best_raw_text,
        raw_confidence=best_raw_conf,
        candidate_count=candidate_count,
        ocr_status="OK",
    )


def read_plate(crop: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
    """Read license plate text from a vehicle image crop.

    Args:
        crop: NumPy array representing the cropped vehicle image (BGR or RGB).

    Returns:
        (plate_text, confidence): Normalized plate text and real OCR confidence,
                                  or (None, None) if nothing readable is detected.
    """
    evidence = read_plate_evidence(crop)
    return evidence.plate, evidence.plate_confidence
