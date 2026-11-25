"""License plate reading module for Project Third-Eye.

Extracts text candidates from plate regions or legacy whole-vehicle crops using
EasyOCR with lazy loading. Normalizes plate text (uppercase, alphanumeric only)
without fabricating characters. Returns real OCR confidence or (None, None) when
unreadable, while preserving the exact raw OCR text and its status.

A small deterministic set of preprocessing variants is evaluated so a slightly
noisy crop does not lose a real read. Results are memoized per reader/image so
callers that inspect both the processed and the raw crop never pay for the same
OCR inference twice. Format validity is only a ranking hint: verification remains
the UI's ``>= 0.70`` real-OCR-confidence gate.
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import re
from typing import Optional, Tuple
import weakref

import cv2
import easyocr
import numpy as np

logger = logging.getLogger("third_eye.ocr.plate_reader")

# Project root models directory for isolated local storage
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "easyocr"
USER_NETWORK_DIR = MODEL_DIR / "user_network"

# Verification threshold used by the UI. Kept here so module-level checks agree
# with the UI; a format-valid string is never verified on its own.
PLATE_TRUST_THRESHOLD = 0.7

_READER: Optional[easyocr.Reader] = None
_DEVICE: str = "cpu"

# (reader -> OrderedDict[image sha256, raw EasyOCR results]).
# Weak keys prevent stale mock readers from leaking between tests and avoid
# re-running EasyOCR on the same pixels (the UI evaluates variants on both the
# processed and the raw crop).
_OCR_RESULT_CACHE: "weakref.WeakKeyDictionary[object, OrderedDict]" = weakref.WeakKeyDictionary()
_OCR_CACHE_MAX_ENTRIES = 64


@dataclass(frozen=True)
class PlateOCREvidence:
    """Real OCR output for one vehicle crop, with no inferred characters.

    ``plate`` / ``plate_confidence`` are the same values returned by ``read_plate``.
    ``raw_text`` / ``raw_confidence`` keep the highest-confidence text that EasyOCR
    actually emitted, even when normalization rejects it. This lets the UI show
    recognized text honestly without presenting it as a verified registration.
    Verification itself stays in the UI at PLATE_TRUST_THRESHOLD = 0.7.

    ``format_valid`` and ``preprocessing_variant`` are appended with defaults so
    older callers constructing this dataclass positionally keep working.
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
    format_valid: bool = False
    preprocessing_variant: Optional[str] = None


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


def is_verified_plate_confidence(confidence: Optional[float]) -> bool:
    """Apply the same ``>= 0.70`` real-confidence gate used by the UI.

    Registration-format validity is deliberately ignored here: a beautifully
    formatted read at 0.69 is still unverified.
    """
    if confidence is None:
        return False
    try:
        return float(confidence) >= PLATE_TRUST_THRESHOLD
    except (TypeError, ValueError):
        return False


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


def is_valid_indian_plate_format(candidate: Optional[str]) -> bool:
    """Validate whether a candidate matches Indian registration plate patterns.

    This is a *ranking hint only*, never evidence of verification:
    - Standard: 2 state letters + 1-2 district digits + 1-3 series letters + 4 digits.
    - Bharat series (BH): 2 year digits + BH + 4 digits + 1-2 series letters.
    """
    if not candidate or not isinstance(candidate, str):
        return False
    norm = candidate.upper().strip()
    if re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$", norm):
        return True
    if re.match(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$", norm):
        return True
    return False


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


def generate_plate_preprocessing_variants(region: np.ndarray) -> dict[str, np.ndarray]:
    """Build a small, deterministic set of non-fabricating OCR variants.

    Order is stable so ranking stays reproducible:
    ``raw`` (untouched), ``padded`` (edge replication), ``contrast_enhanced``
    (the existing mild CLAHE pipeline), ``sharpened`` (light unsharp mask) and
    ``deskew`` only when a small, reliable rotation is detected. No image is
    binarised and no character is added, removed or corrected.
    """
    if region is None or not isinstance(region, np.ndarray) or region.size == 0:
        return {}

    variants: dict[str, np.ndarray] = {"raw": region}

    height, width = region.shape[:2]
    pad_y = max(4, int(round(height * 0.1)))
    pad_x = max(6, int(round(width * 0.1)))
    variants["padded"] = cv2.copyMakeBorder(
        region, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
    )

    try:
        variants["contrast_enhanced"] = preprocess_plate_region(region)
    except Exception as exc:
        logger.warning("Contrast-enhanced plate variant unavailable: %s", exc)

    try:
        blur = cv2.GaussianBlur(region, (0, 0), 3)
        variants["sharpened"] = cv2.addWeighted(region, 1.3, blur, -0.3, 0)
    except Exception as exc:
        logger.warning("Sharpened plate variant unavailable: %s", exc)

    # Deskew is only safe for a small, well-supported angle; large rotations can
    # clip characters, so they are left to the raw/padded variants.
    try:
        gray = (
            cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            if region.ndim == 3 and region.shape[2] >= 3
            else region
        )
        points = cv2.findNonZero(cv2.Canny(gray, 50, 150))
        if points is not None and len(points) > 20:
            angle = cv2.minAreaRect(points)[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            if 0.5 <= abs(angle) <= 15.0:
                matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
                variants["deskew"] = cv2.warpAffine(
                    region,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
    except Exception as exc:
        logger.warning("Deskew plate variant unavailable: %s", exc)

    return variants


def _score_candidate(norm: str, confidence: float) -> tuple[int, float]:
    """Rank candidates: registration format, then alphanumeric, then partial.

    Within a tier the *real* OCR confidence wins; no confidence is invented.
    """
    tier = 2 if is_valid_indian_plate_format(norm) else (
        1 if re.search(r"[A-Z]", norm) and re.search(r"[0-9]", norm) else 0
    )
    return tier, confidence


def _readtext_cached(reader, image: np.ndarray):
    """Run ``readtext`` once per (reader, exact pixels), regardless of variant."""
    digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    try:
        cache = _OCR_RESULT_CACHE.get(reader)
    except TypeError:
        # Unhashable reader: fall back to a plain OCR call.
        return reader.readtext(image)
    if cache is None:
        cache = OrderedDict()
        _OCR_RESULT_CACHE[reader] = cache
    if digest in cache:
        cache.move_to_end(digest)
        return cache[digest]
    results = reader.readtext(image)
    cache[digest] = results
    while len(cache) > _OCR_CACHE_MAX_ENTRIES:
        cache.popitem(last=False)
    return results


def read_plate_evidence(
    crop: np.ndarray, evaluate_variants: bool = True
) -> PlateOCREvidence:
    """Collect real EasyOCR evidence from a plate region without guessing text.

    ``evaluate_variants`` controls the deterministic variant drill-down; it has a
    default so existing single-argument callers and monkeypatches keep working.
    No confidence threshold is applied here: the exact recognized text and its
    real confidence are preserved for the UI's 0.70 verification gate.
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

    variants = (
        generate_plate_preprocessing_variants(crop)
        if evaluate_variants
        else {"raw": crop}
    )
    if not variants:
        variants = {"raw": crop}

    best_plate: Optional[str] = None
    best_plate_conf: Optional[float] = None
    best_raw_text: Optional[str] = None
    best_raw_conf: Optional[float] = None
    best_variant: Optional[str] = None
    best_score: tuple[int, float] = (-1, -1.0)
    # Count distinct real reads, not the same text repeated by several variants.
    seen_candidates: set[tuple[str, float]] = set()
    had_any_results = False
    inference_errors = 0

    for variant_name, image in variants.items():
        try:
            results = _readtext_cached(reader, image)
        except Exception as exc:
            logger.warning("OCR inference failed on variant %s: %s", variant_name, exc)
            inference_errors += 1
            continue

        if not results:
            continue
        had_any_results = True

        # EasyOCR results format: [(bbox, text, prob), ...]
        for _, text, prob in results:
            if not text:
                continue
            conf = float(prob)
            seen_candidates.add((str(text), conf))

            # Keep the exact highest-confidence text EasyOCR emitted, unrounded.
            if best_raw_conf is None or conf > best_raw_conf:
                best_raw_text = str(text)
                best_raw_conf = conf

            norm = normalize_plate_text(str(text))
            if norm is None:
                continue
            score = _score_candidate(norm, conf)
            if score > best_score:
                best_score = score
                best_plate = norm
                best_plate_conf = conf
                best_variant = variant_name

    if not had_any_results:
        if inference_errors == len(variants):
            return PlateOCREvidence(ocr_status="OCR INFERENCE FAILURE")
        return PlateOCREvidence(ocr_status="NO OCR RESULTS")

    return PlateOCREvidence(
        plate=best_plate,
        plate_confidence=best_plate_conf,
        raw_text=best_raw_text,
        raw_confidence=best_raw_conf,
        candidate_count=len(seen_candidates),
        ocr_status="OK",
        format_valid=bool(best_plate and is_valid_indian_plate_format(best_plate)),
        preprocessing_variant=best_variant,
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
