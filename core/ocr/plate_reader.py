"""Classical plate reading using multi-variant thresholding and glyph templates.

Recognition scores are heuristic similarity measures, not calibrated probabilities.
No neural OCR or learned detector is used.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from core.ocr.preprocessing import preprocess_plate_variants
from core.ocr.character_segmentation import segment_characters
from core.ocr.template_recognition import TemplateRecognizer
from core.ocr.plate_validation import validate_indian_plate


class ClassicalPlateReader:
    """Deterministic Indian number-plate OCR with multiple CV preprocessing paths."""

    def __init__(
        self,
        template_dir: Optional[str] = None,
        verification_threshold: float = 0.70,
    ):
        self.recognizer = TemplateRecognizer(template_dir=template_dir)
        self.verification_threshold = verification_threshold

    def _recognize_binary(self, binary: np.ndarray) -> Dict[str, Any]:
        segmented = segment_characters(binary)
        recognized_chars: List[Dict[str, Any]] = []
        raw_chars: List[str] = []
        similarities: List[float] = []
        use_priors = len(segmented) == 10

        for idx, (char_img, bbox) in enumerate(segmented):
            prior = "any"
            if use_priors:
                prior = "letter" if idx in (0, 1, 4, 5) else "digit"
            char, similarity, top3 = self.recognizer.match_character(
                char_img, prior=prior
            )
            raw_chars.append(char)
            similarities.append(similarity)
            recognized_chars.append({
                "position": idx,
                "character": char,
                "similarity": similarity,
                "bbox": list(bbox),
                "top_candidates": top3,
                "char_image": char_img,
            })

        raw_text = "".join(raw_chars)
        mean_similarity = (
            round(float(np.mean(similarities)), 4) if similarities else 0.0
        )
        validation = validate_indian_plate(raw_text)
        char_count = len(recognized_chars)
        # Indian private-vehicle plates commonly use 10 characters. Treat this
        # only as a ranking hint; other legal formats and incomplete crops exist.
        count_fit = max(0.0, 1.0 - abs(char_count - 10) / 10.0)
        rank_score = (
            0.55 * mean_similarity
            + 0.20 * count_fit
            + (0.25 if validation["is_valid"] else 0.0)
        )
        return {
            "raw_text": raw_text,
            "mean_similarity": mean_similarity,
            "character_count": char_count,
            "characters": recognized_chars,
            "validation": validation,
            "rank_score": round(rank_score, 4),
            "binary_plate": binary,
        }

    def read_plate(
        self,
        plate_bgr: np.ndarray,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read a plate crop, trying multiple deterministic preprocessing variants."""
        if plate_bgr is None or getattr(plate_bgr, "size", 0) == 0:
            return self._empty_result("NO_PLATE_DETECTED", ground_truth)

        try:
            variants = preprocess_plate_variants(plate_bgr)
        except Exception as exc:
            result = self._empty_result("UNREADABLE", ground_truth)
            result["error"] = f"Preprocessing failed: {exc}"
            return result

        attempts: List[Dict[str, Any]] = []
        for method, enhanced_gray, binary in variants:
            outcome = self._recognize_binary(binary)
            outcome["preprocessing_method"] = method
            outcome["enhanced_gray"] = enhanced_gray
            attempts.append(outcome)

        if not attempts or max(item["character_count"] for item in attempts) == 0:
            result = self._empty_result("UNREADABLE", ground_truth)
            result["threshold_variants_tried"] = [item[0] for item in variants]
            if attempts:
                result["binary_plate"] = attempts[0]["binary_plate"]
                result["enhanced_gray"] = attempts[0]["enhanced_gray"]
            return result

        # Prefer syntactically valid candidates; then rank by glyph similarity
        # and a weak expected-length prior.
        best = max(
            attempts,
            key=lambda item: (
                bool(item["validation"]["is_valid"]),
                item["rank_score"],
                item["mean_similarity"],
            ),
        )
        validation = best["validation"]
        mean_similarity = best["mean_similarity"]

        if validation["is_valid"] and mean_similarity >= self.verification_threshold:
            status, is_verified = "VERIFIED", True
        elif validation["is_valid"] or mean_similarity >= 0.50:
            status, is_verified = "UNVERIFIED", False
        else:
            status, is_verified = "UNREADABLE", False

        normalized_text = (
            validation["normalized"] if validation["is_valid"] else best["raw_text"]
        )
        exact_match = False
        if ground_truth:
            gt_clean = "".join(ch for ch in ground_truth.upper() if ch.isalnum())
            exact_match = normalized_text == gt_clean

        return {
            "status": status,
            "raw_text": best["raw_text"],
            "normalized_text": normalized_text,
            "mean_similarity": mean_similarity,
            "character_count": best["character_count"],
            "characters": best["characters"],
            "validation": validation,
            "is_verified": is_verified,
            "ground_truth": ground_truth,
            "exact_match": exact_match,
            "binary_plate": best["binary_plate"],
            "enhanced_gray": best["enhanced_gray"],
            "preprocessing_method": best["preprocessing_method"],
            "variant_attempts": [
                {
                    "method": item["preprocessing_method"],
                    "raw_text": item["raw_text"],
                    "character_count": item["character_count"],
                    "mean_similarity": item["mean_similarity"],
                    "format_valid": item["validation"]["is_valid"],
                    "rank_score": item["rank_score"],
                }
                for item in attempts
            ],
            "similarity_is_probability": False,
        }

    @staticmethod
    def _empty_result(status: str, ground_truth: Optional[str]) -> Dict[str, Any]:
        return {
            "status": status,
            "raw_text": "",
            "normalized_text": "",
            "mean_similarity": 0.0,
            "character_count": 0,
            "characters": [],
            "validation": validate_indian_plate(""),
            "is_verified": False,
            "ground_truth": ground_truth,
            "exact_match": False,
            "similarity_is_probability": False,
        }
