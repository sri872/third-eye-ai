"""Classical Plate Reader Pipeline.

Orchestrates preprocessing, character segmentation, template matching,
and Indian format validation for license plate recognition.
"""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from core.ocr.preprocessing import preprocess_plate
from core.ocr.character_segmentation import segment_characters
from core.ocr.template_recognition import TemplateRecognizer
from core.ocr.plate_validation import validate_indian_plate


class ClassicalPlateReader:
    """End-to-end classical OCR reader for Indian number plates."""

    def __init__(self, template_dir: Optional[str] = None, verification_threshold: float = 0.70):
        self.recognizer = TemplateRecognizer(template_dir=template_dir)
        self.verification_threshold = verification_threshold

    def read_plate(
        self,
        plate_bgr: np.ndarray,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read text from license plate image using classical OCR.

        Args:
            plate_bgr: Candidate license plate image crop.
            ground_truth: Optional ground truth string for evaluation.

        Returns:
            Dictionary of OCR results.
        """
        if plate_bgr is None or plate_bgr.size == 0:
            return {
                "status": "NO_PLATE_DETECTED",
                "raw_text": "",
                "normalized_text": "",
                "mean_similarity": 0.0,
                "character_count": 0,
                "characters": [],
                "validation": validate_indian_plate(""),
                "is_verified": False,
                "ground_truth": ground_truth,
                "exact_match": False,
            }

        # 1. Preprocess
        try:
            enhanced_gray, binary_plate = preprocess_plate(plate_bgr)
        except Exception as e:
            return {
                "status": "UNREADABLE",
                "error": f"Preprocessing failed: {str(e)}",
                "raw_text": "",
                "normalized_text": "",
                "mean_similarity": 0.0,
                "character_count": 0,
                "characters": [],
                "validation": validate_indian_plate(""),
                "is_verified": False,
                "ground_truth": ground_truth,
                "exact_match": False,
            }

        # 2. Segment characters
        segmented = segment_characters(binary_plate)

        if not segmented:
            return {
                "status": "UNREADABLE",
                "raw_text": "",
                "normalized_text": "",
                "mean_similarity": 0.0,
                "character_count": 0,
                "characters": [],
                "validation": validate_indian_plate(""),
                "is_verified": False,
                "ground_truth": ground_truth,
                "exact_match": False,
                "binary_plate": binary_plate,
            }

        # 3. Template matching with positional prior heuristic
        # If candidate count is ~10, use standard Indian plate positional priors:
        # Pos 0, 1: State code letters
        # Pos 2, 3: RTO digits
        # Pos 4, 5: Series letters
        # Pos 6, 7, 8, 9: Digits
        num_chars = len(segmented)
        use_priors = (num_chars == 10)

        recognized_chars: List[Dict[str, Any]] = []
        raw_chars: List[str] = []
        similarities: List[float] = []

        for idx, (char_img, bbox) in enumerate(segmented):
            prior = "any"
            if use_priors:
                if idx in (0, 1):
                    prior = "letter"
                elif idx in (2, 3):
                    prior = "digit"
                elif idx in (4, 5):
                    prior = "letter"
                elif idx >= 6:
                    prior = "digit"

            ch, sim, top3 = self.recognizer.match_character(char_img, prior=prior)
            raw_chars.append(ch)
            similarities.append(sim)
            recognized_chars.append(
                {
                    "position": idx,
                    "character": ch,
                    "similarity": sim,
                    "bbox": list(bbox),
                    "top_candidates": top3,
                    "char_image": char_img,
                }
            )

        raw_text = "".join(raw_chars)
        mean_sim = round(float(np.mean(similarities)), 4) if similarities else 0.0

        # 4. Format validation
        validation = validate_indian_plate(raw_text)

        # 5. Verification status determination
        # Never mark verified if mean similarity < threshold or format is invalid
        if validation["is_valid"] and mean_sim >= self.verification_threshold:
            status = "VERIFIED"
            is_verified = True
        elif validation["is_valid"] or mean_sim >= 0.50:
            status = "UNVERIFIED"
            is_verified = False
        else:
            status = "UNREADABLE"
            is_verified = False

        normalized_text = validation["normalized"] if validation["is_valid"] else raw_text

        # 6. Evaluation against ground truth if provided
        exact_match = False
        if ground_truth:
            gt_clean = ground_truth.replace(" ", "").upper()
            exact_match = (normalized_text == gt_clean)

        return {
            "status": status,
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "mean_similarity": mean_sim,
            "character_count": len(raw_chars),
            "characters": recognized_chars,
            "validation": validation,
            "is_verified": is_verified,
            "ground_truth": ground_truth,
            "exact_match": exact_match,
            "binary_plate": binary_plate,
            "enhanced_gray": enhanced_gray,
        }
