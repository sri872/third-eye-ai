"""Classical Template-Based Character Recognition.

Matches segmented character bitmaps against standardized alphanumeric templates
without using neural networks, deep learning models, or external OCR engines.
"""

import os
from typing import Dict, Tuple, List, Optional
import cv2
import numpy as np


TEMPLATE_WIDTH = 24
TEMPLATE_HEIGHT = 32

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"


class TemplateRecognizer:
    """Deterministic template-based character recognizer for alphanumeric plate characters."""

    def __init__(self, template_dir: Optional[str] = None):
        if template_dir is None:
            # Default to project template directory
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            template_dir = os.path.abspath(
                os.path.join(curr_dir, "..", "..", "data", "ocr_templates")
            )

        self.template_dir = template_dir
        self.letter_templates: Dict[str, np.ndarray] = {}
        self.digit_templates: Dict[str, np.ndarray] = {}
        self._load_or_generate_templates()

    def _load_or_generate_templates(self) -> None:
        """Load templates from disk or synthesize them if not yet generated."""
        letters_path = os.path.join(self.template_dir, "letters")
        digits_path = os.path.join(self.template_dir, "digits")

        os.makedirs(letters_path, exist_ok=True)
        os.makedirs(digits_path, exist_ok=True)

        for ch in LETTERS:
            p = os.path.join(letters_path, f"{ch}.png")
            if os.path.exists(p):
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    self.letter_templates[ch] = cv2.resize(
                        bin_img, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT)
                    )
            if ch not in self.letter_templates:
                synth = self._synthesize_char_template(ch)
                self.letter_templates[ch] = synth
                cv2.imwrite(p, synth)

        for d in DIGITS:
            p = os.path.join(digits_path, f"{d}.png")
            if os.path.exists(p):
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    self.digit_templates[d] = cv2.resize(
                        bin_img, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT)
                    )
            if d not in self.digit_templates:
                synth = self._synthesize_char_template(d)
                self.digit_templates[d] = synth
                cv2.imwrite(p, synth)

    def _synthesize_char_template(self, char: str) -> np.ndarray:
        """Synthesize high-contrast binary template for a character."""
        canvas = np.zeros((TEMPLATE_HEIGHT * 3, TEMPLATE_WIDTH * 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(char, font, 2.7, 5)
        x = (canvas.shape[1] - tw) // 2
        y = (canvas.shape[0] + th) // 2 - baseline
        cv2.putText(canvas, char, (x, y), font, 2.7, 255, 5, cv2.LINE_AA)

        coords = cv2.findNonZero(canvas)
        if coords is not None:
            bx, by, bw, bh = cv2.boundingRect(coords)
            char_crop = canvas[by : by + bh, bx : bx + bw]
            pad_y = max(1, int(bh * 0.10))
            pad_x = max(1, int(bw * 0.10))
            padded = cv2.copyMakeBorder(
                char_crop, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0
            )
            resized = cv2.resize(
                padded, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT), interpolation=cv2.INTER_AREA
            )
            _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
            return binary
        else:
            return cv2.resize(canvas, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT))

    def match_character(
        self,
        char_img: np.ndarray,
        prior: str = "any",
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Match character image against templates with optional positional prior.

        Args:
            char_img: Normalized binary character image (24x32).
            prior: 'letter' (test only A-Z), 'digit' (test only 0-9), or 'any'.

        Returns:
            Tuple of:
                best_char: Best matching character symbol.
                best_similarity: Match similarity score in range [0.0, 1.0].
                top_candidates: List of top 3 (char, score) tuples.
        """
        if char_img is None or char_img.size == 0:
            return "?", 0.0, []

        if char_img.shape != (TEMPLATE_HEIGHT, TEMPLATE_WIDTH):
            char_img = cv2.resize(
                char_img, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT), interpolation=cv2.INTER_AREA
            )
            _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)

        targets: Dict[str, np.ndarray] = {}
        if prior == "letter":
            targets = self.letter_templates
        elif prior == "digit":
            targets = self.digit_templates
        else:
            targets = {**self.letter_templates, **self.digit_templates}

        scores: List[Tuple[str, float]] = []

        char_bool = char_img > 127

        for ch, tmpl in targets.items():
            tmpl_bool = tmpl > 127

            # 1. Normalized cross correlation via template matching
            res = cv2.matchTemplate(char_img, tmpl, cv2.TM_CCOEFF_NORMED)
            corr = float(res[0][0]) if res is not None else 0.0
            norm_corr = max(0.0, (corr + 1.0) / 2.0)  # Map [-1, 1] to [0, 1]

            # 2. Binary Jaccard IoU similarity
            intersection = np.logical_and(char_bool, tmpl_bool).sum()
            union = np.logical_or(char_bool, tmpl_bool).sum()
            iou = float(intersection) / float(union) if union > 0 else 0.0

            # Combined similarity metric
            sim = 0.65 * norm_corr + 0.35 * iou
            scores.append((ch, round(sim, 4)))

        scores.sort(key=lambda s: s[1], reverse=True)

        if not scores:
            return "?", 0.0, []

        best_char, best_score = scores[0]
        top_candidates = scores[:3]
        return best_char, best_score, top_candidates
