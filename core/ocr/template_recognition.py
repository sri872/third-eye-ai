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
        self.letter_template_bank: Dict[str, List[np.ndarray]] = {}
        self.digit_template_bank: Dict[str, List[np.ndarray]] = {}
        self._load_or_generate_templates()
        self._build_template_banks()

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

    def _synthesize_char_template(
        self,
        char: str,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        scale: float = 2.7,
        thickness: int = 5,
    ) -> np.ndarray:
        """Render and normalize one deterministic OpenCV glyph template."""
        canvas = np.zeros((TEMPLATE_HEIGHT * 3, TEMPLATE_WIDTH * 3), dtype=np.uint8)
        (tw, th), baseline = cv2.getTextSize(char, font, scale, thickness)
        x = max(0, (canvas.shape[1] - tw) // 2)
        y = max(th, (canvas.shape[0] + th) // 2 - baseline)
        cv2.putText(
            canvas, char, (x, y), font, scale, 255, thickness, cv2.LINE_AA
        )

        coords = cv2.findNonZero(canvas)
        if coords is None:
            return np.zeros((TEMPLATE_HEIGHT, TEMPLATE_WIDTH), dtype=np.uint8)

        bx, by, bw, bh = cv2.boundingRect(coords)
        char_crop = canvas[by:by + bh, bx:bx + bw]
        pad_y = max(1, int(bh * 0.10))
        pad_x = max(1, int(bw * 0.10))
        padded = cv2.copyMakeBorder(
            char_crop, pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_CONSTANT, value=0,
        )
        resized = cv2.resize(
            padded, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        return binary

    def _build_template_banks(self) -> None:
        """Build several deterministic font/weight variants for each glyph.

        Templates are generic OpenCV-rendered examples, not real-world plate
        training data. Multiple variants reduce dependence on one font weight.
        """
        fonts = (
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_PLAIN,
        )
        scales_and_weights = (
            (1.8, 2), (2.2, 2), (2.4, 3), (2.7, 4), (2.7, 5),
        )
        for glyph, base in self.letter_templates.items():
            variants = [base]
            for font in fonts:
                for scale, thickness in scales_and_weights:
                    candidate = self._synthesize_char_template(
                        glyph, font=font, scale=scale, thickness=thickness
                    )
                    if not any(np.array_equal(candidate, item) for item in variants):
                        variants.append(candidate)
            self.letter_template_bank[glyph] = variants

        for glyph, base in self.digit_templates.items():
            variants = [base]
            for font in fonts:
                for scale, thickness in scales_and_weights:
                    candidate = self._synthesize_char_template(
                        glyph, font=font, scale=scale, thickness=thickness
                    )
                    if not any(np.array_equal(candidate, item) for item in variants):
                        variants.append(candidate)
            self.digit_template_bank[glyph] = variants

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

        targets: Dict[str, List[np.ndarray]] = {}
        if prior == "letter":
            targets = self.letter_template_bank
        elif prior == "digit":
            targets = self.digit_template_bank
        else:
            targets = {**self.letter_template_bank, **self.digit_template_bank}

        scores: List[Tuple[str, float]] = []
        char_bool = char_img > 127

        for glyph, templates in targets.items():
            best_similarity = 0.0
            for template in templates:
                template_bool = template > 127

                # Normalized cross-correlation captures stroke arrangement.
                correlation = cv2.matchTemplate(
                    char_img, template, cv2.TM_CCOEFF_NORMED
                )
                corr = float(correlation[0][0]) if correlation is not None else 0.0
                normalized_corr = max(0.0, (corr + 1.0) / 2.0)

                # Binary Jaccard overlap adds a shape-level comparison.
                intersection = np.logical_and(char_bool, template_bool).sum()
                union = np.logical_or(char_bool, template_bool).sum()
                jaccard = float(intersection) / float(union) if union else 0.0

                similarity = 0.65 * normalized_corr + 0.35 * jaccard
                best_similarity = max(best_similarity, similarity)

            scores.append((glyph, round(best_similarity, 4)))

        scores.sort(key=lambda s: s[1], reverse=True)

        if not scores:
            return "?", 0.0, []

        best_char, best_score = scores[0]
        top_candidates = scores[:3]
        return best_char, best_score, top_candidates
