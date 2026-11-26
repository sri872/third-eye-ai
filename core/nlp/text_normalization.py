"""Classical NLP Text Normalization.

Cleans and normalizes emergency incident text using deterministic regular expressions
and standard Python string manipulation without neural language models.
"""

import re
from typing import List, Dict


# Common phonetic and typographical errors in transcribed emergency calls
TYPO_CORRECTIONS = {
    "acident": "accident",
    "accidnt": "accident",
    "acidentt": "accident",
    "vehical": "vehicle",
    "vehicl": "vehicle",
    "ambulence": "ambulance",
    "ambulanse": "ambulance",
    "colision": "collision",
    "collison": "collision",
    "pedistrian": "pedestrian",
    "hiway": "highway",
    "hi-way": "highway",
    "injuired": "injured",
    "bleedng": "bleeding",
    "unconcious": "unconscious",
    "unconsious": "unconscious",
    "overturnd": "overturned",
}


def normalize_text(text: str) -> str:
    """Normalize whitespace, punctuation, and known typographical errors."""
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase for uniform processing
    normalized = text.lower().strip()

    # Normalize multiple whitespace, newlines, and tabs
    normalized = re.sub(r"\s+", " ", normalized)

    # Word-level typo correction
    words = normalized.split()
    corrected_words = [TYPO_CORRECTIONS.get(w.strip(".,!?;:()"), w) for w in words]
    return " ".join(corrected_words)


def tokenize_words(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using standard punctuation delimiters."""
    if not text:
        return []
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]
