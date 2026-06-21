"""Raw OCR output — the independent signal the critic checks extraction against.

The extractor reads the image with a vision model; OCR reads the same image with
a different engine. When the model reports a value that never appears in the OCR
text, that value was most likely hallucinated. The two sources disagreeing is the
cheapest, strongest anti-hallucination signal we have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NON_ALPHANUMERIC = re.compile(r"[^0-9a-z]")

# Characters Tesseract routinely confuses on real scans, each folded to one
# canonical form. Without this, a correct ``07AAECS...`` read as ``O7AAECS...``
# would look like a hallucination and hold a perfectly good invoice.
_CONFUSABLES = str.maketrans(
    {
        "o": "0",
        "l": "1",
        "i": "1",
        "s": "5",
        "b": "8",
        "z": "2",
        "g": "6",
    }
)


@dataclass(frozen=True, slots=True)
class WordBox:
    text: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: float


@dataclass(frozen=True, slots=True)
class RawText:
    """Full OCR text plus per-word boxes for the page(s)."""

    text: str
    words: tuple[WordBox, ...] = ()

    def contains(self, needle: str) -> bool:
        """Whether ``needle`` appears in the OCR text, tolerant of formatting and
        common OCR misreads.

        Both sides are reduced to bare alphanumerics with confusable characters
        folded, so ``1,84,200`` matches ``184200`` and a GSTIN whose leading ``0``
        was scanned as ``O`` still matches — we test the value itself, not its
        incidental formatting or the OCR engine's character-level noise.
        """
        normalised_needle = fold_confusables(needle)
        return bool(normalised_needle) and normalised_needle in fold_confusables(self.text)

    def contains_most_tokens(self, value: str, min_ratio: float = 0.5) -> bool:
        """Whether most of a multi-word value's significant tokens appear on the page.

        Exact substring matching is too strict for names: OCR splits, reorders and
        abbreviates ``"Sinclair Broadcast c/o WSTM"`` so the whole string rarely
        survives, yet the key tokens do. Requiring a share of the tokens tolerates
        that while still rejecting a value (a swapped vendor) that shares nothing
        with the document.
        """
        tokens = [t for word in value.split() if len(t := fold_confusables(word)) >= 3]
        if not tokens:
            return self.contains(value)
        haystack = fold_confusables(self.text)
        present = sum(1 for token in tokens if token in haystack)
        return present / len(tokens) >= min_ratio


def canonical(value: str) -> str:
    """Lowercased alphanumerics only — formatting stripped, characters unchanged."""
    return _NON_ALPHANUMERIC.sub("", value.lower())


def fold_confusables(value: str) -> str:
    """``canonical`` with OCR-confusable characters folded to one form, so a value
    misread as ``O7AAECS`` still matches ``07AAECS``."""
    return canonical(value).translate(_CONFUSABLES)
