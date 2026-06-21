from __future__ import annotations

import pytest

from apverify.domain.ocr import RawText, canonical, fold_confusables


def test_canonical_strips_punctuation_and_case_without_folding() -> None:
    assert canonical("INV-1001") == "inv1001"
    assert canonical("1,84,200") == "184200"


def test_fold_confusables_folds_ocr_lookalikes() -> None:
    # l->1, o->0, so an OCR misread of INV-1001 collapses onto the same key.
    assert fold_confusables("INV-l00l") == fold_confusables("INV-1001")
    assert fold_confusables("INV-1001") == "1nv1001"


@pytest.mark.parametrize(
    ("needle", "text"),
    [
        ("184200", "Total INR 1,84,200.00"),
        ("27AABCU9603R1ZN", "GSTIN: 27AABCU9603R1ZN"),
        # leading 0 scanned as the letter O — the confusable fold must absorb it
        ("07AAECS5678K1ZR", "GSTIN O7AAECS5678K1ZR"),
    ],
)
def test_contains_matches_through_formatting_and_ocr_confusables(needle: str, text: str) -> None:
    assert RawText(text=text).contains(needle)


def test_contains_still_rejects_a_genuinely_absent_value() -> None:
    assert not RawText(text="Total 184200").contains("999999")


def test_contains_rejects_empty_needle() -> None:
    assert not RawText(text="anything").contains("")


def test_contains_most_tokens_tolerates_ocr_mangling() -> None:
    page = RawText(text="REMIT TO Sinclair Broadcast c/o WSTM accounting dept")
    # The full string never survives OCR, but the significant tokens do.
    assert page.contains_most_tokens("Sinclair Broadcast Group")


def test_contains_most_tokens_rejects_an_unrelated_value() -> None:
    page = RawText(text="Sinclair Broadcast c/o WSTM")
    assert not page.contains_most_tokens("Phantom Holdings International")


def test_contains_most_tokens_falls_back_for_short_values() -> None:
    assert RawText(text="invoice no AB").contains_most_tokens("AB")
