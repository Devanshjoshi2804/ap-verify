from __future__ import annotations

import pytest
from tests.support import build_invoice

from apverify.eval import accuracy
from apverify.eval.accuracy import Outcome, aggregate, score_document


def _truth(**overrides: str) -> dict[str, str]:
    base = {
        accuracy.VENDOR: "ACME Steel Pvt Ltd",
        accuracy.DATE: "2025-06-04",
        accuracy.CURRENCY: "INR",
        accuracy.SUBTOTAL: "156100",
        accuracy.TAX: "28100",
        accuracy.TOTAL: "184200",
    }
    base.update(overrides)
    return base


def test_perfect_extraction_matches_every_field() -> None:
    # build_invoice's defaults already equal the truth above.
    scored = score_document(build_invoice(invoice_date="04-06-2025"), _truth())
    assert set(scored.values()) == {Outcome.MATCH}


def test_wrong_total_is_a_mismatch() -> None:
    scored = score_document(build_invoice(), _truth(total="999999"))
    assert scored[accuracy.TOTAL] is Outcome.MISMATCH


def test_absent_prediction_is_a_miss() -> None:
    scored = score_document(build_invoice(vendor_name=""), _truth())
    assert scored[accuracy.VENDOR] is Outcome.MISSED


def test_vendor_match_is_fuzzy() -> None:
    predicted = build_invoice(vendor_name="ACME Steel Pvt Ltd")
    scored = score_document(predicted, _truth(vendor="ACME Steel Pvt Ltd."))
    assert scored[accuracy.VENDOR] is Outcome.MATCH


def test_vendor_legal_suffix_only_difference_matches() -> None:
    # The model read the name correctly but dropped the legal suffix.
    assert accuracy.value_matches(accuracy.VENDOR, "PHILIP MORRIS", "PHILIP MORRIS INCORPORATED")


def test_vendor_superset_with_trailing_tokens_matches() -> None:
    # The prediction contains the whole ground-truth name plus extra trailing words.
    assert accuracy.value_matches(
        accuracy.VENDOR,
        "DISPLAY TECHNOLOGIES LLC CUSTOM GROUP",
        "DISPLAY TECHNOLOGIES LLC",
    )


def test_vendor_station_call_sign_suffix_matches() -> None:
    assert accuracy.value_matches(accuracy.VENDOR, "KGMB", "KGMB TV")


def test_vendor_brand_subset_matches() -> None:
    assert accuracy.value_matches(accuracy.VENDOR, "Bozell", "BOZELL WORLDWIDE, INC.")


def test_unrelated_vendor_names_still_mismatch() -> None:
    assert not accuracy.value_matches(accuracy.VENDOR, "WTTG", "American Cancer Society")


def test_short_token_does_not_falsely_contain() -> None:
    # A two-letter fragment must not count as containing a real multi-word name.
    assert not accuracy.value_matches(accuracy.VENDOR, "AB", "AB Global Logistics Partners")


def test_date_match_is_format_agnostic() -> None:
    scored = score_document(build_invoice(invoice_date="04-06-2025"), _truth(date="2025-06-04"))
    assert scored[accuracy.DATE] is Outcome.MATCH


def test_currency_symbol_matches_iso_code() -> None:
    # DocILE labels currency as "$"; the extractor returns "USD" — both correct.
    scored = score_document(build_invoice(currency="USD"), _truth(currency="$"))
    assert scored[accuracy.CURRENCY] is Outcome.MATCH


def test_line_item_scoring_matches_rows_by_description_and_amount() -> None:
    from apverify.domain.invoice import LineItem
    from apverify.domain.value_objects import Money
    from apverify.eval.accuracy import score_line_items

    predicted = (
        LineItem("Steel bars", 2, Money.of("100"), Money.of("200")),
        LineItem("Bonus widget", 1, Money.of("50"), Money.of("50")),  # spurious
    )
    truth = [
        {"description": "Steel Bars", "amount": "200"},  # matches (fuzzy + amount)
        {"description": "Cement bags", "amount": "300"},  # missed
    ]

    stats = score_line_items(predicted, truth)

    assert stats.matched == 1
    assert stats.spurious == 1
    assert stats.missed == 1
    assert stats.precision == pytest.approx(0.5)
    assert stats.recall == pytest.approx(0.5)
    assert stats.f1 == pytest.approx(0.5)


def test_line_item_amount_mismatch_is_not_a_match() -> None:
    from apverify.domain.invoice import LineItem
    from apverify.domain.value_objects import Money
    from apverify.eval.accuracy import score_line_items

    predicted = (LineItem("Steel bars", 2, Money.of("100"), Money.of("999")),)
    stats = score_line_items(predicted, [{"description": "Steel bars", "amount": "200"}])

    assert stats.matched == 0


def test_aggregate_computes_precision_recall_f1() -> None:
    report = aggregate(
        [
            {accuracy.TOTAL: Outcome.MATCH},
            {accuracy.TOTAL: Outcome.MISMATCH},
            {accuracy.TOTAL: Outcome.MISSED},
        ]
    )
    stat = next(s for s in report.stats if s.field == accuracy.TOTAL)

    assert stat.precision == pytest.approx(0.5)  # 1 correct of 2 predicted
    assert stat.recall == pytest.approx(1 / 3)  # 1 correct of 3 with ground truth
    assert stat.f1 == pytest.approx(0.4)
    assert report.documents == 3
