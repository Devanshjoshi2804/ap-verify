from __future__ import annotations

import pytest

from apverify.domain.critique import ApprovalDecision
from apverify.domain.value_objects import Money
from apverify.eval.cord import cord_to_example
from apverify.eval.dataset_eval import parse_amount, run_dataset_eval


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("50.000", "50000.00"),  # Indonesian thousands
        ("1.000.000", "1000000.00"),
        ("10,000", "10000.00"),  # comma thousands
        ("10.50", "10.50"),  # dot decimal
        ("10,50", "10.50"),  # comma decimal
        ("10.000,75", "10000.75"),  # both: comma decimal
        ("1,234.56", "1234.56"),  # both: dot decimal
        ("Rp 25.000", "25000.00"),  # currency word stripped
    ],
)
def test_parse_amount_handles_mixed_conventions(raw: str, expected: str) -> None:
    assert parse_amount(raw) == Money.of(expected)


def test_parse_amount_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="no digits"):
        parse_amount("free")


_GROUND_TRUTH: dict[str, object] = {
    "gt_parse": {
        "menu": [
            {"nm": "Nasi Goreng", "cnt": "2 x", "price": "50.000"},
            {"nm": "Es Teh", "cnt": "2", "price": "10.000"},
        ],
        "sub_total": {"subtotal_price": "60.000", "tax_price": "6.000"},
        "total": {"total_price": "66.000"},
    },
    "valid_line": [
        {"words": [{"text": "Nasi"}, {"text": "Goreng"}, {"text": "50.000"}]},
        {"words": [{"text": "Es"}, {"text": "Teh"}, {"text": "10.000"}]},
        {"words": [{"text": "Subtotal"}, {"text": "60.000"}]},
        {"words": [{"text": "Tax"}, {"text": "6.000"}]},
        {"words": [{"text": "Total"}, {"text": "66.000"}]},
    ],
}


def test_cord_ground_truth_maps_to_a_domain_invoice() -> None:
    example = cord_to_example(_GROUND_TRUTH, "cord-test-0001")
    invoice = example.invoice

    assert invoice.subtotal == Money.of("60000")
    assert invoice.tax == Money.of("6000")
    assert invoice.total == Money.of("66000")
    assert len(invoice.line_items) == 2
    assert invoice.line_items[0].quantity == 2
    assert example.raw_text.contains("66000")


def test_reconciling_receipt_auto_approves() -> None:
    example = cord_to_example(_GROUND_TRUTH, "cord-test-0001")
    report = run_dataset_eval([example])

    assert report.auto_approve_rate == 1.0
    assert report.failed_checks == {}


def test_non_reconciling_receipt_is_held_and_reported() -> None:
    broken: dict[str, object] = {
        "gt_parse": {
            "menu": [{"nm": "Item", "cnt": "1", "price": "60.000"}],
            "sub_total": {"subtotal_price": "60.000", "tax_price": "6.000"},
            "total": {"total_price": "50.000"},  # does not reconcile (e.g. unmodelled discount)
        },
        "valid_line": [{"words": [{"text": "Total"}, {"text": "50.000"}]}],
    }
    report = run_dataset_eval([cord_to_example(broken, "cord-broken")])

    assert report.decisions.get(ApprovalDecision.AUTO_APPROVE.value, 0) == 0
    assert any("arithmetic" in label for label in report.failed_checks)
