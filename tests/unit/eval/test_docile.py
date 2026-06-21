from __future__ import annotations

from apverify.domain.value_objects import Money
from apverify.eval.dataset_eval import run_dataset_eval
from apverify.eval.docile import docile_to_example

# A synthetic, DocILE-shaped KILE annotation.
_FIELDS: list[dict[str, object]] = [
    {"fieldtype": "vendor_name", "text": "Globex GmbH"},
    {"fieldtype": "date_issue", "text": "2025-06-04"},
    {"fieldtype": "currency_code_amount_due", "text": "EUR"},
    {"fieldtype": "amount_total_net", "text": "1.000,00"},
    {"fieldtype": "amount_total_tax", "text": "190,00"},
    {"fieldtype": "amount_total_gross", "text": "1.190,00"},
]
_WORDS = ["Globex", "GmbH", "1.000,00", "190,00", "1.190,00"]


def test_docile_annotation_maps_to_a_domain_invoice() -> None:
    invoice = docile_to_example(_FIELDS, _WORDS, "docile-0001").invoice

    assert invoice.vendor_name == "Globex GmbH"
    assert invoice.currency == "EUR"
    assert invoice.subtotal == Money.of("1000")
    assert invoice.tax == Money.of("190")
    assert invoice.total == Money.of("1190")


def test_fully_labelled_reconciling_invoice_auto_approves() -> None:
    report = run_dataset_eval([docile_to_example(_FIELDS, _WORDS, "docile-0001")])
    assert report.auto_approve_rate == 1.0


def test_missing_subtotal_falls_back_to_total_so_arithmetic_does_not_false_fail() -> None:
    sparse: list[dict[str, object]] = [
        {"fieldtype": "vendor_name", "text": "Initech"},
        {"fieldtype": "amount_total_gross", "text": "500,00"},
    ]
    invoice = docile_to_example(sparse, ["Initech", "500,00"], "docile-0002").invoice

    assert invoice.subtotal == invoice.total == Money.of("500")
    assert invoice.tax == Money.of(0)


def test_docile_skips_gst_specific_fields() -> None:
    invoice = docile_to_example(_FIELDS, _WORDS, "docile-0001").invoice
    assert invoice.vendor_gstin is None
    assert invoice.invoice_number == ""
