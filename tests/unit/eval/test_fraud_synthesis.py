from __future__ import annotations

from apverify.eval.fraud_synthesis import (
    FRAUD_KINDS,
    LEGIT_KINDS,
    build_fraud_cases,
)
from apverify.eval.synthetic import generate_dataset


def test_each_base_invoice_yields_one_case_per_kind() -> None:
    base = generate_dataset(count=4)
    cases = build_fraud_cases(base)
    kinds = {case.kind for case in cases}
    assert kinds == set(FRAUD_KINDS) | set(LEGIT_KINDS)
    assert all(case.priors for case in cases)  # every case has a ledger to check against


def test_fraud_kinds_are_labelled_fraud_and_legit_kinds_are_not() -> None:
    cases = build_fraud_cases(generate_dataset(count=4))
    for case in cases:
        assert case.is_fraud == (case.kind in FRAUD_KINDS)


def test_legit_recurring_keeps_vendor_and_amount_but_changes_date_and_number() -> None:
    base = generate_dataset(count=1)
    recurring = next(c for c in build_fraud_cases(base) if c.kind == "legit_recurring")
    original = base[0].invoice
    assert recurring.candidate.vendor_name == original.vendor_name
    assert recurring.candidate.total == original.total
    assert recurring.candidate.invoice_date != original.invoice_date
    assert recurring.candidate.invoice_number != original.invoice_number
