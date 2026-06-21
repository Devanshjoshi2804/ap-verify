from __future__ import annotations

from apverify.eval.fraud_suite_synthesis import (
    CLEAN,
    FRAUD_LABELS,
    LABELS,
    build_fraud_suite,
)
from apverify.eval.synthetic import generate_dataset


def test_every_label_appears_with_full_context() -> None:
    cases = build_fraud_suite(generate_dataset(count=3))
    assert {case.label for case in cases} == set(LABELS)
    for case in cases:
        assert case.priors  # ledger present
        assert case.master  # vendor master present
        assert len(case.history) >= 3  # vendor history present
        assert case.is_fraud == (case.label in FRAUD_LABELS)


def test_clean_case_is_not_a_duplicate_of_the_ledger() -> None:
    # The clean candidate must carry a new invoice number so it is not a resend.
    case = next(c for c in build_fraud_suite(generate_dataset(count=1)) if c.label == CLEAN)
    assert all(case.invoice.invoice_number != prior.invoice.invoice_number for prior in case.priors)
