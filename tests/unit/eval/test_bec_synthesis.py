from __future__ import annotations

from apverify.eval.bec_synthesis import SCENARIOS, build_bec_cases
from apverify.eval.synthetic import generate_dataset


def test_each_base_invoice_yields_one_case_per_scenario() -> None:
    cases = build_bec_cases(generate_dataset(count=3))
    scenarios = {case.scenario for case in cases}
    assert scenarios == set(SCENARIOS)
    assert all(case.master for case in cases)  # a master is always supplied


def test_known_clean_uses_a_known_bank_account() -> None:
    case = next(
        c for c in build_bec_cases(generate_dataset(count=1)) if c.scenario == "known_clean"
    )
    known = {acct for vendor in case.master for acct in vendor.bank_accounts}
    assert case.invoice.bank_account in known


def test_impersonation_name_differs_from_every_known_vendor() -> None:
    case = next(
        c for c in build_bec_cases(generate_dataset(count=1)) if c.scenario == "impersonation"
    )
    assert all(case.invoice.vendor_name != vendor.name for vendor in case.master)
