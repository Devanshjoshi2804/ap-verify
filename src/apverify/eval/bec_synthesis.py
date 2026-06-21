"""Synthesise a labelled BEC benchmark over base invoices.

We build a vendor master (each base vendor with one known bank account), then for each
base invoice emit the attack variants — a changed bank account, a typo-squatted vendor
name, a brand-new payee — and the legitimate look-alikes that must not raise a HIGH
flag: the same vendor paid to its known account, and a genuinely unrelated new vendor.

Deterministic: fixed transforms of the base, no randomness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from apverify.domain.invoice import Invoice
from apverify.domain.vendor_master import KnownVendor
from apverify.eval.synthetic import GroundTruth

BANK_CHANGE = "bank_change"
IMPERSONATION = "impersonation"
NEW_PAYEE = "new_payee"
KNOWN_CLEAN = "known_clean"
LEGIT_NEW = "legit_new"

SCENARIOS = (BANK_CHANGE, IMPERSONATION, NEW_PAYEE, KNOWN_CLEAN, LEGIT_NEW)
HIGH_SCENARIOS = (BANK_CHANGE, IMPERSONATION)

_ATTACKER_BANK = "ACCT-9999"
_NEW_BANK = "ACCT-7777"
# Letter -> confusable digit for typo-squatting a known name.
_LOOKALIKE = (("e", "3"), ("o", "0"), ("a", "4"), ("i", "1"))


@dataclass(frozen=True, slots=True)
class BecCase:
    invoice: Invoice
    master: tuple[KnownVendor, ...]
    scenario: str


def build_bec_cases(base: Sequence[GroundTruth]) -> list[BecCase]:
    names = sorted({truth.invoice.vendor_name for truth in base})
    account_of = {name: f"ACCT-{index:04d}" for index, name in enumerate(names)}
    master = tuple(KnownVendor(name, frozenset({account_of[name]})) for name in names)

    cases: list[BecCase] = []
    for index, truth in enumerate(base):
        invoice = truth.invoice
        known_account = account_of[invoice.vendor_name]
        cases.extend(
            [
                BecCase(replace(invoice, bank_account=_ATTACKER_BANK), master, BANK_CHANGE),
                BecCase(
                    replace(
                        invoice,
                        vendor_name=_typosquat(invoice.vendor_name),
                        bank_account=_ATTACKER_BANK,
                    ),
                    master,
                    IMPERSONATION,
                ),
                BecCase(
                    replace(invoice, vendor_name=f"New Supplier {index}", bank_account=_NEW_BANK),
                    master,
                    NEW_PAYEE,
                ),
                BecCase(replace(invoice, bank_account=known_account), master, KNOWN_CLEAN),
                BecCase(
                    replace(
                        invoice, vendor_name=f"Unrelated Trader {index}", bank_account=_NEW_BANK
                    ),
                    master,
                    LEGIT_NEW,
                ),
            ]
        )
    return cases


def _typosquat(name: str) -> str:
    """One confusable substitution — looks like the vendor, is not the vendor."""
    lowered = name.lower()
    for original, lookalike in _LOOKALIKE:
        index = lowered.find(original)
        if index != -1:
            return name[:index] + lookalike + name[index + 1 :]
    return name + "X"
