from __future__ import annotations

from collections.abc import Sequence

from apverify.application.ports import PageImage
from apverify.domain.invoice import Invoice
from apverify.domain.value_objects import Money
from apverify.eval.accuracy_eval import LabelledDocument
from apverify.eval.uncertainty_eval import collect_uncertainty_signals


def _invoice(total: str) -> Invoice:
    return Invoice(
        vendor_name="ACME Steel Pvt Ltd",
        invoice_number="INV-1",
        invoice_date="04-06-2025",
        currency="INR",
        subtotal=Money.of("100"),
        tax=Money.of("18"),
        total=Money.of(total),
    )


class _Sampler:
    def __init__(self, totals: list[str]) -> None:
        self._totals = totals

    def extract_samples(self, pages: Sequence[PageImage], samples: int) -> tuple[Invoice, ...]:
        return tuple(_invoice(total) for total in self._totals)


def _document() -> LabelledDocument:
    return LabelledDocument(
        label="doc",
        truth={"total": "118", "vendor": "ACME Steel Pvt Ltd"},
        pages=(PageImage(data=b"img"),),
    )


def test_unanimous_samples_are_maximally_consistent_and_correct() -> None:
    signals = collect_uncertainty_signals([_document()], _Sampler(["118", "118", "118", "118"]), 4)

    # every field is unanimous (consistency 1.0); the total's consensus matches truth
    assert all(score == 1.0 for score, _ in signals.self_consistency)
    assert (1.0, True) in signals.self_consistency


def test_split_samples_lower_consistency_and_flag_a_wrong_consensus() -> None:
    # 3 of 5 say the wrong total (999); consensus is wrong, agreement is 0.6
    signals = collect_uncertainty_signals(
        [_document()], _Sampler(["999", "999", "999", "118", "118"]), 5
    )

    consensus_total = next(s for s in signals.self_consistency if s[0] == 0.6)
    assert consensus_total == (0.6, False)


def test_entropy_signal_is_collected_alongside_consistency() -> None:
    signals = collect_uncertainty_signals([_document()], _Sampler(["118", "118"]), 2)
    assert len(signals.semantic_entropy) == len(signals.self_consistency)
    assert all(0.0 < conf <= 1.0 for conf, _ in signals.semantic_entropy)
