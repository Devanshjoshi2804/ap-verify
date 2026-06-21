from __future__ import annotations

from collections.abc import Sequence

from apverify.application.ports import (
    AnomalyDetector,
    InvoiceLedger,
    VendorHistoryRepository,
    VendorMasterRepository,
)
from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.fraud import IdentifiedInvoice
from apverify.domain.invoice import Invoice
from apverify.domain.vendor_master import KnownVendor


def test_a_simple_object_satisfies_the_invoice_ledger_port() -> None:
    class _Ledger:
        def known_invoices(self) -> tuple[IdentifiedInvoice, ...]:
            return ()

    assert isinstance(_Ledger(), InvoiceLedger)


def test_a_simple_object_satisfies_the_vendor_master_port() -> None:
    class _Master:
        def known_vendors(self) -> tuple[KnownVendor, ...]:
            return ()

    assert isinstance(_Master(), VendorMasterRepository)


def test_objects_satisfy_the_anomaly_ports() -> None:
    class _Detector:
        def score(self, invoice: Invoice, history: Sequence[Invoice]) -> AnomalyAssessment:
            return AnomalyAssessment(0.0, AnomalySeverity.NONE, "history", "n/a")

    class _History:
        def history_for(self, invoice: Invoice) -> tuple[Invoice, ...]:
            return ()

    assert isinstance(_Detector(), AnomalyDetector)
    assert isinstance(_History(), VendorHistoryRepository)
