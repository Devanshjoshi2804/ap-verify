"""Collect sampling-based uncertainty signals from real documents.

For each labelled document we resample the extractor a few times, then per field
measure self-consistency and semantic entropy across the samples and pair each with
whether the *consensus* (modal) value is actually correct. The result feeds the same
signal comparison (AUROC / ECE) as the other uncertainty signals, answering the
roadmap question directly: does agreeing-with-itself predict being right?
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from apverify.application.errors import PortError
from apverify.application.ports import SamplingExtractor
from apverify.domain.critique import InvoiceField
from apverify.domain.invoice import Invoice
from apverify.eval import accuracy
from apverify.eval.accuracy_eval import LabelledDocument
from apverify.eval.calibration import Sample
from apverify.eval.calibration_eval import _FIELD_MAP
from apverify.eval.uncertainty import (
    Equivalent,
    cluster_values,
    confidence_from_entropy,
    self_consistency,
    semantic_entropy,
)


@dataclass(frozen=True, slots=True)
class UncertaintySignals:
    self_consistency: list[Sample]
    semantic_entropy: list[Sample]


def collect_uncertainty_signals(
    documents: Iterable[LabelledDocument],
    extractor: SamplingExtractor,
    samples: int = 5,
) -> UncertaintySignals:
    consistency_samples: list[Sample] = []
    entropy_samples: list[Sample] = []
    for document in documents:
        if not document.truth or not document.pages:
            continue
        try:
            extractions = extractor.extract_samples(document.pages, samples)
        except PortError:
            continue

        for invoice_field, canonical in _FIELD_MAP.items():
            truth_value = document.truth.get(canonical)
            if truth_value is None:
                continue
            values = [_field_value(extraction, invoice_field) for extraction in extractions]
            values = [value for value in values if value]
            if not values:
                continue

            equivalent = _equivalence(canonical)
            modal = max(cluster_values(values, equivalent), key=len)[0]
            correct = accuracy.value_matches(canonical, modal, truth_value)

            consistency_samples.append((self_consistency(values, equivalent), correct))
            entropy_samples.append(
                (confidence_from_entropy(semantic_entropy(values, equivalent)), correct)
            )
    return UncertaintySignals(
        self_consistency=consistency_samples, semantic_entropy=entropy_samples
    )


def _equivalence(canonical: str) -> Equivalent:
    return lambda left, right: accuracy.value_matches(canonical, left, right)


def _field_value(invoice: Invoice, field: InvoiceField) -> str:
    return {
        InvoiceField.VENDOR: invoice.vendor_name,
        InvoiceField.INVOICE_DATE: invoice.invoice_date,
        InvoiceField.CURRENCY: invoice.currency,
        InvoiceField.SUBTOTAL: str(invoice.subtotal.amount),
        InvoiceField.TAX: str(invoice.tax.amount),
        InvoiceField.TOTAL: str(invoice.total.amount),
    }[field]
