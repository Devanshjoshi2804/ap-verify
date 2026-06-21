"""Collect calibration samples: pair the critic's per-field confidence with
whether that field was actually extracted correctly.

For each labelled document we run the real extractor, run the critic over the
extraction and the document's OCR text, and score the extraction against ground
truth. Each field where the critic reported a confidence becomes a
``(confidence, correct)`` sample for :mod:`apverify.eval.calibration`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from apverify.application.errors import PortError
from apverify.application.ports import InvoiceExtractor, SelfReportingExtractor
from apverify.domain.checks import review
from apverify.domain.critique import DEFAULT_POLICY, InvoiceField, Policy
from apverify.domain.invoice import Invoice
from apverify.eval import accuracy
from apverify.eval.accuracy import Outcome, score_document
from apverify.eval.accuracy_eval import LabelledDocument
from apverify.eval.calibration import Sample

_FIELD_MAP = {
    InvoiceField.VENDOR: accuracy.VENDOR,
    InvoiceField.INVOICE_DATE: accuracy.DATE,
    InvoiceField.CURRENCY: accuracy.CURRENCY,
    InvoiceField.SUBTOTAL: accuracy.SUBTOTAL,
    InvoiceField.TAX: accuracy.TAX,
    InvoiceField.TOTAL: accuracy.TOTAL,
}


@dataclass(frozen=True, slots=True)
class UncertaintySamples:
    """Two signals over the same extractions, scored against the same truth.

    ``critic`` is the structural confidence the critic derives from its checks;
    ``verbalized`` is what the model says about itself. Comparing their calibration
    answers a concrete question: which signal should gate auto-approval?
    """

    critic: list[Sample]
    verbalized: list[Sample]


def collect_calibration_samples(
    documents: Iterable[LabelledDocument],
    extractor: InvoiceExtractor,
    policy: Policy = DEFAULT_POLICY,
) -> list[Sample]:
    samples: list[Sample] = []
    for document in documents:
        if not document.truth or not document.pages:
            continue
        try:
            predicted = extractor.extract(document.pages)
        except PortError:
            continue
        samples.extend(_critic_samples(predicted, document, policy))
    return samples


def collect_uncertainty_samples(
    documents: Iterable[LabelledDocument],
    extractor: SelfReportingExtractor,
    policy: Policy = DEFAULT_POLICY,
) -> UncertaintySamples:
    """Critic vs. verbalized confidence from one extraction pass per document."""
    critic: list[Sample] = []
    verbalized: list[Sample] = []
    for document in documents:
        if not document.truth or not document.pages:
            continue
        try:
            extraction = extractor.extract_with_confidence(document.pages)
        except PortError:
            continue
        predicted = extraction.invoice
        outcomes = score_document(predicted, document.truth)
        critic.extend(_critic_samples(predicted, document, policy, outcomes))
        for invoice_field, canonical in _FIELD_MAP.items():
            if invoice_field in extraction.confidences and canonical in outcomes:
                correct = outcomes[canonical] is Outcome.MATCH
                verbalized.append((extraction.confidences[invoice_field], correct))
    return UncertaintySamples(critic=critic, verbalized=verbalized)


def _critic_samples(
    predicted: Invoice,
    document: LabelledDocument,
    policy: Policy,
    outcomes: dict[str, Outcome] | None = None,
) -> list[Sample]:
    if outcomes is None:
        outcomes = score_document(predicted, document.truth)
    confidences = {
        fc.field: fc.confidence
        for fc in review(predicted, document.raw_text, policy).field_confidences
    }
    samples: list[Sample] = []
    for invoice_field, canonical in _FIELD_MAP.items():
        if invoice_field in confidences and canonical in outcomes:
            samples.append((confidences[invoice_field], outcomes[canonical] is Outcome.MATCH))
    return samples
