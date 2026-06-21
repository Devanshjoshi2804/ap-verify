"""Collect multi-signal feature rows from real documents.

For each labelled document this runs the primary extractor (with verbalized
confidence), the critic, and a *second* extractor, then records — per field — every
trust signal alongside whether the field was actually correct. One pass yields the
fusion training data, the calibration samples, and the cross-model diagnostic.

A document is only usable when the second model also returns an extraction: the
cross-model signal is the whole point, so a row without it is dropped rather than
silently defaulted to "agrees".
"""

from __future__ import annotations

from collections.abc import Iterable

from apverify.application.errors import PortError
from apverify.application.ports import InvoiceExtractor, SelfReportingExtractor
from apverify.domain.checks import review
from apverify.domain.consistency import Agreement, compare_extractions
from apverify.domain.critique import (
    DEFAULT_POLICY,
    CheckCategory,
    FieldConfidence,
    Policy,
)
from apverify.eval.accuracy import Outcome, score_document
from apverify.eval.accuracy_eval import LabelledDocument
from apverify.eval.calibration_eval import _FIELD_MAP
from apverify.eval.fusion import FeatureRow


def collect_feature_rows(
    documents: Iterable[LabelledDocument],
    primary: SelfReportingExtractor,
    secondary: InvoiceExtractor,
    policy: Policy = DEFAULT_POLICY,
    primary_name: str = "",
    secondary_name: str = "",
) -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    for document in documents:
        if not document.truth or not document.pages:
            continue
        try:
            extraction = primary.extract_with_confidence(document.pages)
            secondary_invoice = secondary.extract(document.pages)
        except PortError:
            continue

        predicted = extraction.invoice
        outcomes = score_document(predicted, document.truth)
        critic = {
            fc.field: fc for fc in review(predicted, document.raw_text, policy).field_confidences
        }
        agreement = {
            comparison.field: comparison.agreement
            for comparison in compare_extractions(predicted, secondary_invoice).comparisons
        }

        for invoice_field, canonical in _FIELD_MAP.items():
            field_critic = critic.get(invoice_field)
            if field_critic is None or canonical not in outcomes:
                continue
            rows.append(
                FeatureRow(
                    label=document.label,
                    field=str(invoice_field),
                    critic_confidence=field_critic.confidence,
                    verbalized_confidence=extraction.confidences.get(invoice_field, 0.0),
                    cross_check_passed=_passed(field_critic, CheckCategory.CROSS_CHECK),
                    arithmetic_passed=_passed(field_critic, CheckCategory.ARITHMETIC),
                    format_passed=_passed(field_critic, CheckCategory.FORMAT),
                    cross_model_agrees=agreement.get(invoice_field) is Agreement.AGREES,
                    correct=outcomes[canonical] is Outcome.MATCH,
                    primary=primary_name,
                    secondary=secondary_name,
                )
            )
    return rows


def _passed(field_confidence: FieldConfidence, category: CheckCategory) -> bool:
    """True unless a check of this category actually failed (absent or skipped is
    treated as passed — the critic never raised a flag)."""
    return not any(check.category is category and check.failed for check in field_confidence.checks)
