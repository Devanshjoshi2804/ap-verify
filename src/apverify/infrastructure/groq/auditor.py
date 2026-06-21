"""Groq LLM-as-auditor implementing the ``SemanticAuditor`` port.

A fast, cheap text model gives a second opinion on the fields the deterministic
critic was unsure about. It sees the OCR text and the extracted values and judges,
per field, whether the value is genuinely supported by the document — catching
plausible, on-page, arithmetically-consistent errors the cheap checks cannot.

Text-only by design: the auditor reasons over the OCR text, not the image, so it
is independent of the vision extractor it is checking.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from groq import APIError, Groq

from apverify.domain.audit import AuditVerdict
from apverify.domain.critique import InvoiceField
from apverify.domain.invoice import Invoice
from apverify.domain.ocr import RawText
from apverify.infrastructure.errors import AuditError

_MAX_TEXT_CHARS = 6000
_SYSTEM_PROMPT = (
    "You audit invoice field extractions for an accounts-payable system. "
    "Given the raw OCR text of an invoice and values an extractor produced for "
    "specific fields, decide for each field whether the value is genuinely "
    "supported by the document. Reply with JSON only: "
    '{"verdicts": [{"field": <name>, "trustworthy": <bool>, '
    '"confidence": <0..1>, "reason": <short string>}]}. '
    "Judge only the fields asked about; do not invent fields."
)


class GroqSemanticAuditor:
    def __init__(self, client: Groq, model: str) -> None:
        self._client = client
        self._model = model

    def audit(
        self, invoice: Invoice, raw_text: RawText, fields: Sequence[InvoiceField]
    ) -> list[AuditVerdict]:
        if not fields:
            return []

        prompt = self._user_prompt(invoice, raw_text, fields)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            payload = response.choices[0].message.content or ""
        except APIError as exc:
            raise AuditError(f"Groq audit request failed: {exc}") from exc

        return self._parse(payload, set(fields))

    def _user_prompt(
        self, invoice: Invoice, raw_text: RawText, fields: Sequence[InvoiceField]
    ) -> str:
        values = _field_values(invoice)
        asked = "\n".join(f"- {field.value}: {values.get(field, '')!r}" for field in fields)
        document = raw_text.text[:_MAX_TEXT_CHARS]
        return f"OCR TEXT:\n{document}\n\nFIELDS TO AUDIT:\n{asked}"

    def _parse(self, payload: str, allowed: set[InvoiceField]) -> list[AuditVerdict]:
        try:
            raw_verdicts = json.loads(payload).get("verdicts", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            raise AuditError(f"auditor returned malformed JSON: {exc}") from exc

        by_value = {field.value: field for field in allowed}
        verdicts: list[AuditVerdict] = []
        for entry in raw_verdicts:
            field = by_value.get(str(entry.get("field")))
            if field is None:
                continue
            verdicts.append(
                AuditVerdict(
                    field=field,
                    trustworthy=bool(entry.get("trustworthy", True)),
                    confidence=_clamp(entry.get("confidence", 0.0)),
                    reason=str(entry.get("reason", "")),
                )
            )
        return verdicts


def _field_values(invoice: Invoice) -> dict[InvoiceField, str]:
    return {
        InvoiceField.VENDOR: invoice.vendor_name,
        InvoiceField.GSTIN: invoice.vendor_gstin or "",
        InvoiceField.INVOICE_NUMBER: invoice.invoice_number,
        InvoiceField.INVOICE_DATE: invoice.invoice_date,
        InvoiceField.CURRENCY: invoice.currency,
        InvoiceField.SUBTOTAL: str(invoice.subtotal),
        InvoiceField.TAX: str(invoice.tax),
        InvoiceField.TOTAL: str(invoice.total),
        InvoiceField.LINE_ITEMS: f"{len(invoice.line_items)} line(s)",
    }


def _clamp(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
