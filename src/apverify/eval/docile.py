"""DocILE real-invoice eval (``docile.rossum.ai``, ``docile-benchmark``).

DocILE is the largest public business-document benchmark: real invoices with KILE
key-field and LIR line-item annotations. The mapping below is verified against the
real dataset.

A deliberate modelling choice: DocILE annotates only the *subset* of fields present
and recognised on each document — ``amount_total_net`` / ``amount_total_tax`` and
individual line items are frequently unlabelled even when printed. Reconciling
arithmetic against partial ground truth would manufacture failures that reflect
annotation gaps, not extraction quality, so this adapter evaluates what DocILE
faithfully supports: the **key-amount OCR cross-check** (do the extracted vendor,
subtotal, tax and total actually appear in the real OCR text) plus format checks.
The subtotal falls back to the total when unlabelled, making the arithmetic check a
no-op except where the net/tax are genuinely annotated. GST-specific fields
(tax id, GST invoice-number rules) are left empty so those India-specific checks
skip on non-Indian invoices.
"""

from __future__ import annotations

from collections.abc import Iterable

from apverify.domain.invoice import Invoice, TaxBreakdown
from apverify.domain.ocr import RawText
from apverify.domain.value_objects import Money
from apverify.eval.dataset_eval import DatasetExample, parse_amount

# DocILE KILE field types we consume (a subset of its ~55).
_TOTAL = "amount_total_gross"
_AMOUNT_DUE = "amount_due"
_SUBTOTAL = "amount_total_net"
_TAX = "amount_total_tax"
_VENDOR = "vendor_name"
_DATE = "date_issue"
_CURRENCY = "currency_code_amount_due"


def docile_to_example(
    fields: Iterable[dict[str, object]], words: Iterable[str], label: str
) -> DatasetExample:
    by_type = _first_by_type(fields)

    tax = _optional_amount(by_type.get(_TAX))
    total = _optional_amount(by_type.get(_TOTAL)) or _optional_amount(by_type.get(_AMOUNT_DUE))
    net = _optional_amount(by_type.get(_SUBTOTAL))
    subtotal = net if net is not None else (total if total is not None else Money.of(0))
    if total is None:
        total = subtotal + (tax or Money.of(0))

    invoice = Invoice(
        vendor_name=_text(by_type.get(_VENDOR)),
        invoice_number="",
        invoice_date=_text(by_type.get(_DATE)),
        currency=_currency(by_type.get(_CURRENCY)),
        subtotal=subtotal,
        tax=tax or Money.of(0),
        total=total,
        line_items=(),  # DocILE line items are partially annotated; not reliable to reconcile
        tax_breakdown=TaxBreakdown(cgst=tax),
    )
    return DatasetExample(label=label, invoice=invoice, raw_text=RawText(text=" ".join(words)))


def load_docile(
    dataset_path: str, split: str = "val", limit: int | None = None
) -> list[DatasetExample]:
    try:
        from docile.dataset import Dataset
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("install the dataset extra: pip install -e '.[docile]'") from exc

    dataset = Dataset(split, dataset_path)
    examples: list[DatasetExample] = []
    for index, document in enumerate(dataset):
        if limit is not None and len(examples) >= limit:
            break
        try:
            with document:  # loads annotation + OCR from disk for the duration
                examples.append(_document_to_example(document, f"docile-{split}-{index:04d}"))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return examples


def _document_to_example(document: object, label: str) -> DatasetExample:
    annotation = document.annotation  # type: ignore[attr-defined]
    fields = [{"fieldtype": f.fieldtype, "text": f.text} for f in annotation.fields]
    words = [
        word.text
        for page in range(document.page_count)  # type: ignore[attr-defined]
        for word in document.ocr.get_all_words(page)  # type: ignore[attr-defined]
        if getattr(word, "text", None)
    ]
    return docile_to_example(fields, words, label)


def _first_by_type(fields: Iterable[dict[str, object]]) -> dict[str, str]:
    by_type: dict[str, str] = {}
    for field in fields:
        fieldtype, text = field.get("fieldtype"), field.get("text")
        if isinstance(fieldtype, str) and isinstance(text, str):
            by_type.setdefault(fieldtype, text)
    return by_type


def _optional_amount(value: object) -> Money | None:
    if not isinstance(value, str):
        return None
    try:
        return parse_amount(value)
    except ValueError:
        return None


def _currency(value: object) -> str:
    code = value.strip().upper() if isinstance(value, str) else ""
    return code if code.isalpha() and len(code) == 3 else ""


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""
