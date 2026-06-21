"""CORD-v2 real-receipt eval.

CORD (``naver-clova-ix/cord-v2``, CC-BY-4.0) is a public set of real Indonesian
receipts with human ground-truth labels. The mapping from CORD's schema to our
domain is pure and unit-tested; the dataset download is isolated in
:func:`load_cord` behind the optional ``datasets`` extra.

CORD carries no vendor tax-id / invoice number / date, so those (GST-specific)
checks skip; the eval exercises the amount cross-check and arithmetic against real
OCR tokens.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.ocr import RawText
from apverify.domain.value_objects import Money
from apverify.eval.dataset_eval import DatasetExample, parse_amount

_LEADING_INT = re.compile(r"\d+")


def cord_to_example(ground_truth: dict[str, object], label: str) -> DatasetExample:
    parse = _as_dict(ground_truth.get("gt_parse"))
    line_items = _line_items(parse.get("menu"))
    sub_total = _as_dict(parse.get("sub_total"))
    total_section = _as_dict(parse.get("total"))

    subtotal = _optional_amount(sub_total.get("subtotal_price"))
    subtotal = subtotal if subtotal is not None else _sum(item.line_total for item in line_items)
    tax = _optional_amount(sub_total.get("tax_price"))
    service = _optional_amount(sub_total.get("service_price"))
    total = _optional_amount(total_section.get("total_price"))
    total = total if total is not None else subtotal + _sum_optional(tax, service)

    invoice = Invoice(
        vendor_name="",
        invoice_number="",
        invoice_date="",
        currency="IDR",
        subtotal=subtotal,
        tax=_sum_optional(tax, service),
        total=total,
        line_items=tuple(line_items),
        # CORD's tax / service are additive components that sum into the tax bucket;
        # they occupy the breakdown slots so the cross-check tests printed values.
        tax_breakdown=TaxBreakdown(cgst=tax, sgst=service),
    )
    return DatasetExample(label=label, invoice=invoice, raw_text=_raw_text(ground_truth))


def load_cord(split: str = "test", limit: int | None = None) -> list[DatasetExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("install the dataset extra: pip install -e '.[datasets]'") from exc

    dataset = load_dataset("naver-clova-ix/cord-v2", split=split)
    examples: list[DatasetExample] = []
    for index, row in enumerate(dataset):
        if limit is not None and len(examples) >= limit:
            break
        try:
            ground_truth = json.loads(row["ground_truth"])
            examples.append(cord_to_example(ground_truth, f"cord-{split}-{index:04d}"))
        except (ValueError, KeyError, TypeError):
            continue
    return examples


def _line_items(menu: object) -> list[LineItem]:
    rows = menu if isinstance(menu, list) else [menu]
    items: list[LineItem] = []
    for row in rows:
        entry = _as_dict(row)
        price = entry.get("price")
        if not isinstance(price, str):
            continue
        try:
            line_total = parse_amount(price)
        except ValueError:
            continue
        quantity = _quantity(entry.get("cnt"))
        items.append(LineItem(str(entry.get("nm", "item")), quantity, line_total, line_total))
    return items


def _raw_text(ground_truth: dict[str, object]) -> RawText:
    words: list[str] = []
    for line in _as_list(ground_truth.get("valid_line")):
        for word in _as_list(_as_dict(line).get("words")):
            text = _as_dict(word).get("text")
            if isinstance(text, str) and text:
                words.append(text)
    if words:
        return RawText(text=" ".join(words))
    return RawText(text=" ".join(_leaves(ground_truth.get("gt_parse"))))


def _quantity(raw: object) -> int:
    match = _LEADING_INT.search(str(raw)) if raw is not None else None
    return int(match.group()) if match else 1


def _optional_amount(value: object) -> Money | None:
    if not isinstance(value, str):
        return None
    try:
        return parse_amount(value)
    except ValueError:
        return None


def _sum(amounts: Iterable[Money]) -> Money:
    total = Money.of(0)
    for amount in amounts:
        total += amount
    return total


def _sum_optional(*amounts: Money | None) -> Money:
    return _sum(amount for amount in amounts if amount is not None)


def _leaves(value: object) -> list[str]:
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in _leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _leaves(child)]
    return [str(value)] if value is not None else []


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []
