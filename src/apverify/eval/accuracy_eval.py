"""Run the live extractor over labelled dataset images and score accuracy.

A ``LabelledDocument`` pairs ground-truth field values (and line items) with the
page image(s), so the real extractor can be run and its output scored against the
labels. The scoring is pure (:mod:`apverify.eval.accuracy`); this module is the
impure glue that reads the datasets and calls the model.
"""

from __future__ import annotations

import io
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from apverify.application.errors import PortError
from apverify.application.ports import InvoiceExtractor, PageImage
from apverify.domain.ocr import RawText
from apverify.eval import accuracy
from apverify.eval.accuracy import (
    AccuracyReport,
    LineItemStats,
    aggregate,
    score_document,
    score_line_items,
)

_DOCILE_TRUTH = {
    "vendor_name": accuracy.VENDOR,
    "date_issue": accuracy.DATE,
    "currency_code_amount_due": accuracy.CURRENCY,
    "amount_total_net": accuracy.SUBTOTAL,
    "amount_total_tax": accuracy.TAX,
    "amount_total_gross": accuracy.TOTAL,
}


@dataclass(frozen=True, slots=True)
class LabelledDocument:
    label: str
    truth: dict[str, str]
    pages: tuple[PageImage, ...]
    truth_lines: tuple[dict[str, str], ...] = field(default_factory=tuple)
    raw_text: RawText = field(default_factory=lambda: RawText(""))


def run_field_accuracy(
    documents: Iterable[LabelledDocument], extractor: InvoiceExtractor
) -> AccuracyReport:
    scored: list[dict[str, accuracy.Outcome]] = []
    matched = spurious = missed = 0
    has_lines = False
    for document in documents:
        if not document.truth or not document.pages:
            continue
        try:
            predicted = extractor.extract(document.pages)
        except PortError:
            continue
        scored.append(score_document(predicted, document.truth))
        if document.truth_lines:
            has_lines = True
            line_stats = score_line_items(predicted.line_items, list(document.truth_lines))
            matched += line_stats.matched
            spurious += line_stats.spurious
            missed += line_stats.missed

    report = aggregate(scored)
    line_items = LineItemStats(matched, spurious, missed) if has_lines else None
    return AccuracyReport(report.documents, report.stats, line_items)


def load_docile_labelled(
    dataset_path: str, split: str = "val", limit: int | None = None, max_pages: int = 2
) -> list[LabelledDocument]:
    try:
        from docile.dataset import Dataset
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("install the dataset extra: pip install -e '.[docile]'") from exc

    dataset = Dataset(split, dataset_path)
    documents: list[LabelledDocument] = []
    for index, document in enumerate(dataset):
        if limit is not None and len(documents) >= limit:
            break
        try:
            with document:
                truth = _docile_truth(document.annotation.fields)
                lines = _docile_lines(document.annotation.li_fields)
                pages = _docile_pages(document, max_pages) if truth else ()
                text = _docile_text(document) if truth else RawText("")
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        if truth and pages:
            documents.append(
                LabelledDocument(f"docile-{split}-{index:04d}", truth, pages, lines, text)
            )
    return documents


def load_cord_labelled(split: str = "test", limit: int | None = None) -> list[LabelledDocument]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("install the dataset extra: pip install -e '.[datasets]'") from exc

    dataset = load_dataset("naver-clova-ix/cord-v2", split=split)
    documents: list[LabelledDocument] = []
    for index, row in enumerate(dataset):
        if limit is not None and len(documents) >= limit:
            break
        try:
            ground_truth = json.loads(row["ground_truth"])
            parse = ground_truth.get("gt_parse", {})
            truth = _cord_truth(parse)
            lines = _cord_lines(parse.get("menu"))
            pages = (_encode(row["image"]),)
            text = _cord_text(ground_truth)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        if truth:
            documents.append(
                LabelledDocument(f"cord-{split}-{index:04d}", truth, pages, lines, text)
            )
    return documents


def _docile_truth(fields: Any) -> dict[str, str]:
    truth: dict[str, str] = {}
    for item in fields:
        key = _DOCILE_TRUTH.get(item.fieldtype)
        if key and key not in truth and isinstance(item.text, str) and item.text.strip():
            truth[key] = item.text
    return truth


def _docile_lines(li_fields: Any) -> tuple[dict[str, str], ...]:
    grouped: dict[Any, dict[str, str]] = defaultdict(dict)
    for item in li_fields:
        if isinstance(item.text, str) and item.text.strip():
            grouped[item.line_item_id].setdefault(item.fieldtype, item.text)
    lines: list[dict[str, str]] = []
    for _line_id, row in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        amount = row.get("line_item_amount_gross")
        if amount is None:
            continue
        lines.append({"description": row.get("line_item_description", ""), "amount": amount})
    return tuple(lines)


def _docile_text(document: Any) -> RawText:
    words = [
        word.text
        for page in range(document.page_count)
        for word in document.ocr.get_all_words(page)
        if getattr(word, "text", None)
    ]
    return RawText(" ".join(words))


def _cord_text(ground_truth: dict[str, Any]) -> RawText:
    words: list[str] = []
    for line in ground_truth.get("valid_line", []):
        for word in line.get("words", []):
            text = word.get("text")
            if isinstance(text, str) and text:
                words.append(text)
    return RawText(" ".join(words))


def _docile_pages(document: Any, max_pages: int) -> tuple[PageImage, ...]:
    pages: list[PageImage] = []
    for page in range(min(document.page_count, max_pages)):
        pages.append(_encode(document.page_image(page)))
    return tuple(pages)


def _cord_truth(parse: dict[str, Any]) -> dict[str, str]:
    sub_total = parse.get("sub_total") or {}
    total = parse.get("total") or {}
    truth: dict[str, str] = {}
    for key, value in (
        (accuracy.SUBTOTAL, sub_total.get("subtotal_price")),
        (accuracy.TAX, sub_total.get("tax_price")),
        (accuracy.TOTAL, total.get("total_price")),
    ):
        if isinstance(value, str) and value.strip():
            truth[key] = value
    return truth


def _cord_lines(menu: Any) -> tuple[dict[str, str], ...]:
    rows = menu if isinstance(menu, list) else [menu]
    lines: list[dict[str, str]] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("price"), str):
            lines.append({"description": str(row.get("nm", "")), "amount": row["price"]})
    return tuple(lines)


def _encode(image: Any) -> PageImage:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return PageImage(data=buffer.getvalue())
