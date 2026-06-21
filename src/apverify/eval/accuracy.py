"""Field-level extraction accuracy — the headline metric.

The decision evals (synthetic, CORD, DocILE) score whether the *critic* makes the
right call. This scores something different and more fundamental: how often the
*extractor* reads each field correctly, measured against ground-truth labels.

Per field we count matches, mismatches and misses across a set of documents and
report precision / recall / F1 — the standard key-information-extraction metric and
the baseline every accuracy improvement is measured against.

The comparison is pure and unit-tested; running the live extractor over real
dataset images lives in the accuracy runner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from difflib import SequenceMatcher
from enum import StrEnum

from apverify.domain.invoice import Invoice, LineItem
from apverify.eval.dataset_eval import parse_amount

_LINE_DESCRIPTION_THRESHOLD = 0.60

VENDOR = "vendor"
DATE = "date"
CURRENCY = "currency"
SUBTOTAL = "subtotal"
TAX = "tax"
TOTAL = "total"

_AMOUNT_TOLERANCE = Decimal("0.01")
_VENDOR_THRESHOLD = 0.80
# Legal/entity and broadcast-station suffixes carry no identifying signal: the model
# reading "PHILIP MORRIS" against a ground truth of "PHILIP MORRIS INCORPORATED", or
# "KGMB" against "KGMB TV", is correct. Stripped before the containment check so a
# suffix-only difference is not scored as a miss.
_VENDOR_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "llc",
        "llp",
        "lp",
        "ltd",
        "limited",
        "co",
        "corp",
        "corporation",
        "company",
        "plc",
        "gmbh",
        "pvt",
        "tv",
        "fm",
        "am",
    }
)
# A containing fragment must carry at least one token this long, so a stray initialism
# can't be read as "contained in" a real multi-word vendor name.
_VENDOR_MIN_TOKEN = 3
# Ground truth records currencies inconsistently (a "$" symbol vs the "USD" code);
# both are correct extractions, so compare on a normalised code.
_CURRENCY_CODES = {
    "$": "USD",
    "us$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "¥": "JPY",
    "jpy": "JPY",
}
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m/%d/%y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)


class Outcome(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    MISSED = "missed"  # ground truth present, extractor produced nothing


def score_document(predicted: Invoice, truth: dict[str, str]) -> dict[str, Outcome]:
    """Score one extraction against its ground-truth field values.

    ``truth`` carries only the fields that were actually annotated, so fields the
    dataset never labelled are not held against the extractor.
    """
    return {field: _score_field(field, value, predicted) for field, value in truth.items()}


def value_matches(field: str, predicted: str, truth: str) -> bool:
    """Value-to-value field equivalence using the same rules as scoring.

    Lets other modules cluster resampled answers and check a consensus value against
    truth without re-implementing the amount-tolerance / fuzzy-vendor / currency-code /
    date-format logic.
    """
    if field in (SUBTOTAL, TAX, TOTAL):
        try:
            predicted_amount = parse_amount(predicted).amount
        except ValueError:
            return False
        return _amount(truth, predicted_amount) is Outcome.MATCH
    if field == VENDOR:
        return _fuzzy(truth, predicted) is Outcome.MATCH
    if field == DATE:
        return _date(truth, predicted) is Outcome.MATCH
    if field == CURRENCY:
        return _currency(truth, predicted) is Outcome.MATCH
    return False


def _score_field(field: str, truth_value: str, predicted: Invoice) -> Outcome:
    if field == SUBTOTAL:
        return _amount(truth_value, predicted.subtotal.amount)
    if field == TAX:
        return _amount(truth_value, predicted.tax.amount)
    if field == TOTAL:
        return _amount(truth_value, predicted.total.amount)
    if field == VENDOR:
        return _fuzzy(truth_value, predicted.vendor_name)
    if field == DATE:
        return _date(truth_value, predicted.invoice_date)
    if field == CURRENCY:
        return _currency(truth_value, predicted.currency)
    return Outcome.MISMATCH


def _amount(truth_value: str, predicted: Decimal) -> Outcome:
    try:
        expected = parse_amount(truth_value).amount
    except ValueError:
        return Outcome.MISMATCH
    return Outcome.MATCH if abs(predicted - expected) <= _AMOUNT_TOLERANCE else Outcome.MISMATCH


def _fuzzy(truth_value: str, predicted: str) -> Outcome:
    if not predicted.strip():
        return Outcome.MISSED
    ratio = SequenceMatcher(None, truth_value.lower(), predicted.lower()).ratio()
    if ratio >= _VENDOR_THRESHOLD:
        return Outcome.MATCH
    if _vendor_contained(_vendor_tokens(truth_value), _vendor_tokens(predicted)):
        return Outcome.MATCH
    return Outcome.MISMATCH


def _vendor_tokens(value: str) -> frozenset[str]:
    """Significant lower-cased word tokens of a vendor name: punctuation split out,
    a leading "remit to" payee prefix and legal/station suffixes dropped."""
    cleaned = value.lower().strip()
    if cleaned.startswith("remit to"):
        cleaned = cleaned[len("remit to") :]
    tokens = re.split(r"[^a-z0-9]+", cleaned)
    return frozenset(token for token in tokens if token and token not in _VENDOR_SUFFIXES)


def _vendor_contained(left: frozenset[str], right: frozenset[str]) -> bool:
    """True when the smaller name's tokens are wholly contained in the larger's — a
    suffix drop, an abbreviation, or extra trailing words — guarded so a tiny
    fragment can't be read as a match."""
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if not shorter or max(len(token) for token in shorter) < _VENDOR_MIN_TOKEN:
        return False
    return shorter <= longer


def _currency(truth_value: str, predicted: str) -> Outcome:
    if not predicted.strip():
        return Outcome.MISSED
    matched = _currency_code(truth_value) == _currency_code(predicted)
    return Outcome.MATCH if matched else Outcome.MISMATCH


def _currency_code(value: str) -> str:
    stripped = value.strip().lower()
    return _CURRENCY_CODES.get(stripped, stripped.upper())


def _date(truth_value: str, predicted: str) -> Outcome:
    if not predicted.strip():
        return Outcome.MISSED
    expected, got = _to_date(truth_value), _to_date(predicted)
    if expected is not None and got is not None:
        return Outcome.MATCH if expected == got else Outcome.MISMATCH
    return Outcome.MATCH if _canon(truth_value) == _canon(predicted) else Outcome.MISMATCH


def _to_date(value: str) -> date | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _canon(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


@dataclass(frozen=True, slots=True)
class FieldStats:
    field: str
    matched: int
    mismatched: int
    missed: int

    @property
    def support(self) -> int:
        return self.matched + self.mismatched + self.missed

    @property
    def precision(self) -> float:
        predicted = self.matched + self.mismatched
        return self.matched / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


@dataclass(frozen=True, slots=True)
class LineItemStats:
    matched: int
    spurious: int  # predicted lines with no ground-truth match
    missed: int  # ground-truth lines the extractor didn't produce

    @property
    def precision(self) -> float:
        predicted = self.matched + self.spurious
        return self.matched / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        truth = self.matched + self.missed
        return self.matched / truth if truth else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def score_line_items(
    predicted: tuple[LineItem, ...], truth_lines: list[dict[str, str]]
) -> LineItemStats:
    """Row-level line-item match: a predicted line matches a ground-truth line when
    their descriptions are similar and their amounts agree. Greedy one-to-one."""
    used: set[int] = set()
    matched = 0
    for truth in truth_lines:
        index = _best_line(truth, predicted, used)
        if index is not None:
            used.add(index)
            matched += 1
    return LineItemStats(
        matched=matched,
        spurious=len(predicted) - matched,
        missed=len(truth_lines) - matched,
    )


def _best_line(
    truth: dict[str, str], predicted: tuple[LineItem, ...], used: set[int]
) -> int | None:
    truth_amount = _parse(truth.get("amount"))
    for index, line in enumerate(predicted):
        if index in used:
            continue
        ratio = SequenceMatcher(
            None, truth.get("description", "").lower(), line.description.lower()
        )
        if ratio.ratio() < _LINE_DESCRIPTION_THRESHOLD:
            continue
        if (
            truth_amount is not None
            and abs(line.line_total.amount - truth_amount) > _AMOUNT_TOLERANCE
        ):
            continue
        return index
    return None


def _parse(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return parse_amount(value).amount
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class AccuracyReport:
    documents: int
    stats: tuple[FieldStats, ...]
    line_items: LineItemStats | None = None

    @property
    def macro_f1(self) -> float:
        return sum(s.f1 for s in self.stats) / len(self.stats) if self.stats else 0.0


def aggregate(scored: list[dict[str, Outcome]]) -> AccuracyReport:
    tallies: dict[str, list[int]] = {}
    for document in scored:
        for field, outcome in document.items():
            counts = tallies.setdefault(field, [0, 0, 0])
            if outcome is Outcome.MATCH:
                counts[0] += 1
            elif outcome is Outcome.MISMATCH:
                counts[1] += 1
            else:
                counts[2] += 1
    stats = tuple(
        FieldStats(field, matched, mismatched, missed)
        for field, (matched, mismatched, missed) in sorted(tallies.items())
    )
    return AccuracyReport(documents=len(scored), stats=stats)
