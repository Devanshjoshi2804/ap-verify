"""Score the duplicate matcher against a labelled fraud benchmark.

For each case we take the candidate's best duplicate score against the ledger (0 if
the matcher finds nothing non-DISTINCT), then report the two numbers that matter for a
fraud control: catch-rate (recall on true duplicates) and false-positive-rate (legit
invoices wrongly flagged). A threshold sweep gives the catch-rate-vs-false-positive
curve and the safe (zero-false-positive) operating point, mirroring the v4 risk-
coverage view; AUROC summarises how well the score separates fraud from legitimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from apverify.domain.fraud import find_duplicates
from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.value_objects import Money
from apverify.eval.accuracy_eval import LabelledDocument
from apverify.eval.fraud_synthesis import FraudCase
from apverify.eval.fusion import auroc


@dataclass(frozen=True, slots=True)
class FraudOperatingPoint:
    threshold: float
    catch_rate: float
    false_positive_rate: float


@dataclass(frozen=True, slots=True)
class FraudReport:
    case_count: int
    fraud_count: int
    threshold: float
    catch_rate: float
    false_positive_rate: float
    precision: float
    auroc: float
    sweep: tuple[FraudOperatingPoint, ...]
    per_kind: dict[str, float]


def evaluate_fraud(cases: Sequence[FraudCase], threshold: float | None = None) -> FraudReport:
    scored = [(_best_score(case), case.is_fraud, case.kind) for case in cases]
    samples = [(score, is_fraud) for score, is_fraud, _ in scored]
    sweep = _sweep(samples)
    chosen = threshold if threshold is not None else _zero_fp_threshold(sweep)
    flagged = [(score >= chosen, is_fraud, kind) for score, is_fraud, kind in scored]

    frauds = [s for s in flagged if s[1]]
    legit = [s for s in flagged if not s[1]]
    caught = [s for s in frauds if s[0]]
    false_pos = [s for s in legit if s[0]]
    flagged_total = [s for s in flagged if s[0]]

    return FraudReport(
        case_count=len(cases),
        fraud_count=len(frauds),
        threshold=chosen,
        catch_rate=len(caught) / len(frauds) if frauds else 0.0,
        false_positive_rate=len(false_pos) / len(legit) if legit else 0.0,
        precision=len(caught) / len(flagged_total) if flagged_total else 0.0,
        auroc=auroc(samples),
        sweep=sweep,
        per_kind=_per_kind(flagged),
    )


def invoice_from_labelled(document: LabelledDocument) -> Invoice:
    """Build an Invoice from a dataset document's ground-truth fields, so the benchmark
    runs on real invoices without a model call (quota-free realism check)."""
    truth = document.truth
    total = Money.of(_amount(truth.get("total", "0")))
    subtotal = Money.of(_amount(truth.get("subtotal", truth.get("total", "0"))))
    return Invoice(
        vendor_name=truth.get("vendor_name", ""),
        invoice_number=truth.get("invoice_number", ""),
        invoice_date=truth.get("invoice_date", ""),
        currency=truth.get("currency", "INR"),
        subtotal=subtotal,
        tax=Money.of(0),
        total=total,
        line_items=(LineItem("", 1, total, total),),
        tax_breakdown=TaxBreakdown(),
        vendor_gstin=truth.get("vendor_gstin", ""),
        purchase_order_ref="",
    )


def _best_score(case: FraudCase) -> float:
    matches = find_duplicates(case.candidate, case.priors)
    return matches[0].score if matches else 0.0


def _sweep(
    samples: Sequence[tuple[float, bool]], steps: int = 20
) -> tuple[FraudOperatingPoint, ...]:
    frauds = [s for s in samples if s[1]]
    legit = [s for s in samples if not s[1]]
    points: list[FraudOperatingPoint] = []
    for step in range(steps + 1):
        threshold = step / steps
        caught = sum(1 for score, _ in frauds if score >= threshold)
        false_pos = sum(1 for score, _ in legit if score >= threshold)
        points.append(
            FraudOperatingPoint(
                threshold=threshold,
                catch_rate=caught / len(frauds) if frauds else 0.0,
                false_positive_rate=false_pos / len(legit) if legit else 0.0,
            )
        )
    return tuple(points)


def _zero_fp_threshold(sweep: Sequence[FraudOperatingPoint]) -> float:
    """The lowest threshold with no false positives (most catch at zero FP); 1.0 if
    none is clean."""
    clean = [point for point in sweep if point.false_positive_rate == 0.0]
    if not clean:
        return 1.0
    return min(clean, key=lambda point: point.threshold).threshold


def _per_kind(flagged: Sequence[tuple[bool, bool, str]]) -> dict[str, float]:
    kinds = sorted({kind for _, _, kind in flagged})
    result: dict[str, float] = {}
    for kind in kinds:
        rows = [is_flagged for is_flagged, _, k in flagged if k == kind]
        result[kind] = sum(1 for flag in rows if flag) / len(rows) if rows else 0.0
    return result


def _amount(value: str) -> str:
    """A clean decimal string for ``Money.of``; ``"0"`` when the field is unparseable."""
    cleaned = str(value).replace(",", "").strip()
    try:
        Decimal(cleaned)
    except (ArithmeticError, ValueError):
        return "0"
    return cleaned or "0"
