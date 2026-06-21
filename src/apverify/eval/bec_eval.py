"""Score the vendor-master assessor against the labelled BEC benchmark.

The HIGH-flag rate per scenario is the headline: bank-change and impersonation should
flag HIGH every time, while the legitimate scenarios (a known vendor paid to its known
account, a genuinely new vendor) must never flag HIGH. AUROC on the name-similarity
score over impersonation vs legitimate-new measures the one continuous boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apverify.domain.vendor_master import Severity, assess_vendor_risk
from apverify.eval.bec_synthesis import HIGH_SCENARIOS, IMPERSONATION, LEGIT_NEW, BecCase
from apverify.eval.fusion import auroc


@dataclass(frozen=True, slots=True)
class BecReport:
    case_count: int
    catch_rate: float
    false_positive_rate: float
    precision: float
    impersonation_auroc: float
    per_kind: dict[str, float]
    threshold: float


def evaluate_bec(cases: Sequence[BecCase], threshold: float = 0.85) -> BecReport:
    results = [
        (assess_vendor_risk(case.invoice, case.master, threshold), case.scenario) for case in cases
    ]
    flagged = [(assessment.severity is Severity.HIGH, scenario) for assessment, scenario in results]

    high_cases = [f for f, scenario in flagged if scenario in HIGH_SCENARIOS]
    legit_cases = [f for f, scenario in flagged if scenario not in HIGH_SCENARIOS]
    flagged_high = [(f, scenario) for f, scenario in flagged if f]

    impersonation_samples = [
        (assessment.score, scenario == IMPERSONATION)
        for assessment, scenario in results
        if scenario in (IMPERSONATION, LEGIT_NEW)
    ]

    return BecReport(
        case_count=len(cases),
        catch_rate=_rate([f for f in high_cases if f], high_cases),
        false_positive_rate=_rate([f for f in legit_cases if f], legit_cases),
        precision=_rate(
            [scenario for _, scenario in flagged_high if scenario in HIGH_SCENARIOS], flagged_high
        ),
        impersonation_auroc=auroc(impersonation_samples),
        per_kind=_per_kind(flagged),
        threshold=threshold,
    )


def _rate(hits: Sequence[object], population: Sequence[object]) -> float:
    return len(hits) / len(population) if population else 0.0


def _per_kind(flagged: Sequence[tuple[bool, str]]) -> dict[str, float]:
    scenarios = sorted({scenario for _, scenario in flagged})
    return {
        scenario: _rate(
            [f for f, s in flagged if s == scenario and f],
            [f for f, s in flagged if s == scenario],
        )
        for scenario in scenarios
    }
