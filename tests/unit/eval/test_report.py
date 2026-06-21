from __future__ import annotations

from rich.console import Console

from apverify.domain.explanation import Factor, explanation
from apverify.eval.anomaly_eval import AnomalyReport, DetectorResult
from apverify.eval.bec_eval import BecReport
from apverify.eval.collusion_eval import CollusionReport
from apverify.eval.fraud_eval import FraudOperatingPoint, FraudReport
from apverify.eval.fraud_suite_eval import FraudSuiteReport
from apverify.eval.report import (
    render_anomaly,
    render_bec,
    render_collusion,
    render_explanation,
    render_fraud,
    render_fraud_suite,
)


def test_render_collusion_prints_headline() -> None:
    report = CollusionReport(
        pair_count=9,
        colluding_count=3,
        catch_rate=1.0,
        false_positive_rate=0.0,
        precision=1.0,
        auroc=1.0,
    )
    console = Console(record=True, width=100)
    render_collusion(report, console)
    assert "collusion" in console.export_text().lower()


def test_render_explanation_prints_ranked_factors() -> None:
    exp = explanation(
        "fusion",
        "P(correct) 0.30",
        [Factor("arithmetic_passed", "0.00", -1.2, "weight -1.20 x 0.00")],
    )
    console = Console(record=True, width=100)
    render_explanation(exp, console)
    text = console.export_text()
    assert "arithmetic_passed" in text
    assert "P(correct)" in text


def test_render_fraud_suite_prints_combined_and_per_label() -> None:
    report = FraudSuiteReport(
        case_count=35,
        fraud_count=30,
        catch_rate=1.0,
        false_positive_rate=0.0,
        precision=1.0,
        per_label={"dup_resend": 1.0, "clean": 0.0},
        per_detector={"duplicate": 10, "bec": 10, "anomaly": 10},
    )
    console = Console(record=True, width=100)
    render_fraud_suite(report, console)
    text = console.export_text()
    assert "dup_resend" in text
    assert "fraud" in text.lower()


def test_render_anomaly_prints_each_detector() -> None:
    report = AnomalyReport(
        case_count=15,
        anomaly_count=10,
        results=(DetectorResult("robust-statistics", 0.97, 1.0, 0.0),),
        sklearn_available=False,
    )
    console = Console(record=True, width=100)
    render_anomaly(report, console)
    text = console.export_text()
    assert "robust-statistics" in text
    assert "anomaly" in text.lower()


def test_render_bec_prints_catch_and_false_positive() -> None:
    report = BecReport(
        case_count=25,
        catch_rate=1.0,
        false_positive_rate=0.0,
        precision=1.0,
        impersonation_auroc=0.98,
        per_kind={"bank_change": 1.0, "known_clean": 0.0},
        threshold=0.85,
    )
    console = Console(record=True, width=100)
    render_bec(report, console)
    text = console.export_text()
    assert "bank_change" in text
    assert "BEC" in text or "vendor" in text.lower()


def test_render_fraud_prints_catch_and_false_positive() -> None:
    report = FraudReport(
        case_count=12,
        fraud_count=8,
        threshold=0.05,
        catch_rate=1.0,
        false_positive_rate=0.0,
        precision=1.0,
        auroc=0.97,
        sweep=(FraudOperatingPoint(0.05, 1.0, 0.0),),
        per_kind={"exact_resend": 1.0, "legit_recurring": 0.0},
    )
    console = Console(record=True, width=100)
    render_fraud(report, console)
    text = console.export_text()
    assert "catch" in text.lower()
    assert "exact_resend" in text
