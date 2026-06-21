from __future__ import annotations

from apverify.domain.checks import review
from apverify.domain.critique import ApprovalDecision
from apverify.eval.corruptor import corruptions
from apverify.eval.runner import run_eval
from apverify.eval.synthetic import faithful_raw_text, generate_dataset


def test_dataset_is_deterministic() -> None:
    first = generate_dataset(5)
    second = generate_dataset(5)
    assert [g.invoice for g in first] == [g.invoice for g in second]


def test_every_clean_synthetic_invoice_auto_approves() -> None:
    for item in generate_dataset(25):
        report = review(item.invoice, faithful_raw_text(item.invoice))
        assert report.decision is ApprovalDecision.AUTO_APPROVE, item.label


def test_every_corruption_is_caught_on_every_invoice() -> None:
    for item in generate_dataset(25):
        page = faithful_raw_text(item.invoice)
        for corruption in corruptions():
            decision = review(corruption.apply(item.invoice), page).decision
            assert decision is not ApprovalDecision.AUTO_APPROVE, (item.label, corruption.kind)


def test_eval_report_hits_the_trust_targets() -> None:
    report = run_eval(25)

    assert report.catch_rate == 1.0
    assert report.false_hold_rate == 0.0
    assert report.safe_auto_approval_rate == 1.0
    assert report.escaped == 0
    assert all(score.catch_rate == 1.0 for score in report.per_kind())
