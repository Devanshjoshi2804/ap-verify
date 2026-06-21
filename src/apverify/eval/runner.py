"""Run the critic over the synthetic set and its corrupted copies."""

from __future__ import annotations

from apverify.domain.checks import review
from apverify.domain.critique import DEFAULT_POLICY, ApprovalDecision, Policy
from apverify.eval.corruptor import corruptions
from apverify.eval.metrics import CleanOutcome, CorruptOutcome, EvalReport
from apverify.eval.synthetic import faithful_raw_text, generate_dataset


def run_eval(count: int = 25, policy: Policy = DEFAULT_POLICY) -> EvalReport:
    dataset = generate_dataset(count)
    injected = corruptions()

    clean: list[CleanOutcome] = []
    corrupt: list[CorruptOutcome] = []

    for item in dataset:
        page = faithful_raw_text(item.invoice)

        clean_decision = review(item.invoice, page, policy).decision
        clean.append(CleanOutcome(item.label, clean_decision is ApprovalDecision.AUTO_APPROVE))

        for corruption in injected:
            decision = review(corruption.apply(item.invoice), page, policy).decision
            caught = decision is not ApprovalDecision.AUTO_APPROVE
            corrupt.append(CorruptOutcome(item.label, corruption.kind, caught))

    return EvalReport(clean=tuple(clean), corrupt=tuple(corrupt))
