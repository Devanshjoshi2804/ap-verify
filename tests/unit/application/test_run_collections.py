from __future__ import annotations

from datetime import date

from tests.support import FailingSender, RecordingSender, build_receivable

from apverify.application.run_collections import RunCollectionsUseCase
from apverify.domain.collections import ReminderTier
from apverify.infrastructure.receivables_memory import InMemoryReceivablesRepository

_TODAY = date(2026, 6, 30)


def _today() -> date:
    return _TODAY


def test_sends_to_overdue_and_skips_not_yet_due() -> None:
    overdue = build_receivable(invoice_number="AR-1", due_date=date(2026, 6, 20))
    future = build_receivable(invoice_number="AR-2", due_date=date(2026, 7, 10))
    sender = RecordingSender()
    use_case = RunCollectionsUseCase(
        InMemoryReceivablesRepository([overdue, future]), sender, today=_today
    )

    run = use_case.execute()

    assert run.sent == 1
    assert len(sender.sent) == 1
    assert run.outcomes[0].tier is ReminderTier.FIRM


def test_send_failure_is_recorded_and_does_not_stop_the_batch() -> None:
    receivables = [
        build_receivable(invoice_number="AR-1", due_date=date(2026, 6, 1)),
        build_receivable(invoice_number="AR-2", due_date=date(2026, 6, 1)),
    ]
    use_case = RunCollectionsUseCase(
        InMemoryReceivablesRepository(receivables), FailingSender(), today=_today
    )

    run = use_case.execute()

    assert run.failed == 2
    assert all(not outcome.sent for outcome in run.outcomes)
