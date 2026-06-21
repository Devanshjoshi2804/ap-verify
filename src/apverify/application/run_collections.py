"""Run a collections pass: nudge every overdue receivable, once.

Orchestration only — it asks the domain what (if anything) to send for each
receivable and dispatches it through the ``MessageSender`` port. A send that fails
is recorded and the run continues; one unreachable customer never stops the batch.
The clock is injected so "what is overdue today" is deterministic under test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from apverify.application.errors import PortError
from apverify.application.ports import MessageSender, ReceivablesRepository
from apverify.domain.collections import (
    DEFAULT_COLLECTIONS_POLICY,
    CollectionsPolicy,
    Receivable,
    ReminderTier,
    decide_reminder,
)


@dataclass(frozen=True, slots=True)
class CollectionOutcome:
    receivable: Receivable
    tier: ReminderTier
    message: str
    sent: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CollectionsRun:
    outcomes: tuple[CollectionOutcome, ...]

    @property
    def sent(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.sent)

    @property
    def failed(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.sent)


class RunCollectionsUseCase:
    def __init__(
        self,
        repository: ReceivablesRepository,
        sender: MessageSender,
        today: Callable[[], date],
        policy: CollectionsPolicy = DEFAULT_COLLECTIONS_POLICY,
    ) -> None:
        self._repository = repository
        self._sender = sender
        self._today = today
        self._policy = policy

    def execute(self) -> CollectionsRun:
        as_of = self._today()
        outcomes: list[CollectionOutcome] = []
        for receivable in self._repository.list_receivables():
            decision = decide_reminder(receivable, as_of, self._policy)
            if decision is None:
                continue
            try:
                message_id = self._sender.send(str(receivable.phone), decision.message)
            except PortError as exc:
                outcomes.append(
                    CollectionOutcome(receivable, decision.tier, decision.message, False, str(exc))
                )
                continue
            outcomes.append(
                CollectionOutcome(receivable, decision.tier, decision.message, True, message_id)
            )
        return CollectionsRun(outcomes=tuple(outcomes))
