"""In-memory receivables repository."""

from __future__ import annotations

from collections.abc import Iterable

from apverify.domain.collections import Receivable


class InMemoryReceivablesRepository:
    def __init__(self, receivables: Iterable[Receivable] = ()) -> None:
        self._receivables = list(receivables)

    def list_receivables(self) -> list[Receivable]:
        return list(self._receivables)
