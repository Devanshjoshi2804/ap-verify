"""Eval results and the trust metrics derived from them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CleanOutcome:
    label: str
    auto_approved: bool


@dataclass(frozen=True, slots=True)
class CorruptOutcome:
    label: str
    kind: str
    caught: bool


@dataclass(frozen=True, slots=True)
class KindScore:
    kind: str
    caught: int
    total: int

    @property
    def catch_rate(self) -> float:
        return self.caught / self.total if self.total else 1.0


@dataclass(frozen=True, slots=True)
class EvalReport:
    clean: tuple[CleanOutcome, ...]
    corrupt: tuple[CorruptOutcome, ...]

    @property
    def clean_count(self) -> int:
        return len(self.clean)

    @property
    def corrupt_count(self) -> int:
        return len(self.corrupt)

    @property
    def clean_auto_approved(self) -> int:
        return sum(1 for outcome in self.clean if outcome.auto_approved)

    @property
    def escaped(self) -> int:
        """Corrupted invoices that were auto-approved anyway — the dangerous case."""
        return sum(1 for outcome in self.corrupt if not outcome.caught)

    @property
    def catch_rate(self) -> float:
        """Share of injected errors flagged before approval (the headline number)."""
        if not self.corrupt_count:
            return 1.0
        return (self.corrupt_count - self.escaped) / self.corrupt_count

    @property
    def false_hold_rate(self) -> float:
        """Share of clean invoices wrongly withheld from auto-approval."""
        if not self.clean_count:
            return 0.0
        return (self.clean_count - self.clean_auto_approved) / self.clean_count

    @property
    def safe_auto_approval_rate(self) -> float:
        """Of everything auto-approved, the share that was actually clean (precision)."""
        auto_approved = self.clean_auto_approved + self.escaped
        return self.clean_auto_approved / auto_approved if auto_approved else 1.0

    def per_kind(self) -> tuple[KindScore, ...]:
        kinds = sorted({outcome.kind for outcome in self.corrupt})
        scores = []
        for kind in kinds:
            outcomes = [o for o in self.corrupt if o.kind == kind]
            scores.append(KindScore(kind, sum(o.caught for o in outcomes), len(outcomes)))
        return tuple(scores)

    def to_snapshot(self) -> EvalSnapshot:
        """A flat, JSON-serialisable summary for comparing runs over time."""
        return EvalSnapshot(
            clean_count=self.clean_count,
            corrupt_count=self.corrupt_count,
            catch_rate=self.catch_rate,
            false_hold_rate=self.false_hold_rate,
            safe_auto_approval_rate=self.safe_auto_approval_rate,
            escaped=self.escaped,
            per_kind={score.kind: score.catch_rate for score in self.per_kind()},
        )


@dataclass(frozen=True, slots=True)
class EvalSnapshot:
    clean_count: int
    corrupt_count: int
    catch_rate: float
    false_hold_rate: float
    safe_auto_approval_rate: float
    escaped: int
    per_kind: dict[str, float]
