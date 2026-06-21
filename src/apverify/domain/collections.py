"""Accounts-receivable collections — the flip side of the AP pipeline.

When an invoice we issued goes unpaid, a follow-up agent nudges the customer over
WhatsApp. The tone escalates with how overdue the payment is, and inbound replies
are classified so the system knows whether a payment was claimed, promised, or
disputed. All of this is pure policy; sending and receiving live behind ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from apverify.domain.value_objects import Money, PhoneNumber


class ReminderTier(StrEnum):
    GENTLE = "gentle"
    FIRM = "firm"
    FINAL = "final"


class ReplyIntent(StrEnum):
    PAID = "paid"
    PROMISE_TO_PAY = "promise_to_pay"
    DISPUTE = "dispute"
    QUERY = "query"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Receivable:
    customer_name: str
    phone: PhoneNumber
    invoice_number: str
    amount_due: Money
    currency: str
    due_date: date


@dataclass(frozen=True, slots=True)
class CollectionsDecision:
    tier: ReminderTier
    message: str


@dataclass(frozen=True, slots=True)
class CollectionsPolicy:
    gentle_after_days: int = 1
    firm_after_days: int = 7
    final_after_days: int = 30


DEFAULT_COLLECTIONS_POLICY = CollectionsPolicy()


def days_overdue(due_date: date, today: date) -> int:
    return (today - due_date).days


def decide_reminder(
    receivable: Receivable, today: date, policy: CollectionsPolicy = DEFAULT_COLLECTIONS_POLICY
) -> CollectionsDecision | None:
    """The reminder to send today, or ``None`` if the invoice isn't overdue yet."""
    overdue = days_overdue(receivable.due_date, today)
    if overdue < policy.gentle_after_days:
        return None
    if overdue >= policy.final_after_days:
        tier = ReminderTier.FINAL
    elif overdue >= policy.firm_after_days:
        tier = ReminderTier.FIRM
    else:
        tier = ReminderTier.GENTLE
    return CollectionsDecision(tier=tier, message=_compose(receivable, tier, overdue))


def classify_reply(text: str) -> ReplyIntent:
    """A cheap, deterministic first pass at what a customer's reply means."""
    lowered = text.lower()
    if _any(lowered, ("paid", "payment done", "transferred", "cleared", "made the payment")):
        return ReplyIntent.PAID
    if _any(lowered, ("will pay", "pay by", "pay on", "next week", "tomorrow", "shortly")):
        return ReplyIntent.PROMISE_TO_PAY
    if _any(lowered, ("dispute", "wrong", "incorrect", "not received", "already paid", "mistake")):
        return ReplyIntent.DISPUTE
    if "?" in text or _any(lowered, ("what", "why", "how", "when", "which")):
        return ReplyIntent.QUERY
    return ReplyIntent.UNKNOWN


def _compose(receivable: Receivable, tier: ReminderTier, overdue: int) -> str:
    amount = f"{receivable.currency} {receivable.amount_due}"
    invoice = receivable.invoice_number
    name = receivable.customer_name
    if tier is ReminderTier.GENTLE:
        return (
            f"Hi {name}, a friendly reminder that invoice {invoice} for {amount} is {overdue} "
            f"day(s) past due. Kindly arrange payment at your convenience — thank you!"
        )
    if tier is ReminderTier.FIRM:
        return (
            f"Hi {name}, invoice {invoice} for {amount} is now {overdue} days overdue. "
            f"Please arrange payment promptly to keep your account in good standing."
        )
    return (
        f"Hi {name}, FINAL NOTICE: invoice {invoice} for {amount} is {overdue} days overdue. "
        f"Please settle immediately to avoid further action."
    )


def _any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)
