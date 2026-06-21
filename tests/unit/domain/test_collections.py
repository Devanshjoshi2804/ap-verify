from __future__ import annotations

from datetime import date

import pytest
from tests.support import build_receivable

from apverify.domain.collections import (
    CollectionsPolicy,
    ReminderTier,
    ReplyIntent,
    classify_reply,
    decide_reminder,
)
from apverify.domain.errors import InvalidPhoneNumberError
from apverify.domain.value_objects import PhoneNumber

_TODAY = date(2026, 6, 30)


@pytest.mark.parametrize(
    ("due", "expected"),
    [
        (date(2026, 7, 5), None),  # not due yet
        (date(2026, 6, 29), ReminderTier.GENTLE),  # 1 day
        (date(2026, 6, 20), ReminderTier.FIRM),  # 10 days
        (date(2026, 5, 1), ReminderTier.FINAL),  # 60 days
    ],
)
def test_reminder_tier_escalates_with_age(due: date, expected: ReminderTier | None) -> None:
    decision = decide_reminder(build_receivable(due_date=due), _TODAY)
    assert (decision.tier if decision else None) == expected


def test_message_names_the_customer_and_invoice() -> None:
    decision = decide_reminder(build_receivable(due_date=date(2026, 6, 20)), _TODAY)
    assert decision is not None
    assert "Acme Buyer Ltd" in decision.message
    assert "AR-2025-0001" in decision.message


def test_policy_thresholds_are_configurable() -> None:
    strict = CollectionsPolicy(gentle_after_days=0)
    assert decide_reminder(build_receivable(due_date=_TODAY), _TODAY, strict) is not None


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Already paid yesterday", ReplyIntent.PAID),
        ("I will pay by tomorrow", ReplyIntent.PROMISE_TO_PAY),
        ("This amount is incorrect", ReplyIntent.DISPUTE),
        ("When is it due?", ReplyIntent.QUERY),
        ("ok", ReplyIntent.UNKNOWN),
    ],
)
def test_reply_classification(text: str, intent: ReplyIntent) -> None:
    assert classify_reply(text) == intent


def test_phone_number_validates_e164() -> None:
    assert PhoneNumber(" +91 98123 45678 ").value == "+919812345678"
    with pytest.raises(InvalidPhoneNumberError):
        PhoneNumber("98123")
