"""Live WhatsApp send test — opt-in, never runs by default.

Sends one real message, so it is skipped unless WHATSAPP_ACCESS_TOKEN,
WHATSAPP_PHONE_NUMBER_ID and an explicit WHATSAPP_TEST_RECIPIENT (a number you own
and have opted in) are all set. This guard exists so the suite can never message a
real customer by accident.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.contract

_READY = bool(
    os.getenv("WHATSAPP_ACCESS_TOKEN")
    and os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    and os.getenv("WHATSAPP_TEST_RECIPIENT")
)


@pytest.mark.skipif(not _READY, reason="set WHATSAPP_TEST_RECIPIENT (a number you own) to run")
def test_whatsapp_send_returns_a_message_id() -> None:
    from apverify.infrastructure.whatsapp.sender import WhatsAppMessageSender

    sender = WhatsAppMessageSender(
        access_token=os.environ["WHATSAPP_ACCESS_TOKEN"],
        phone_number_id=os.environ["WHATSAPP_PHONE_NUMBER_ID"],
        api_version=os.getenv("WHATSAPP_API_VERSION", "v21.0"),
    )
    message_id = sender.send(os.environ["WHATSAPP_TEST_RECIPIENT"], "ap-verify test message")
    assert message_id
