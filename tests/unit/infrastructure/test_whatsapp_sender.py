from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from apverify.infrastructure.errors import MessagingError
from apverify.infrastructure.whatsapp.sender import WhatsAppMessageSender


def test_sends_text_and_returns_the_message_id() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"messages": [{"id": "wamid.ABC"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = WhatsAppMessageSender("TOKEN", "999000", "v21.0", client=client)

    message_id = sender.send("+919812345678", "Payment reminder")

    assert message_id == "wamid.ABC"
    assert captured["url"] == "https://graph.facebook.com/v21.0/999000/messages"
    assert captured["auth"] == "Bearer TOKEN"
    assert captured["body"]["to"] == "+919812345678"
    assert captured["body"]["text"]["body"] == "Payment reminder"
    assert captured["body"]["messaging_product"] == "whatsapp"


def test_http_error_becomes_a_messaging_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid token"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = WhatsAppMessageSender("BAD", "1", "v21.0", client=client)

    with pytest.raises(MessagingError):
        sender.send("+919812345678", "hi")
