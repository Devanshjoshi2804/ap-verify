"""WhatsApp Cloud API sender implementing the ``MessageSender`` port.

Posts a text message to the Graph API. The HTTP client is injected so the adapter
is tested against a fake transport without touching the network; a failed send
surfaces as a ``MessagingError`` the use case records and moves past.
"""

from __future__ import annotations

import httpx

from apverify.infrastructure.errors import MessagingError


class WhatsAppMessageSender:
    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        api_version: str = "v21.0",
        client: httpx.Client | None = None,
    ) -> None:
        self._token = access_token
        self._url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
        self._client = client or httpx.Client(timeout=15.0)

    def send(self, phone: str, text: str) -> str:
        payload = {
            "messaging_product": "whatsapp",
            "to": phone,
            "type": "text",
            "text": {"body": text},
        }
        try:
            response = self._client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._token}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise MessagingError(f"WhatsApp send to {phone} failed: {exc}") from exc

        messages = data.get("messages") or []
        if not messages:
            raise MessagingError(f"WhatsApp returned no message id for {phone}")
        return str(messages[0].get("id", ""))
