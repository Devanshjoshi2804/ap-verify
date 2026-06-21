from __future__ import annotations

import hashlib
import hmac
import json

from fastapi.testclient import TestClient
from tests.support import (
    PO_NUMBER,
    FakeExtractor,
    FakeOcr,
    FakeRenderer,
    build_goods_receipt,
    build_invoice,
    build_purchase_order,
    build_raw_text,
)

from apverify.application.review_payable import ReviewPayableUseCase
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository
from apverify.interface.api.app import ReviewOptions, WebhookConfig, create_app


def _provider(_options: ReviewOptions) -> ReviewPayableUseCase:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    return ReviewPayableUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(invoice),
        ocr=FakeOcr(build_raw_text(invoice)),
        procurement=InMemoryProcurementRepository(
            purchase_orders=[build_purchase_order()],
            goods_receipts=[build_goods_receipt()],
        ),
    )


_CLIENT = TestClient(create_app(provider=_provider))
_WEBHOOK = WebhookConfig(verify_token="verify-me", app_secret="topsecret")
_WEBHOOK_CLIENT = TestClient(create_app(provider=_provider, webhook=_WEBHOOK))


def test_health() -> None:
    assert _CLIENT.get("/api/health").json() == {"status": "ok"}


def test_webhook_verification_echoes_the_challenge() -> None:
    ok = _WEBHOOK_CLIENT.get(
        "/api/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "verify-me",
            "hub.challenge": "12345",
        },
    )
    assert ok.status_code == 200
    assert ok.text == "12345"

    bad = _WEBHOOK_CLIENT.get(
        "/api/whatsapp/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "x"},
    )
    assert bad.status_code == 403


def test_webhook_verifies_signature_and_classifies_replies() -> None:
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"from": "919812345678", "text": {"body": "Already paid it"}}
                            ]
                        }
                    }
                ]
            }
        ]
    }
    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(b"topsecret", body, hashlib.sha256).hexdigest()

    response = _WEBHOOK_CLIENT.post(
        "/api/whatsapp/webhook", content=body, headers={"X-Hub-Signature-256": signature}
    )

    assert response.status_code == 200
    processed = response.json()["processed"]
    assert processed == [{"from": "919812345678", "text": "Already paid it", "intent": "paid"}]


def test_webhook_rejects_a_bad_signature() -> None:
    response = _WEBHOOK_CLIENT.post(
        "/api/whatsapp/webhook",
        content=b'{"entry": []}',
        headers={"X-Hub-Signature-256": "sha256=deadbeef"},
    )
    assert response.status_code == 403


def test_samples_are_listed_and_downloadable() -> None:
    names = _CLIENT.get("/api/samples").json()
    assert "clean_invoice_01.pdf" in names

    download = _CLIENT.get("/api/samples/clean_invoice_01.pdf")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/pdf"
    assert download.content.startswith(b"%PDF")


def test_unknown_sample_is_rejected() -> None:
    assert _CLIENT.get("/api/samples/../../etc/passwd").status_code in {404, 400}


def test_review_returns_a_full_serialised_decision() -> None:
    response = _CLIENT.post(
        "/api/review",
        files={"file": ("invoice.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"audit": "false", "cross_check": "false"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "AUTO_APPROVE"
    assert body["invoice"]["vendor_name"] == "ACME Steel Pvt Ltd"
    assert body["invoice"]["total"] == "184200.00"
    assert body["match"]["outcome"] == "MATCHED"
    assert {entry["step"] for entry in body["trace"]} >= {"extract", "critic", "match", "approve"}
    assert any(field["field"] == "total" for field in body["fields"])
