"""FastAPI application exposing the review pipeline.

The use case is built per request from an injectable provider, so the app is
tested with in-memory fakes and never needs a network or API key in CI. The
default provider wires the real adapters and matches against a procurement file
if one is configured.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from apverify.application.errors import PortError
from apverify.application.review_payable import ReviewPayableUseCase
from apverify.domain.collections import classify_reply
from apverify.infrastructure.procurement_loader import load_procurement
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository
from apverify.infrastructure.settings import Settings
from apverify.interface.api.schemas import ReviewResponse
from apverify.interface.api.serializers import to_response
from apverify.interface.cli.bootstrap import build_review_use_case

_DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


@dataclass(frozen=True, slots=True)
class ReviewOptions:
    audit: bool
    cross_check: bool


@dataclass(frozen=True, slots=True)
class WebhookConfig:
    verify_token: str
    app_secret: str | None


UseCaseProvider = Callable[[ReviewOptions], ReviewPayableUseCase]


def create_app(
    provider: UseCaseProvider | None = None,
    allow_origins: tuple[str, ...] = _DEFAULT_ORIGINS,
    webhook: WebhookConfig | None = None,
) -> FastAPI:
    build = provider or _default_provider
    app = FastAPI(title="ap-verify", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allow_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    resolve_webhook = _webhook_resolver(webhook)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/whatsapp/webhook")
    def verify_webhook(
        mode: Annotated[str, Query(alias="hub.mode")] = "",
        token: Annotated[str, Query(alias="hub.verify_token")] = "",
        challenge: Annotated[str, Query(alias="hub.challenge")] = "",
    ) -> PlainTextResponse:
        config = resolve_webhook()
        if mode == "subscribe" and config.verify_token and token == config.verify_token:
            return PlainTextResponse(challenge)
        raise HTTPException(status_code=403, detail="verification failed")

    @app.post("/api/whatsapp/webhook")
    async def receive_webhook(request: Request) -> dict[str, list[dict[str, str]]]:
        config = resolve_webhook()
        body = await request.body()
        if config.app_secret and not _valid_signature(
            config.app_secret, body, request.headers.get("X-Hub-Signature-256", "")
        ):
            raise HTTPException(status_code=403, detail="bad signature")
        replies = _parse_inbound(json.loads(body or b"{}"))
        processed = [
            {"from": sender, "text": text, "intent": classify_reply(text).value}
            for sender, text in replies
        ]
        return {"processed": processed}

    @app.get("/api/samples")
    def samples() -> list[str]:
        return _sample_names()

    @app.get("/api/samples/{name}")
    def sample(name: str) -> FileResponse:
        # Only names actually present in the bundled directory are served, so the
        # path parameter can never escape it.
        if name not in _sample_names():
            raise HTTPException(status_code=404, detail="unknown sample")
        return FileResponse(_SAMPLES_DIR / name, media_type="application/pdf", filename=name)

    @app.post("/api/review", response_model=ReviewResponse)
    async def review(
        file: Annotated[UploadFile, File()],
        audit: Annotated[bool, Form()] = False,
        cross_check: Annotated[bool, Form()] = False,
    ) -> ReviewResponse:
        document = await _spool(file)
        try:
            result = build(ReviewOptions(audit=audit, cross_check=cross_check)).execute(document)
        except PortError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            document.unlink(missing_ok=True)
        return to_response(result)

    return app


async def _spool(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(await file.read())
        return Path(handle.name)


def _default_provider(options: ReviewOptions) -> ReviewPayableUseCase:
    return build_review_use_case(
        _procurement_repository(),
        enable_audit=options.audit,
        enable_cross_check=options.cross_check,
    )


def _procurement_repository() -> InMemoryProcurementRepository | None:
    configured = os.getenv("APVERIFY_PROCUREMENT_FILE")
    path = Path(configured) if configured else _bundled_procurement()
    return load_procurement(path) if path and path.exists() else None


def _webhook_resolver(injected: WebhookConfig | None) -> Callable[[], WebhookConfig]:
    """Resolve the webhook config once, lazily — so tests inject it and the real
    app reads it from settings only when a webhook request actually arrives."""
    cache: list[WebhookConfig] = [injected] if injected is not None else []

    def resolve() -> WebhookConfig:
        if not cache:
            settings = Settings()
            secret = settings.whatsapp_app_secret
            cache.append(
                WebhookConfig(
                    verify_token=settings.whatsapp_verify_token,
                    app_secret=secret.get_secret_value() if secret else None,
                )
            )
        return cache[0]

    return resolve


def _valid_signature(app_secret: str, body: bytes, header: str) -> bool:
    expected = "sha256=" + hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)


def _parse_inbound(payload: dict[str, Any]) -> list[tuple[str, str]]:
    replies: list[tuple[str, str]] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            for message in change.get("value", {}).get("messages", []):
                sender = message.get("from")
                text = message.get("text", {}).get("body")
                if isinstance(sender, str) and isinstance(text, str):
                    replies.append((sender, text))
    return replies


_SAMPLES_DIR = Path(__file__).resolve().parents[4] / "samples"


def _sample_names() -> list[str]:
    if not _SAMPLES_DIR.is_dir():
        return []
    return sorted(path.name for path in _SAMPLES_DIR.glob("*.pdf"))


def _bundled_procurement() -> Path:
    return _SAMPLES_DIR / "procurement.json"


app = create_app()
