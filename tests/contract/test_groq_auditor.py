"""Live contract test for the Groq auditor.

Skipped unless GROQ_API_KEY is set, since it makes a real, billable API call. It
checks the adapter honours the ``SemanticAuditor`` port: given a clearly wrong
total against the document text, it returns a verdict for that field.
"""

from __future__ import annotations

import os

import pytest
from tests.support import build_invoice, build_raw_text

from apverify.domain.critique import InvoiceField

pytestmark = pytest.mark.contract


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="no GROQ_API_KEY in environment")
def test_groq_auditor_returns_a_verdict_for_the_audited_field() -> None:
    from groq import Groq

    from apverify.infrastructure.groq.auditor import GroqSemanticAuditor

    page_invoice = build_invoice()
    auditor = GroqSemanticAuditor(
        Groq(api_key=os.environ["GROQ_API_KEY"]),
        os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )

    verdicts = auditor.audit(page_invoice, build_raw_text(page_invoice), [InvoiceField.TOTAL])

    assert [verdict.field for verdict in verdicts] == [InvoiceField.TOTAL]
    assert 0.0 <= verdicts[0].confidence <= 1.0
