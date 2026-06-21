"""A second-opinion verdict from a semantic auditor.

The deterministic critic catches values that aren't on the page or don't add up.
It cannot catch a value that is wrong but *plausible and internally consistent* —
a line item paraphrased into a different product, a total that happens to appear
elsewhere on the page. An LLM auditor reasons over the text and the extracted
fields to give exactly that judgement; an ``AuditVerdict`` is one such opinion.
"""

from __future__ import annotations

from dataclasses import dataclass

from apverify.domain.critique import InvoiceField


@dataclass(frozen=True, slots=True)
class AuditVerdict:
    field: InvoiceField
    trustworthy: bool
    confidence: float
    reason: str
