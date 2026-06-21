"""Live field-accuracy test — gated.

Skipped unless APV_DOCILE_PATH points at a local DocILE dataset and GEMINI_API_KEY
is set, since it renders real invoices and runs the vision model on each.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.contract

_READY = bool(os.getenv("APV_DOCILE_PATH") and os.getenv("GEMINI_API_KEY"))


@pytest.mark.skipif(not _READY, reason="set APV_DOCILE_PATH and GEMINI_API_KEY to run")
def test_field_accuracy_runs_over_real_docile() -> None:
    from apverify.eval.accuracy_eval import load_docile_labelled, run_field_accuracy
    from apverify.interface.cli.bootstrap import build_extractor

    path = os.environ["APV_DOCILE_PATH"]
    documents = load_docile_labelled(path, split="val", limit=5)
    assert documents, "no labelled DocILE documents loaded"

    report = run_field_accuracy(documents, build_extractor())
    assert report.stats
    assert all(0.0 <= stat.f1 <= 1.0 for stat in report.stats)
