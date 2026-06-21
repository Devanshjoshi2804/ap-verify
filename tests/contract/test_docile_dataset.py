"""Live DocILE loader test.

Skipped unless APV_DOCILE_PATH points at a locally downloaded DocILE dataset
(access-gated) and the `docile` extra is installed. The mapping is an unverified
scaffold; this is the test that would validate it once access is available.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.contract

_DOCILE_PATH = os.getenv("APV_DOCILE_PATH")


@pytest.mark.skipif(not _DOCILE_PATH, reason="set APV_DOCILE_PATH to a local DocILE dataset")
def test_docile_loads_and_evaluates() -> None:
    from apverify.eval.dataset_eval import run_dataset_eval
    from apverify.eval.docile import load_docile

    assert _DOCILE_PATH is not None
    examples = load_docile(_DOCILE_PATH, split="val", limit=20)
    assert examples, "no DocILE examples loaded"

    report = run_dataset_eval(examples)
    assert report.total == len(examples)
    assert 0.0 <= report.auto_approve_rate <= 1.0
