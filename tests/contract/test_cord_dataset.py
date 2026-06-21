"""Live CORD-v2 loader test.

Skipped unless APV_CORD_LIVE=1, since it downloads the dataset (hundreds of MB)
and needs the optional `datasets` extra installed. Verifies the loader maps real
records into domain invoices and the critic runs over them.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.contract


@pytest.mark.skipif(
    not os.getenv("APV_CORD_LIVE"), reason="set APV_CORD_LIVE=1 to download CORD-v2"
)
def test_cord_loads_and_evaluates() -> None:
    from apverify.eval.cord import load_cord
    from apverify.eval.dataset_eval import run_dataset_eval

    examples = load_cord(split="test", limit=20)
    assert examples, "no CORD examples loaded"

    report = run_dataset_eval(examples)
    assert report.total == len(examples)
    assert 0.0 <= report.auto_approve_rate <= 1.0
