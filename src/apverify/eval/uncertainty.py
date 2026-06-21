"""Sampling-based uncertainty — does the model agree with itself?

Resampling an extraction a few times at non-zero temperature turns one answer into a
distribution, and two readings of that distribution estimate how sure the model is:

* **self-consistency** — the share of samples on the modal answer (SelfCheckGPT-style).
  High agreement is evidence the field is real.
* **semantic entropy** (Kuhn, Gal & Farquhar, 2024) — Shannon entropy over *meaning*
  clusters, not raw strings: ``"184200"`` and ``"184200.00"`` are the same answer and
  must not count as disagreement. Low entropy = the mass sits on one meaning = confident.

Both are pure over a list of sampled values plus an equivalence relation; producing the
samples (calling a model N times) is an infrastructure concern, kept out of here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

Equivalent = Callable[[str, str], bool]


def _exact(left: str, right: str) -> bool:
    return left.strip().casefold() == right.strip().casefold()


def cluster_values(values: Sequence[str], equivalent: Equivalent = _exact) -> list[list[str]]:
    """Greedily group values that mean the same thing. Greedy (first matching cluster)
    is enough here: the relations are near-transitive and the sample counts tiny."""
    clusters: list[list[str]] = []
    for value in values:
        for cluster in clusters:
            if equivalent(value, cluster[0]):
                cluster.append(value)
                break
        else:
            clusters.append([value])
    return clusters


def self_consistency(values: Sequence[str], equivalent: Equivalent = _exact) -> float:
    """Fraction of samples that agree with the modal answer, in ``[0, 1]``."""
    if not values:
        return 0.0
    clusters = cluster_values(values, equivalent)
    return max(len(cluster) for cluster in clusters) / len(values)


def semantic_entropy(values: Sequence[str], equivalent: Equivalent = _exact) -> float:
    """Shannon entropy (nats) over meaning-clusters. ``0`` when all samples agree."""
    if not values:
        return 0.0
    total = len(values)
    probabilities = [len(cluster) / total for cluster in cluster_values(values, equivalent)]
    return -sum(p * math.log(p) for p in probabilities)


def confidence_from_entropy(entropy: float) -> float:
    """Map semantic entropy (0 = certain, higher = less) to a ``(0, 1]`` confidence so it
    can be calibrated and compared on the same axis as the other signals."""
    return math.exp(-entropy)
