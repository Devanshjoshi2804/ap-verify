# v5 slice 3 — Anomaly detection (eval + pipeline wiring)

**Date:** 2026-06-20
**Status:** approved design, pre-implementation
**Arc:** v5 (Fraud, anomaly & the security layer) — third of six sub-projects

## Context

v5 slices 1 (duplicate detection) and 2 (vendor-master / BEC) are shipped. This slice
adds **anomaly detection**: flag invoices that are statistically unusual for their
vendor — an amount far above the vendor's history (a spike), or an amount parked just
under a round approval limit (threshold gaming). These catch fraud and error the
deterministic checks miss.

The project has deliberately avoided ML dependencies (no scikit-learn / scipy / torch
declared; numpy is only a transitive dependency). Its differentiator is *measurable,
honest, clean* — so the detector is a pure robust-statistics model by default, with
scikit-learn's Isolation Forest as an optional, eval-only challenger measured
head-to-head. ML earns its dependency only if it wins.

## Scope

**Eval + pipeline wiring.** A pure domain detector + feature extraction, an
`AnomalyDetector` port and a `VendorHistoryRepository` port, a live anomaly stage in
the approval pipeline (HIGH → HOLD), a synthetic benchmark comparing the pure detector
against Isolation Forest, and XAI (the top contributing feature per flag). The
sklearn adapter is **eval-only** — it never sits on the production path, so the running
pipeline stays dependency-free.

## Approach (chosen: pure baseline + optional sklearn, head-to-head)

The pure `RobustAnomalyDetector` uses median/MAD robust z-scores (a single outlier
cannot mask the next) plus a threshold-proximity signal. Isolation Forest is wired only
into the benchmark, behind an optional extra; if scikit-learn is not installed the
benchmark reports the pure detector alone and says so. Rejected: pure-only (no ML
comparison, the roadmap explicitly names Isolation Forest) and sklearn-only (adds a
hard dependency, no honest baseline, and puts ML on the production path).

## Architecture & files

| File | Change | Purpose |
|------|--------|---------|
| `src/apverify/domain/anomaly.py` | NEW | `AnomalyFeatures`, `AnomalySeverity`, `AnomalyAssessment`, `extract_features`, `RobustAnomalyDetector` (pure) |
| `src/apverify/domain/approval.py` | EDIT | `reconcile_with_anomaly` (HIGH→HOLD, MEDIUM→HUMAN_REVIEW, else reason) |
| `src/apverify/application/ports.py` | EDIT | `AnomalyDetector` + `VendorHistoryRepository` ports |
| `src/apverify/application/review_payable.py` | EDIT | optional anomaly step → reconcile; `PayableReview.anomaly` |
| `src/apverify/infrastructure/anomaly/__init__.py` | NEW | package marker |
| `src/apverify/infrastructure/anomaly/history.py` | NEW | `InMemoryVendorHistory` + `load_vendor_history` (JSON) |
| `src/apverify/infrastructure/anomaly/isolation_forest.py` | NEW | sklearn `IsolationForestDetector` (optional extra, eval-only) |
| `src/apverify/infrastructure/settings.py` | EDIT | `anomaly_history_path` |
| `src/apverify/interface/cli/bootstrap.py` | EDIT | wire pure detector + history when configured |
| `src/apverify/eval/anomaly_synthesis.py` | NEW | `build_anomaly_cases` (amount-spike / threshold-gaming + normals) |
| `src/apverify/eval/anomaly_eval.py` | NEW | `evaluate_anomaly` → head-to-head `AnomalyReport` |
| `src/apverify/eval/anomaly_cli.py` | NEW | `apverify-eval-anomaly` |
| `src/apverify/eval/report.py` | EDIT | `render_anomaly` |
| `pyproject.toml` | EDIT | `[project.optional-dependencies] anomaly = ["scikit-learn>=1.5"]`; console script |
| tests | NEW | per unit |

## Domain detector + features

`extract_features(invoice: Invoice, history: Sequence[Invoice]) -> AnomalyFeatures`
computes, relative to the vendor's historical totals:
- `amount_robust_z`: `|amount - median| / (MAD or a small floor)` over history totals —
  MAD-based so one historical outlier cannot hide the next anomaly
- `threshold_proximity`: in `[0,1]`, how close the amount sits *just below* a round
  approval limit (e.g. 9,950 against 10,000); 0 when not near a limit from below
- `history_size`: number of prior invoices (drives abstention)

`AnomalyFeatures`: `amount_robust_z: float`, `threshold_proximity: float`,
`history_size: int` (frozen).

`AnomalySeverity`: `NONE`, `MEDIUM`, `HIGH`.

`AnomalyAssessment`: `score: float ∈ [0,1]`, `severity: AnomalySeverity`,
`top_feature: str`, `reason: str` (frozen).

`RobustAnomalyDetector.score(invoice, history) -> AnomalyAssessment`: with fewer than a
minimum number of prior invoices (default 3) it abstains (`score 0.0`, severity NONE,
reason "insufficient history"). Otherwise `score = max(1 - exp(-robust_z / k),
threshold_proximity)` — either signal alone can raise the flag — and severity tiers
from fixed score cutoffs (e.g. ≥0.8 HIGH, ≥0.5 MEDIUM, else NONE). `top_feature` names the dominant signal; the reason
is specific, e.g. *"amount 1840000 is 11.2x the vendor's median 164000 (robust-z 9.2)"*
or *"amount 9950 sits just under the 10000 approval limit"*. Pure, deterministic, no
dependencies. The class structurally satisfies the `AnomalyDetector` port.

Approval thresholds for `threshold_proximity` are a fixed, documented tuple of round
limits (e.g. 10_000, 50_000, 100_000) — a named constant, configurable on the detector,
not a magic literal.

## Ports + pipeline wiring

`AnomalyDetector` (runtime-checkable Protocol): `score(invoice: Invoice,
history: Sequence[Invoice]) -> AnomalyAssessment`. `VendorHistoryRepository`
(Protocol): `history_for(invoice: Invoice) -> tuple[Invoice, ...]`.

`ReviewPayableUseCase` gains optional `anomaly_detector` and `vendor_history`
dependencies; a new `_anomaly` step (mirrors `_vendor_risk`) fetches history, scores,
records to the trace, and `reconcile_with_anomaly` folds the result in.
`PayableReview.anomaly: AnomalyAssessment | None`.

`reconcile_with_anomaly(decision, assessment, policy)` in `domain/approval.py`:
HIGH → escalate to HOLD; MEDIUM → escalate to HUMAN_REVIEW; NONE → unchanged. A reason
is appended whenever the severity is not NONE. Never lowers an existing decision.

`InMemoryVendorHistory(invoices)` groups prior invoices by vendor name and answers
`history_for`. `load_vendor_history(path)` reads a JSON list of past invoices (reusing
the existing invoice DTO shape). Bootstrap wires the **pure** `RobustAnomalyDetector`
plus the history repository behind `anomaly_history_path`; absent → the step is skipped
and the pipeline is unchanged. The sklearn detector is never wired here.

## Benchmark (pure vs sklearn head-to-head)

`build_anomaly_cases(base) -> list[AnomalyCase]` where
`AnomalyCase = (invoice, history, is_anomaly, kind)`. For each base vendor it
synthesises a plausible history (amounts around a vendor median), then emits:
- `amount_spike` (anomaly) — amount many times the vendor's median
- `threshold_gaming` (anomaly) — amount just under a round approval limit
- `normal` (hard negative) — an amount within the vendor's usual spread

`evaluate_anomaly(cases) -> AnomalyReport`: scores every case with the pure detector,
and — only if scikit-learn is importable — with the Isolation Forest detector, reporting
per detector: AUROC of the score separating anomalies from normals, catch-rate, and
false-positive rate on `normal`. `AnomalyReport` carries a per-detector result list and
a flag for whether sklearn was available. `render_anomaly` prints the head-to-head
table; `apverify-eval-anomaly --count N`.

## Data flow

```
synthetic base invoices
  → anomaly_synthesis.build_anomaly_cases → [AnomalyCase(invoice, history, is_anomaly, kind)]
  → anomaly_eval: pure detector .score per case (+ IsolationForest if installed)
  → per detector: (score, is_anomaly) → AUROC, catch-rate, FP
  → AnomalyReport → report.render_anomaly

live: review_payable._anomaly → vendor_history.history_for(invoice)
  → anomaly_detector.score(invoice, history) → reconcile_with_anomaly → FinalDecision
```

## Error handling

The detector is pure and total: insufficient history → abstain (score 0). MAD of zero
(all historical amounts identical) is floored so the robust-z does not divide by zero.
The JSON history adapter raises a clear error on a missing/malformed file; an absent
`anomaly_history_path` omits the step. The Isolation Forest import is guarded — its
absence is reported, never raised, in the benchmark, and it is never imported on the
production path.

## Testing (TDD) + acceptance bar

Pure domain TDD: robust-z on a known series (and the all-equal / zero-MAD floor),
threshold-proximity (just-under vs not-near), severity tiers, insufficient-history
abstention, reason content. Approval: HIGH→HOLD, MEDIUM→HUMAN_REVIEW, never lowers.
Pipeline: an amount-spike invoice drives the use case to HOLD; a normal invoice is
unaffected. Eval: injection labels, pure-detector AUROC, and the sklearn-absent path
degrades cleanly (pure-only report). Infra: history adapter parses a fixture and
satisfies the port.

Acceptance:
- amount-spike + threshold-gaming caught ≥ 90% at 0% false-positive on `normal`
  (pure detector)
- pure-detector AUROC ≥ 0.9 on synthetic
- Isolation Forest comparison reported when scikit-learn is installed; pure-only
  reported (with a note) when it is not
- pipeline HOLD on an amount-spike invoice verified end-to-end
- gates: `ruff check`, `ruff format --check`, `mypy --strict src tests`, `pytest`;
  **domain layer 100% coverage**

## Out of scope (this slice)

Autoencoder / LSTM temporal models; true multi-invoice split-invoice detection (this
slice approximates threshold gaming per invoice); collusion NLP; the broader XAI/SHAP
layer; the cross-detector fraud benchmark. Each is a later v5 slice.
