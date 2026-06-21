# v5 slice 5 — Unified XAI / explanation layer (domain + pipeline)

**Date:** 2026-06-21
**Status:** approved design, pre-implementation
**Arc:** v5 (Fraud, anomaly & the security layer) — fifth sub-project

## Context

v5 shipped three fraud detectors and a cross-detector benchmark. Each detector already
emits a free-text `reason`, and v4's fusion model is an interpretable logistic
regression. Explainability is non-negotiable for finance (SOX-style auditability), so
this slice adds a **unified, structured explanation**: a ranked set of contributing
factors behind any decision, consistent across all sources.

Honesty constraint: true black-box SHAP does not apply to our detectors. The rule
detectors are glass-box (their "attribution" is the conditions that fired); the one
actual model is the **linear** fusion regression, where the exact per-signal
contribution is `weight × feature` — which *is* SHAP for a linear model, in closed
form. No `shap` dependency, nothing approximated or faked.

## Scope

**Domain module + pipeline integration.** A pure `Explanation`/`Factor` type with
builders for every source (fusion model + the three detector assessments), plus deriving
explanations for the live fraud flags on `PayableReview`, plus an `apverify-explain`
demo showing the fusion attribution. No new dependencies.

## Approach (chosen: honest hybrid)

Exact linear attribution for the fusion model (`weight × feature` per signal, ranked);
structured rule-condition attribution for the deterministic detectors. One unified
`Explanation` type so every source reads the same way.

## Architecture & files

| File | Change | Purpose |
|------|--------|---------|
| `src/apverify/domain/explanation.py` | NEW | `Factor`, `Explanation`, `explain_duplicate`, `explain_vendor_risk`, `explain_anomaly` |
| `src/apverify/eval/fusion.py` | EDIT | `explain_fusion(model, row)` → `Explanation` (exact linear contributions) |
| `src/apverify/application/review_payable.py` | EDIT | `PayableReview.explanations` derived from live flags |
| `src/apverify/eval/report.py` | EDIT | `render_explanation` |
| `src/apverify/eval/explanation_cli.py` | NEW | `apverify-explain` (fusion attribution demo) |
| `pyproject.toml` | EDIT | console script |
| tests | NEW | per unit |

## The Explanation type + rule builders (domain)

`Factor(signal: str, value: str, contribution: float, detail: str)` (frozen) — one
driver of a decision. `contribution` is signed; for the rule builders a positive value
means "toward the flag", for the fusion builder it is the signed contribution to the
log-odds of *correct* (so negative pushes toward untrustworthy). The sign convention is
documented on each builder.

`Explanation(source: str, headline: str, factors: tuple[Factor, ...])` (frozen). A
constructor helper sorts factors by `|contribution|` descending so the dominant driver
is first.

Rule builders (each pure, taking the detector's own assessment type):
- `explain_duplicate(match: DuplicateMatch) -> Explanation` — factors for the matched
  tier (e.g. score, matched id) with the tier as the headline
- `explain_vendor_risk(assessment: VendorRiskAssessment) -> Explanation` — factors for
  the kind, the name-similarity score, and the matched vendor
- `explain_anomaly(assessment: AnomalyAssessment) -> Explanation` — factors for the
  score and the top feature (amount_spike / threshold_gaming)

The rule detectors are glass-box, so these enumerate the conditions that fired rather
than approximating a black box.

## Fusion attribution (eval) — the headline

`explain_fusion(model: LogisticRegression, row: FeatureRow) -> Explanation` in
`eval/fusion.py`. For each of the model's `FEATURES`, the contribution to the logit is
`weight_i × feature_value_i`; the bias is reported as a baseline factor. Factors are
ranked by `|contribution|`. The headline states the fused `P(correct)` and the signals
pushing it up or down. This is exact linear SHAP — closed-form, no sampling, no
dependency — turning the v4 fusion score into a ranked, auditable "why".

## Pipeline integration

`PayableReview` gains `explanations: tuple[Explanation, ...]`. In `execute`, after the
vendor-risk and anomaly steps, derive an explanation for each flag that fired —
`explain_vendor_risk(vendor_risk)` when its severity is not NONE, `explain_anomaly(
anomaly)` when its severity is not NONE — and attach the tuple. Pure post-processing of
fields `PayableReview` already carries; no new ports or detectors. Every held payment
now carries structured, ranked attribution beside its free-text reasons.

## Data flow

```
fusion:   LogisticRegression + FeatureRow → explain_fusion → Explanation (ranked weight×feature)
detector: VendorRiskAssessment / AnomalyAssessment / DuplicateMatch → explain_* → Explanation
pipeline: review_payable.execute → for each fired flag, explain_* → PayableReview.explanations
demo:     apverify-explain → fit fusion on synthetic rows → explain_fusion → render_explanation
```

## Error handling

All builders are pure and total. `explain_fusion` requires the model's weights to align
with `FEATURES` (guaranteed by construction). A clean invoice (no flag fired) yields an
empty `explanations` tuple. No I/O beyond the demo CLI.

## Testing (TDD) + acceptance bar

Domain TDD: factors sorted by `|contribution|`; each rule builder's factors and
headline; sign conventions. Eval TDD: `explain_fusion` numerically exact — on a known
model and row, each factor's contribution equals `weight × feature` and the ranking is
correct. Pipeline: `explanations` populated for a flagged invoice (BEC and/or anomaly),
empty for a clean one. Render/CLI: the demo prints a ranked factor table.

Acceptance:
- `explain_fusion` contributions equal `weight × feature` exactly, ranked by magnitude
- each detector assessment yields a non-empty `Explanation` with the deciding factor
  ranked first
- `PayableReview.explanations` carries one explanation per fired flag; empty when clean
- `apverify-explain` prints a ranked attribution
- gates: `ruff check`, `ruff format --check`, `mypy --strict src tests`, `pytest`;
  **domain layer 100% coverage**

## Out of scope (this slice)

Collusion NLP (the remaining v5 sub-project); SHAP plots / a `shap` dependency;
explanations for the critic/3-way-match decisions (the fraud detectors are the focus);
web-UI rendering of explanations.
