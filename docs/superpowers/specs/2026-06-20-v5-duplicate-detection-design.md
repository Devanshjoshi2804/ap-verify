# v5 slice 1 — Duplicate / near-duplicate detection (eval-only)

**Date:** 2026-06-20
**Status:** approved design, pre-implementation
**Arc:** v5 (Fraud, anomaly & the security layer) — first of six sub-projects

## Context

v3 (accuracy) and v4 (calibration/uncertainty) are shipped. v5 expands trust from
"accurate" to "safe": catch the *malicious* invoice, with explainable reasons. v5 is
six subsystems (duplicate detection, vendor-master/BEC, anomaly detection, collusion
NLP, XAI, synthetic-fraud benchmark); each gets its own spec → plan → build cycle.

This slice is the first: **duplicate / near-duplicate detection** — ~45% of AP fraud,
the largest single share. It is pure rule + fuzzy logic, so it lives in the clean
domain layer with **no new ML dependencies**, it is naturally explainable, and its
synthetic injection is trivial (resend/edit an invoice). It establishes the fraud
subsystem and the synthetic-fraud benchmark spine that every later v5 detector plugs
into.

## Scope

**Eval-only.** Deliver a pure domain matcher, an `InvoiceLedger` application port (the
seam for production), a synthetic-fraud benchmark, XAI reasons, and a CLI. **Deferred
to a follow-up:** wiring a fraud stage into the approval pipeline and a concrete ledger
adapter (JSON/SQLite of seen invoices).

This mirrors how every prior capability landed: build pure logic, measure it with an
eval harness, integrate into the live pipeline afterward.

## Approach (chosen: hybrid)

The matcher emits **both** a discrete tier (the XAI reason) **and** a continuous
similarity score (the benchmark curve + operating point). This is the only option that
satisfies both project non-negotiables at once: an explainable reason per flag, and a
measurable catch-rate-vs-false-positive curve with a chosen operating point (the same
shape as v4's risk-coverage). Rejected: tiered-only (no curve) and weighted-score-only
(magic threshold, reconstructed reasons).

## Architecture & files

Pure domain matcher + eval harness; zero new dependencies. The dependency rule holds —
domain knows nothing of eval/infrastructure.

| File | Change | Purpose |
|------|--------|---------|
| `src/apverify/domain/fraud.py` | NEW | matcher, tiers, `DuplicateMatch`, `IdentifiedInvoice` (pure) |
| `src/apverify/domain/ocr.py` | EDIT | expose `fold_confusables()` (refactor private `_normalise` to reuse it) |
| `src/apverify/application/ports.py` | EDIT | add `InvoiceLedger` port (`known_invoices()`) — seam for later wiring |
| `src/apverify/eval/fraud_synthesis.py` | NEW | inject duplicates + hard negatives → labeled `FraudCase`s |
| `src/apverify/eval/fraud_eval.py` | NEW | run matcher over cases → `FraudReport` metrics |
| `src/apverify/eval/fraud_cli.py` | NEW | `apverify-eval-fraud --dataset synthetic\|docile` |
| `src/apverify/eval/report.py` | EDIT | `render_fraud` |
| `pyproject.toml` | EDIT | `apverify-eval-fraud` console script |
| `tests/unit/domain/test_fraud.py` | NEW | matcher tiers/score/reason |
| `tests/unit/eval/test_fraud_synthesis.py` | NEW | injection labels |
| `tests/unit/eval/test_fraud_eval.py` | NEW | metric math |

## Domain matcher (tiers + score + XAI)

`compare_invoices(candidate: Invoice, prior: IdentifiedInvoice) -> DuplicateMatch`.

`DuplicateMatch`: `tier: DuplicateTier`, `score: float ∈ [0,1]`, `reason: str`,
`matched_id: str`.

`DuplicateTier`: `EXACT_RESEND`, `OCR_VARIANT`, `NEAR_DUPLICATE`, `DISTINCT`.

Per-field similarity over **total amount, invoice date, vendor name (fuzzy),
invoice number (confusable-folded)**, reusing `domain/ocr` confusable folding and the
`eval/accuracy` amount/date/fuzzy comparators (no duplicated logic).

Tier logic (ordered):
1. all key fields raw-equal → `EXACT_RESEND`
2. fields equal but invoice-no differs *only* by confusable characters (O↔0, 1↔l/|,
   etc.) → `OCR_VARIANT`
3. vendor + amount equal, date within the proximity window, invoice-no edited →
   `NEAR_DUPLICATE`
4. otherwise → `DISTINCT`

The date proximity window is a named constant (default ±3 days), exposed as a
parameter on the matcher so the benchmark can sweep it; not a magic literal.

**Score composition:** `score` is a weighted mean of the four per-field similarities,
each in [0,1] — invoice-no (confusable-folded equality/edit-ratio), total amount
(1.0 if equal, else proximity ratio), date (1.0 if equal, decaying to 0 across the
window), vendor (fuzzy ratio). Weights favour the strong duplicate signals
(invoice-no, amount) over the weak ones (vendor, date). The tier is derived from the
field-equality *structure*; the score gives the continuous magnitude the benchmark
curve needs. Both are returned; they are consistent by construction (EXACT_RESEND
scores 1.0, DISTINCT scores low).

`find_duplicates(candidate, priors: Sequence[IdentifiedInvoice]) -> list[DuplicateMatch]`
compares against each prior and returns non-`DISTINCT` matches sorted by score
descending. Empty priors → empty list.

`IdentifiedInvoice`: a frozen `(identifier: str, invoice: Invoice)` value object, so a
prior carries a ledger id distinct from its (shared) invoice number.

**XAI:** the reason is templated per tier and cites the actual matching fields, e.g.
*"same vendor + amount ₹184,200 + date; invoice-no INV-1001↔INV-l00l differs only by
OCR-confusable characters."*

## Synthetic fraud injection + hard negatives

`build_fraud_cases(base: Sequence[GroundTruth]) -> list[FraudCase]` where
`FraudCase = (candidate: Invoice, priors: tuple[IdentifiedInvoice, ...], is_fraud: bool,
kind: str)`. Deterministic (index-derived), reusing `eval/synthetic.generate_dataset`.

True duplicates (`is_fraud=True`):
- `EXACT_RESEND` — verbatim copy of a prior
- `OCR_VARIANT` — invoice-no with confusable swaps + comma in amount
- `SMALL_EDIT` — amount ±rounding or date ±1 day, otherwise identical
- `MULTI_CHANNEL_RESEND` — identical re-submission (models a second channel)

Hard negatives (`is_fraud=False`) — these make the false-positive number meaningful:
- `LEGIT_RECURRING` — same vendor + same amount, **new** invoice-no + **new** date (a
  monthly retainer); a naive "same vendor+amount" matcher wrongly flags this, so the
  matcher distinguishing it is the headline result
- `LEGIT_DISTINCT` — an ordinary unrelated invoice

## Benchmark metrics + DocILE realism check

`evaluate_fraud(cases, threshold) -> FraudReport`: for each case run `find_duplicates`;
a candidate is **flagged** if its best match score ≥ threshold. Reports:
- **catch-rate** (recall on true duplicates)
- **false-positive-rate** (legit cases flagged)
- **precision**
- **AUROC** of the score separating fraud from legit (reuse `fusion.auroc`)
- a **threshold sweep** (catch-rate vs FP at each threshold) with the zero-FP /
  max-catch operating point (mirrors v4 risk-coverage)
- a **per-kind breakdown** (which duplicate types are caught)

`render_fraud` prints these plus example reasons.

DocILE realism check: build `Invoice` objects from DocILE **ground-truth fields**
(via the existing `load_docile_labelled`) — **no model calls, so quota-free** — inject
the same duplicate types, and report the same metrics as a realism sanity-check
alongside the synthetic headline.

CLI: `apverify-eval-fraud --dataset synthetic|docile [--count/--limit] [--threshold]`
(`--threshold` defaults to the zero-FP operating point found by the sweep, so the
default run reports the safe operating point without manual tuning).

## Data flow

```
base invoices (synthetic generator | DocILE ground-truth fields)
  → fraud_synthesis.build_fraud_cases → [FraudCase(candidate, priors, is_fraud, kind)]
  → fraud_eval: for each case, domain.find_duplicates(candidate, priors)
  → collect (best_score, is_fraud, tier, kind)
  → FraudReport (catch-rate, FP-rate, precision, AUROC, sweep, per-kind)
  → report.render_fraud
```

## Error handling

All matcher and synthesis functions are pure and deterministic (no randomness — values
derived from index, per the existing synthetic generator). Empty priors → `DISTINCT` /
empty result. Amounts/dates are already validated `Money`/`Decimal` on the `Invoice`
entity, so no parsing failures in the matcher. The only I/O is the CLI's DocILE read,
which is gated (needs the downloaded dataset) exactly like the other dataset CLIs.

## Testing (TDD) + acceptance bar

Every unit pure and deterministic.

Domain tests: each tier (exact, OCR-variant, near-dup, distinct), score monotonicity,
reason content, `LEGIT_RECURRING → DISTINCT`, `find_duplicates` best-match ordering,
confusable folding.

Eval tests: injection produces correct `is_fraud`/`kind` labels, hard negatives are
`is_fraud=False`, scorer catch/FP/precision math on a hand-built set, AUROC reuse.

Acceptance:
- synthetic `EXACT_RESEND` + `OCR_VARIANT` caught 100% @ 0% false-positive
- `LEGIT_RECURRING` never flagged (0 false-positive)
- score AUROC ≥ 0.9 on synthetic
- DocILE check reported honestly (realism, not a target)
- gates: `ruff check`, `ruff format --check`, `mypy --strict src tests`, `pytest`;
  **domain layer 100% coverage**

## Out of scope (this slice)

Approval-pipeline fraud stage; concrete ledger adapter; the other five v5 subsystems
(vendor-master/BEC, anomaly detection, collusion NLP, and the broader XAI/SHAP and
cross-detector benchmark, which arrive with their own slices).
