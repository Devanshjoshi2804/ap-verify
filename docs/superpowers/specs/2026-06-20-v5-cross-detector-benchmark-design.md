# v5 slice 4 — Cross-detector fraud benchmark (eval-only)

**Date:** 2026-06-20
**Status:** approved design, pre-implementation
**Arc:** v5 (Fraud, anomaly & the security layer) — fourth sub-project; the capstone benchmark

## Context

v5 slices 1–3 shipped three fraud detectors, each with its own benchmark: duplicate /
near-duplicate (`domain/fraud.py`), vendor-master / bank-change / BEC
(`domain/vendor_master.py`), and statistical anomaly (`domain/anomaly.py`). The v5
roadmap's headline deliverable is "a fraud/anomaly module with explainable flags + a
fraud-catch-rate vs false-positive benchmark on synthesized fraud." This slice is that
capstone: one synthesized stream exercising **all** fraud types, scored by **all** three
detectors together, reporting the combined system number.

## Scope

**Eval-only.** Pure integration of the three shipped detectors into one benchmark. No
new domain logic, ports, pipeline wiring, or dependencies. (Each detector is already
wired into the live pipeline by its own slice.)

## Approach (chosen: unified dataset, all detectors per case)

One synthesized stream where every case carries full context (prior-invoice ledger +
vendor master + per-vendor history) and a label (a specific fraud type, or clean). Run
all three detectors on every case; a case is flagged if any detector fires. This yields
the true combined fraud-catch-rate vs false-positive **and** surfaces cross-talk — a
detector wrongly firing on another detector's fraud, or on a clean invoice. Rejected:
aggregating the three existing per-detector benchmarks, which never runs a detector on
another's cases and so cannot measure cross-talk or a system-wide false-positive rate.

## Architecture & files

| File | Change | Purpose |
|------|--------|---------|
| `src/apverify/eval/fraud_suite_synthesis.py` | NEW | `SuiteCase` + `build_fraud_suite` (one stream, full context per case) |
| `src/apverify/eval/fraud_suite_eval.py` | NEW | `evaluate_fraud_suite` → `FraudSuiteReport` (all three detectors per case) |
| `src/apverify/eval/fraud_suite_cli.py` | NEW | `apverify-eval-fraud-suite` |
| `src/apverify/eval/report.py` | EDIT | `render_fraud_suite` |
| `pyproject.toml` | EDIT | console script |
| tests | NEW | synthesis, eval, render, cli |

## Unified dataset

`SuiteCase(invoice: Invoice, priors: tuple[IdentifiedInvoice, ...],
master: tuple[KnownVendor, ...], history: tuple[Invoice, ...], label: str,
is_fraud: bool)` (frozen).

`build_fraud_suite(base: Sequence[GroundTruth]) -> list[SuiteCase]` builds shared
context once:
- **ledger** (`priors`): the base originals as `IdentifiedInvoice`s (already-posted
  invoices)
- **master**: one `KnownVendor` per base vendor, each with a known bank account
- **history**: per-vendor prior invoices clustered around the vendor's median

Then, for each base invoice, it derives candidates — every one carrying the same three
context objects:
- `clean` (negative) — known vendor, known bank, in-range amount, a **new** invoice
  number and date (not a duplicate, not a bank change, not a spike → no detector should
  fire)
- `dup_resend` — a verbatim copy of a ledger original (duplicate detector fires)
- `dup_ocr_variant` — a ledger original with OCR-confused invoice-number digits
- `bank_change` — known vendor, in-range amount, new invoice-no, **attacker** bank
- `impersonation` — typo-squatted vendor name + attacker bank
- `amount_spike` — known vendor + known bank, new invoice-no, amount many× the median
- `threshold_gaming` — known vendor + known bank, new invoice-no, amount just under a
  round limit

`LABELS` and `FRAUD_LABELS` constants enumerate these; `is_fraud = label in
FRAUD_LABELS`.

The construction deliberately isolates each fraud to its own detector (and keeps the
others quiet); the benchmark then *measures* whether that isolation holds.

## Combined evaluator + report

`evaluate_fraud_suite(cases: Sequence[SuiteCase]) -> FraudSuiteReport`. For each case,
run all three detectors and record which fired:
- **duplicate** — `find_duplicates(invoice, priors)` non-empty (any non-DISTINCT match)
- **bec** — `assess_vendor_risk(invoice, master).severity is Severity.HIGH`
- **anomaly** — `RobustAnomalyDetector().score(invoice, history).severity` in
  {HIGH, MEDIUM}

`flagged = any(fired)`; `attribution` = the set of detector names that fired (matching
the live pipeline's thresholds, so the benchmark reflects production behaviour).

`FraudSuiteReport` (frozen): `case_count`, `fraud_count`, `catch_rate` (frauds flagged
by ≥1 detector / frauds), `false_positive_rate` (cleans flagged by any detector /
cleans — the cross-talk number), `precision` (frauds among all flagged / all flagged),
`per_label: dict[str, float]` (flag rate per label), `per_detector: dict[str, int]`
(frauds each detector caught). `render_fraud_suite` prints the headline line plus a
per-label catch table and a per-detector contribution table.

CLI: `apverify-eval-fraud-suite --count N`.

## Data flow

```
synthetic base invoices
  → build_fraud_suite → [SuiteCase(invoice, priors, master, history, label, is_fraud)]
  → evaluate_fraud_suite: for each case run duplicate + BEC + anomaly detectors
  → flagged = any fired; attribution = which fired
  → FraudSuiteReport (combined catch / FP / precision, per-label, per-detector)
  → report.render_fraud_suite
```

## Error handling

All functions pure and deterministic (index-derived synthesis, no randomness). Empty
cases → a zeroed report. The detectors are already total (abstain on insufficient
context). No I/O beyond the CLI.

## Testing (TDD) + acceptance bar

Unit TDD: `build_fraud_suite` produces one case per label with all three context objects
populated and correct `is_fraud`; the evaluator's union-flag + attribution math on a
hand-built set; a `clean` case flagged by no detector; per-label and per-detector
aggregation. 

Acceptance:
- combined catch-rate 100% — every fraud type caught by ≥1 detector
- false-positive-rate 0% — no clean case flagged by any detector (no cross-talk)
- each fraud type attributed to its own detector (duplicate→dup_*, bec→bank_change/
  impersonation, anomaly→amount_spike/threshold_gaming)
- gates: `ruff check`, `ruff format --check`, `mypy --strict src tests`, `pytest`
  (no domain changes; the suite logic is fully covered)

## Out of scope (this slice)

Collusion NLP; the broader XAI/SHAP explanation layer; an autoencoder/temporal anomaly
model; a real-data (DocILE) cross-detector run (the BEC and anomaly legs need
bank/history data DocILE lacks). Each is a later v5/v6 item.
