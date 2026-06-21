# v5 slice 2 — Vendor-master / bank-change / BEC detection (eval + pipeline wiring)

**Date:** 2026-06-20
**Status:** approved design, pre-implementation
**Arc:** v5 (Fraud, anomaly & the security layer) — second of six sub-projects

## Context

v5 slice 1 (duplicate detection) is shipped: a pure `domain/fraud.py` matcher, a
synthetic-fraud benchmark, and the `apverify-eval-fraud` CLI. This slice adds the
**single highest-loss** fraud vector: vendor-master / bank-change / business-email-
compromise (BEC). Over 60% of organisations see BEC attempts; the classic attack
changes a known vendor's bank account at the last minute, or impersonates a vendor
with a typo-squatted name, so payment is redirected to the attacker.

`Invoice.bank_account: str | None` already exists and is populated from the extractor
DTO, so the data this slice needs is already in the pipeline.

## Scope

**Eval + pipeline wiring.** Beyond the pure check and benchmark, this slice wires a
vendor-risk stage into the live approval pipeline: a `VendorMasterRepository` port, a
concrete `JsonVendorMaster` adapter, decision reconciliation (HIGH severity → HOLD),
and the BEC reason recorded in the audit trace. `apverify review` runs the check live.

## Approach (chosen: rule cascade + scored impersonation)

The check is mostly categorical (a bank account matches a known one or it does not),
but impersonation is continuous (how close a typo-squat is to a real vendor name). So
the cascade gives the discrete kind + severity (the XAI reason), and the name-similarity
to the nearest known vendor is a continuous score that drives the impersonation/new-
payee boundary and gives the benchmark a sweepable threshold + AUROC. Same hybrid shape
as slice 1, fitted to this signal. Rejected: pure-rule (no sweepable impersonation
threshold) and fully-scored (bank-change is binary; the reason would be reconstructed
rather than read off the tier).

## Architecture & files

| File | Change | Purpose |
|------|--------|---------|
| `src/apverify/domain/vendor_master.py` | NEW | `KnownVendor`, `VendorRiskKind`, `Severity`, `VendorRiskAssessment`, `assess_vendor_risk` (pure) |
| `src/apverify/domain/approval.py` | EDIT | `reconcile_with_vendor_risk` (HIGH→HOLD, LOW→reason only) |
| `src/apverify/application/ports.py` | EDIT | `VendorMasterRepository` port (`known_vendors()`) |
| `src/apverify/application/review_payable.py` | EDIT | optional vendor-master step → reconcile; `PayableReview.vendor_risk` |
| `src/apverify/infrastructure/vendor_master/repository.py` | NEW | `JsonVendorMaster` adapter (JSON file of known vendors) |
| `src/apverify/infrastructure/settings.py` | EDIT | `vendor_master_path` |
| `src/apverify/interface/cli/bootstrap.py` | EDIT | wire vendor master when configured |
| `src/apverify/eval/bec_synthesis.py` | NEW | `build_bec_cases` (inject bank-swap / typo-squat / new-payee + hard negatives) |
| `src/apverify/eval/bec_eval.py` | NEW | `evaluate_bec` → `BecReport` (per-signal catch/FP, impersonation AUROC) |
| `src/apverify/eval/bec_cli.py` | NEW | `apverify-eval-bec` |
| `src/apverify/eval/report.py` | EDIT | `render_bec` |
| `pyproject.toml` | EDIT | `apverify-eval-bec` console script |
| tests | NEW | per unit (domain, approval, use case, infra, eval, cli) |

## Domain vendor-master check

`assess_vendor_risk(invoice: Invoice, master: Sequence[KnownVendor],
impersonation_threshold: float = 0.85) -> VendorRiskAssessment`.

`KnownVendor`: `name: str`, `bank_accounts: frozenset[str]`, `gstin: str = ""` (frozen).

`VendorRiskKind`: `CLEAN`, `NEW_PAYEE`, `BANK_CHANGE`, `IMPERSONATION`.

`Severity`: `NONE`, `LOW`, `HIGH`.

`VendorRiskAssessment`: `kind: VendorRiskKind`, `severity: Severity`, `score: float`
(name similarity to the nearest known vendor), `matched_vendor: str`, `reason: str`
(frozen).

Logic — find the nearest known vendor by confusable-folded fuzzy name match (reuse
`domain/ocr` folding + `difflib.SequenceMatcher`), then:
1. **Exact name match** (canonical-equal) → known vendor. If `invoice.bank_account` is
   set and not among that vendor's `bank_accounts` → `BANK_CHANGE` (HIGH). Otherwise
   (bank known, or invoice carries no bank account) → `CLEAN` (NONE).
2. **Near match** — best similarity ≥ `impersonation_threshold` but not exact →
   `IMPERSONATION` (HIGH): a name that looks like a known vendor but is not it.
3. **No close match** — best similarity < threshold → `NEW_PAYEE` (LOW).
4. **Empty master** → `NEW_PAYEE` (LOW), score 0.0.

`score` is the best name similarity in all cases. Severity is a fixed function of kind
(CLEAN→NONE, NEW_PAYEE→LOW, BANK_CHANGE→HIGH, IMPERSONATION→HIGH). The reason cites
specifics, e.g. *"bank account on known vendor ACME Steel changed: '****4321' not among
its known accounts"* or *"vendor 'ACME Stee1' is a 0.94 name-match to known 'ACME
Steel' but not identical — possible impersonation"*.

## Pipeline wiring

`VendorMasterRepository` (runtime-checkable Protocol): `known_vendors() -> tuple[
KnownVendor, ...]`.

`ReviewPayableUseCase` gains an optional `vendor_master: VendorMasterRepository | None`
dependency. A new `_vendor_risk` step (same shape as `_audit`/`_consistency`) calls
`assess_vendor_risk` over the invoice and master, records the outcome to the trace, and
returns the assessment; when present, `reconcile_with_vendor_risk` folds it into the
decision. `PayableReview` gains `vendor_risk: VendorRiskAssessment | None`.

`reconcile_with_vendor_risk(decision, assessment, policy)` in `domain/approval.py`:
HIGH severity → escalate to HOLD (a redirected payment is the worst case); LOW
(new-payee) → append the reason but **do not change the decision** (new vendors are
common; surfaced for the human, not blocked); NONE → unchanged. Never lowers an
existing decision, consistent with the other reconcilers.

`JsonVendorMaster` adapter reads a JSON file — a list of `{name, bank_accounts, gstin}`
— mirroring the existing procurement/receivables JSON loaders. Wired in bootstrap
behind a `vendor_master_path` setting; absent → the step is skipped (no behaviour
change to the existing pipeline).

## BEC benchmark (synthetic only)

DocILE ground-truth labels carry no bank-account data, so this benchmark is synthetic
(unlike slice 1's dual synthetic+DocILE). `build_bec_cases(base) -> list[BecCase]`
where `BecCase = (invoice, master, scenario)` and `scenario` is the injected label
below; the expected severity per scenario is fixed and known to the evaluator. It
assigns each synthetic vendor a known bank account to form the master, then emits:
- `bank_change` (HIGH) — known vendor, swapped bank account
- `impersonation` (HIGH) — confusable/edited vendor name + an attacker bank account
- `new_payee` (LOW) — a brand-new vendor not in the master
- hard negative `known_clean` (NONE) — known vendor + a known bank account (must not
  flag HIGH)
- hard negative `legit_new` (LOW) — a genuinely unrelated new vendor (LOW only, never
  mis-escalated to HIGH impersonation)

`evaluate_bec(cases, threshold) -> BecReport` reports per-signal catch-rate (a scenario
assessed at its expected severity), false-positive rate (a `known_clean` case flagged
HIGH), precision on HIGH flags, and the impersonation-similarity AUROC — computed over
the `impersonation` (positive) and `legit_new` (negative) cases only, on the name-
similarity `score`, since that is exactly the boundary the threshold controls.
`render_bec` prints the per-signal table; `apverify-eval-bec --count N [--threshold]`.

## Data flow

```
synthetic base invoices
  → bec_synthesis.build_bec_cases → [BecCase(invoice, master, expected_kind, ...)]
  → bec_eval: assess_vendor_risk(invoice, master) per case
  → collect (kind, severity, score, expected) → BecReport (per-signal catch/FP, AUROC)
  → report.render_bec

live: review_payable._vendor_risk → assess_vendor_risk(invoice, master.known_vendors())
  → reconcile_with_vendor_risk → FinalDecision (HOLD on HIGH) → audit trace
```

## Error handling

The domain check is pure and total: empty master → NEW_PAYEE; missing bank account →
no bank-change assessment (CLEAN if the vendor is known). The `JsonVendorMaster` adapter
raises a clear error on a missing/malformed file at construction; an absent
`vendor_master_path` simply omits the step. The pipeline step mirrors the existing
optional stages — its absence changes nothing.

## Testing (TDD) + acceptance bar

Pure domain TDD: each kind (CLEAN, NEW_PAYEE, BANK_CHANGE, IMPERSONATION), severity
mapping, the exact-vs-near name boundary at the threshold, bank-change only when a bank
account is present and unknown, empty-master path, reason content. Approval:
`reconcile_with_vendor_risk` escalates HIGH→HOLD, leaves LOW as a reason only, never
lowers. Use case: a bank-change invoice drives the pipeline to HOLD with a BEC reason; a
clean known vendor is unaffected; a new payee adds a reason without changing the
decision. Infra: `JsonVendorMaster` parses a fixture and satisfies the port. Eval:
injection labels correct, per-signal metric math, AUROC reuse. CLI: synthetic run exits
0 and prints the BEC table.

Acceptance:
- bank-change + impersonation caught 100% at 0% false-positive on `known_clean`
- new-payee flagged LOW only — never escalated to HOLD
- impersonation-similarity AUROC ≥ 0.9
- pipeline HOLD on a bank-change invoice verified end-to-end
- gates: `ruff check`, `ruff format --check`, `mypy --strict src tests`, `pytest`;
  **domain layer 100% coverage**

## Out of scope (this slice)

Anomaly detection (Isolation Forest / autoencoder), collusion NLP, the broader
XAI/SHAP layer, and a cross-detector fraud benchmark — each its own later v5 slice. A
DocILE-based impersonation realism check is deferred (DocILE lacks bank data; vendor
names alone would only exercise impersonation, not bank-change).
