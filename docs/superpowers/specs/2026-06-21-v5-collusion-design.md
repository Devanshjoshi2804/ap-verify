# v5 slice 6 — Collusion detection (behavioral, eval-only)

**Date:** 2026-06-21
**Status:** approved design, pre-implementation
**Arc:** v5 (Fraud, anomaly & the security layer) — sixth and final sub-project

## Context

v5 shipped duplicate, BEC, anomaly detection, a cross-detector benchmark, and the XAI
layer. The last roadmap item is collusion. The roadmap phrases it as "semantic +
behavioral NLP", but collusion is fundamentally about *who approves what*, not invoice
text — text rarely reveals it and there is no labelled text data. So this slice scopes
collusion as **behavioral detection over approval records**, which is the grounded,
measurable signal. NLP-on-invoice-text is explicitly dropped as not-the-real-signal.

Collusion is a cross-record, batch pattern (an approver's behaviour across many
invoices), so it does not fit the per-invoice `ReviewPayableUseCase`. It is a standalone
analysis tool over an approval log, like the other eval detectors.

## Scope

**Eval-only / standalone analysis.** A pure domain detector over a log of approval
records, plus a synthetic benchmark and CLI. No per-invoice pipeline wiring (collusion
is cross-record by nature). No new dependencies.

## Approach (chosen: behavioral over approval records)

Detect collusion from approval behaviour, not text: an approver funneling one vendor,
approvals parked just under the approver's authorization limit, and rubber-stamping
(approved within seconds of submission). Rejected: literal NLP on invoice text (weak,
speculative, no data) and a behavioral+text hybrid (the text leg is the weak part and
only adds scope).

## Architecture & files

| File | Change | Purpose |
|------|--------|---------|
| `src/apverify/domain/collusion.py` | NEW | `ApprovalRecord`, `CollusionSeverity`, `CollusionSignal`, `detect_collusion` (pure) |
| `src/apverify/eval/collusion_synthesis.py` | NEW | `build_collusion_log` (colluding + normal approver-vendor pairs) |
| `src/apverify/eval/collusion_eval.py` | NEW | `evaluate_collusion` → `CollusionReport` |
| `src/apverify/eval/collusion_cli.py` | NEW | `apverify-eval-collusion` |
| `src/apverify/eval/report.py` | EDIT | `render_collusion` |
| `pyproject.toml` | EDIT | console script |
| tests | NEW | per unit |

## Domain detector

`ApprovalRecord(submitter: str, approver: str, vendor: str, amount: Money,
submitted_at: datetime, approved_at: datetime, approver_limit: Money)` (frozen). Pure
data; datetimes are passed in (no wall-clock in the domain).

`CollusionSeverity`: `NONE`, `MEDIUM`, `HIGH`.

`CollusionSignal(approver: str, vendor: str, concentration: float, under_limit_rate:
float, rubber_stamp_rate: float, score: float, severity: CollusionSeverity, reason:
str)` (frozen).

`detect_collusion(records: Sequence[ApprovalRecord], *, concentration_floor: float =
0.8, under_limit_band: float = 0.05, rubber_stamp_seconds: float = 60.0, flag_score:
float = 0.6) -> list[CollusionSignal]`. Groups records by `(approver, vendor)`; for each
pair computes:
- **concentration** — `count(pair) / count(approvals by that approver)`: how much of an
  approver's activity funnels to this one vendor
- **under_limit_rate** — fraction of the pair's approvals with `amount` within
  `under_limit_band` *below* the approver's limit (gaming the ceiling)
- **rubber_stamp_rate** — fraction with `(approved_at - submitted_at).total_seconds() <=
  rubber_stamp_seconds` (approved without real review)

`score` is the mean of the three rates (each in [0,1]); `severity` is HIGH at
`score >= 0.8`, MEDIUM at `>= flag_score`, else NONE. The detector returns the pairs
scoring `>= flag_score`, ranked by score descending, each with a `reason` naming the
dominant pattern, e.g. *"approver alice cleared 95% of vendor Acme's invoices, 90% just
under alice's 50000 limit, 100% approved within 60s"*.

A pair needs a minimum number of approvals (default 3, the same abstention philosophy as
anomaly) before it can be flagged, so a one-off approval is never collusion.

## Benchmark

`build_collusion_log(pairs: int = 6, per_pair: int = 8) -> tuple[list[ApprovalRecord],
dict[tuple[str, str], bool]]`. Synthesizes, deterministically, an approval log plus a
truth map labeling each `(approver, vendor)` pair colluding or not:
- **colluding** pairs — one approver funnels one vendor: every approval just under the
  approver's limit, approved within seconds, no other vendors for that approver
- **normal** pairs — an approver handles several vendors, amounts spread across the
  range, realistic approval latency (hours)

`evaluate_collusion(records, truth) -> CollusionReport`: runs `detect_collusion`, and
reports **catch-rate** (colluding pairs flagged), **false-positive-rate** (normal pairs
flagged), **precision**, and **AUROC** of the per-pair score separating colluding from
normal (reusing `fusion.auroc`). `CollusionReport(pair_count, colluding_count,
catch_rate, false_positive_rate, precision, auroc)` (frozen). `render_collusion` prints
the headline + a ranked signal table; `apverify-eval-collusion --pairs N --per-pair M`.

## Data flow

```
synthetic approval log + truth map
  → detect_collusion(records) → ranked [CollusionSignal] for flagged pairs
  → evaluate_collusion compares flagged pairs to truth → CollusionReport
  → report.render_collusion
```

## Error handling

Pure and deterministic; synthetic datetimes derived from index (no wall-clock). A pair
below the minimum approval count abstains (never flagged). Empty log → empty result /
zeroed report. A zero approver-total cannot occur for a pair that exists (it has ≥1
record). No I/O beyond the CLI.

## Testing (TDD) + acceptance bar

Domain TDD: each signal in isolation (a concentrated/under-limit/rubber-stamped pair
flags; a diverse, varied-amount, normal-latency pair does not), the minimum-approvals
abstention, ranking, reason content, the datetime-latency computation. Eval: synthesis
labels, per-pair metric math, AUROC reuse.

Acceptance:
- colluding pairs caught ≥ 90% at 0% false-positive on normal pairs
- per-pair score AUROC ≥ 0.9 on synthetic
- a single-approval pair is never flagged
- gates: `ruff check`, `ruff format --check`, `mypy --strict src tests`, `pytest`;
  **domain layer 100% coverage**

## Out of scope (this slice)

NLP / text analysis of invoices; per-invoice pipeline wiring (collusion is cross-record);
graph-based ring detection beyond pairwise approver-vendor patterns; real approval-log
data (synthetic only — no public labelled collusion dataset exists). This slice
completes v5's six sub-projects.
