# Error taxonomy

When the field-accuracy harness (`apverify-eval-accuracy`) marks an extraction
"wrong", the miss falls into one of three buckets. Only the first is worth
optimising; conflating them is how teams chase phantom accuracy gains. Every
number in the README's benchmark is reported after this triage.

| Bucket | What it is | What to do |
|---|---|---|
| **model-error** | The extractor genuinely read the field wrong. | Fix it (prompt, ensemble, layout model). |
| **ground-truth-noise** | The label is messy/ambiguous; the extraction is arguably right. | Don't chase; note it, optionally clean the eval. |
| **metric-unfairness** | The comparison was unfair — two correct representations scored as different. | Fix the *metric*, not the model. |

## Document-failure attribution — picking the lever

Per-field F1 hides the metric that matters for automation: how often a *whole
document* is error-free. Per-field accuracy compounds — at ~0.9 per field over
~6 fields, ≈ half of documents carry at least one wrong field. So before
optimising anything, we attribute *which field is responsible for the most
failed documents* (`score_document` over the DocILE `val` split, local Ollama
model, n=50):

| Field | Documents it broke (baseline) | Share of failures |
|---|---|---|
| line_items | 22 | 69% |
| vendor | 18 | 56% |
| date | 10 | 31% |
| total | 9 | 28% |
| currency | 5 | 16% |
| subtotal | 1 | 3% |

Baseline: **32/50 documents (64%) fail**. The two levers are **line items** and
**vendor** — and failures overlap, so fixing one field rarely clears a whole
document. This attribution, not the lowest per-field F1, is what selects the work.

## Real examples (from DocILE `val` + CORD)

### metric-unfairness
- **Currency `$` vs `USD`.** DocILE labels currency as the symbol `$`; the
  extractor correctly returns the ISO code `USD`. The exact comparison scored
  these as different → currency F1 **0.00**. Fixed by normalising symbol↔code
  (`_currency_code` in `eval/accuracy.py`) → currency F1 **1.00**. The canonical
  example: a metric bug was hiding a perfect extractor.
- **Vendor legal suffix / superset.** `PHILIP MORRIS` vs
  `PHILIP MORRIS INCORPORATED`; `KGMB` vs `KGMB TV`; most starkly
  `DISPLAY TECHNOLOGIES LLC` vs the prediction `DISPLAY TECHNOLOGIES LLC CUSTOM
  GROUP` — a *superset* of the truth that missed the 0.80 similarity cut by 0.01.
  All correct reads scored as misses. Fixed with a guarded containment +
  suffix-stripping check (`_vendor_contained` / `_vendor_tokens`): a match counts
  when one name's significant tokens are contained in the other after dropping
  entity/station suffixes (Inc, LLC, Ltd, TV, FM, …), guarded by a minimum token
  length so a stray initialism can't be read as a match. Vendor breaks **18 → 11
  documents**.
- **Amount formatting.** `1.190,00` (EU) vs `1190.00` — handled by
  `parse_amount`'s separator heuristic before comparison.

### ground-truth-noise
- **Vendor with boilerplate.** DocILE labels a vendor as
  `"REMIT TO Sinclair Broadcast\nc/o WSMH"` — a multi-line string with a
  `REMIT TO` prefix. The clean payee is the *more* correct answer; the matcher
  strips the prefix before comparing.
- **Degenerate line tables.** Truth lines with amount `-` (doc 32) or empty
  descriptions (doc 39); the model correctly emits only the real rows but is
  scored as "missing" the placeholders.

### model-error (the real targets)
- **Seller-vs-buyer confusion** — the dominant *genuine* vendor error. On media
  and advertising invoices the model returned the **buyer** (the advertiser /
  customer the bill is addressed to) instead of the **seller / payee**: `WTTG` →
  `American Cancer Society`, `EXPERIAN` → `LORILLARD TOBACCO COMPANY`. Fixed by a
  shared prompt instruction (`VENDOR_GUIDANCE`, embedded in all four providers)
  pinning `vendor_name` to the party that issued the invoice and is owed payment.
  Vendor breaks **11 → 7 documents**; document failure **62% → 56%**.
- **Line-item amount misreads** — the residual binding constraint. The model
  picks the wrong column or scales a row (doc 50: a `6,300.00` line read as
  `496,800.00`). Prompting does not fix this; it needs layout-aware / table
  extraction (tracked as v6). Line items remain ~68% of document failures.

### robustness — silent loss, worse than a flagged miss
Not an accuracy bucket but a reliability one: a model emitting `quantity=""` or
`"87.3"` (a measure where a count belongs), or the literal string `"null"` for a
vendor, used to fail the *entire* extraction or persist junk. Three DocILE pages
were being dropped silently this way. Coercion at the wire boundary
(`_quantity_or_one`, `_blank_if_nullish` in `infrastructure/mapping.py`) turns
these into a scored document with a flagged field — a dropped invoice is worse
than a flagged one.

## Why this matters

Reporting accuracy without this triage overstates the problem and sends you
optimising the wrong thing. Two safe changes — a fairer scorer and a one-line
seller-vs-buyer prompt — cut vendor failures **~60% (18 → 7 documents)** and
document failure **64% → 56%**, while line items stayed flat: proof that
attributing the failure first is what makes the optimisation real. (n=50, local
model, stochastic run-to-run; only the vendor delta is attributed — the rest is
re-run variance.) We keep the buckets explicit and the messy examples in the
open; that honesty is the signal.
