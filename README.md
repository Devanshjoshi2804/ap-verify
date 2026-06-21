# ap-verify

**An accounts-payable agent that knows when it's wrong.**

Anyone can wire an LLM to read an invoice. The hard, valuable problem in finance
AI is *trust*: an agent that approves payments must never hallucinate a total, a
vendor, or a bank account, and must never pay for goods that were not ordered or
received. `ap-verify` treats *"knowing when the extraction is wrong"* as the core
engineering problem — a vision model extracts the invoice, then two independent
verifiers (a **critic** and a **3-way matcher**) decide whether it is safe to pay.

## Results at a glance

**The finding that matters:** strong *per-field* extraction (DocILE macro-F1 0.92) does
**not** translate into safe *per-invoice* automation — across ~6 fields those errors
compound, so roughly half of full-page invoices carry at least one wrong field. This
project's contribution is the **measurement infrastructure that proves that and pinpoints
the exact lever** (vendor at 0.71, line items at 0.66 F1), with calibrated confidence and
an honest per-invoice operating curve — not a black-box "99% accurate" auto-approver.
That is the difference between *measuring* trust and *claiming* it.

And it is actionable: attributing *which field breaks the most whole documents* (not the
lowest per-field F1) selected the work, and two safe changes — a fairer vendor scorer and
a one-line seller-vs-buyer prompt — cut **vendor document failures ~60% (18 → 7)** and
document-level failure **64% → 56%** on the local model, while line items (the harder,
table-extraction frontier) stayed flat. See
[docs/ERROR_TAXONOMY.md](docs/ERROR_TAXONOMY.md).

The numbers worth trusting are the ones measured on **real public invoices**:

| What | Result | Where |
|---|---|---|
| Extraction accuracy (DocILE, real invoices) | per-field macro-F1 **0.92** | `apverify-eval-accuracy` |
| Confidence calibration (DocILE) | ECE **0.12 → 0.03** after temperature scaling | `apverify-eval-calibration` |
| Multi-signal fusion (Gemini×Groq, **5-fold CV, n=511**) | AUROC **0.70** (95% CI 0.60–0.80), ECE **0.03** | `apverify-eval-fusion` |
| Per-invoice auto-approval (real DocILE, n=143) | **no error-free threshold yet** — per-field 0.92 compounds to ~50% document error | `apverify-eval-fusion` |
| Document-level diagnosis + fix (DocILE, local model, n=50) | vendor doc-failures **18 → 7**, document-fail **64% → 56%** | [`ERROR_TAXONOMY.md`](docs/ERROR_TAXONOMY.md) |
| Sampling-based uncertainty | **null result, published** — AUROC ~0.55, no better than chance | `apverify-eval-uncertainty` |
| Critic throughput | **~4,000 invoices/sec** single core (verification layer) | `apverify-bench` |

A note on honesty, because it matters more than any single number: the fusion and
calibration figures carry honest caveats (the fusion AUROC is cross-validated at n=511
with a wide-ish CI, and the per-invoice auto-approval gate has *no error-free threshold
yet* on real full-page invoices — see below), and the **fraud-detection benchmarks below
run on *synthesized* fraud, so
their 100% scores validate the detection *logic*, not real-world catch rates** — there
is no public labelled-fraud dataset, so production catch rates would need data we don't
have. What *is* measured on real data (extraction, calibration, fusion discrimination,
the false-positive behaviour) is reported as-is, imperfections included.

## Benchmark

The critic, measured against an injected-error set (synthetic invoices with perfect
ground truth, each corrupted six ways) — a **logic-validation** test (does the critic
catch the corruptions it is designed to catch?), not a real-world accuracy claim:

| Metric | Value |
|---|---|
| Invoices (clean / corrupted) | 25 / 150 |
| Hallucination-catch rate | 100.0% |
| Safe-auto-approval rate | 100.0% |
| False-hold rate | 0.0% |
| Escaped (corrupt auto-approved) | 0 |

Reproduce with `apverify-eval --count 25`. This measures the **critic's** catch
logic against ground truth — not the vision model's raw field accuracy, which is a
separate, model-dependent number. The honest claim is: *on this injected-error
set, every corruption the critic is designed to catch is caught before approval,
and no clean invoice is wrongly held.*

**Field-level extraction accuracy.** Separately from the critic's decisions,
`apverify-eval-accuracy` runs the live extractor over real invoice images and
scores precision / recall / F1 — **per field and per line item** — against ground
truth. This is the headline extraction metric.

| Dataset | Per-field macro-F1 | Line-item (LIR) F1 |
|---|---|---|
| CORD-v2 (receipts) | **1.00** | **0.91** |
| DocILE (full-page invoices) | **0.92** | **0.66** |

DocILE per-field: currency / subtotal / tax ≈ 1.0, date 0.91, total 0.89, **vendor
0.71** and **line items 0.66** are the honest targets to improve (line-item
extraction is the roadmap's hardest task, and difficulty clearly scales with
document complexity — CORD receipts 0.91 vs DocILE invoices 0.66). The
document-failure attribution that turns these field numbers into "which field to
fix first", and the vendor scorer + seller-vs-buyer prompt fixes that followed, are
written up in [docs/ERROR_TAXONOMY.md](docs/ERROR_TAXONOMY.md).

**Cross-provider leaderboard.** `apverify-eval-leaderboard` runs *every* configured
extractor over the same real invoices and ranks them — the honest provider trade-off,
reproducible by anyone. Measured on DocILE (a provider down or quota-exhausted is
skipped, not faked):

| # | Provider | Per-field macro-F1 | Line-item F1 | Notes |
|---|---|---|---|---|
| 1 | Gemini (gemini-flash-latest) | **0.92** | 0.66 | highest quality; paid |
| 2 | Ollama (qwen2.5vl:7b) | **0.81** | 0.54 | **local, free, no API key** |
| 3 | Mistral (pixtral-12b) | 0.56 | — | burst rate-limited |
| – | Groq (llama-4-scout) | — | — | vision rate-limit invalidated the run |

The finding that matters for an open-source user: **the local model (0.81) is the
strongest non-Gemini option** — good enough to make the keyless path a real default,
and the natural cheap leg of a cost cascade (route easy invoices local, escalate hard
ones to Gemini). Sample sizes vary by provider quota; rerun with
`apverify-eval-leaderboard --dataset docile --dataset-path ./docile`.

Building this harness immediately paid off: it surfaced that DocILE labels currency
as the symbol `$` while the model returns the ISO code `USD` — a *metric* bug
hiding a strong extractor (currency F1 0.00 → 1.00 once fixed). Every "wrong"
extraction is triaged into model-error / ground-truth-noise / metric-unfairness;
see [docs/ERROR_TAXONOMY.md](docs/ERROR_TAXONOMY.md).

**Confidence calibration (v4).** A confidence is only useful if it's *honest*:
invoices rated 0.9 should be right ~90% of the time. `apverify-eval-calibration`
pairs the critic's per-field confidence with whether the field was actually
extracted correctly (against ground truth) and reports **Expected Calibration
Error (ECE)**, a reliability table, and a **risk-coverage curve** with the
operating point. On DocILE the raw critic measures **ECE ≈ 0.12** — overconfident
(it rates fields ~1.0 but is right ~89%). **Temperature scaling** (`--calibrate`,
fit on the samples, no optimiser dependency) recalibrates it to **ECE ≈ 0.03**: a
scaled 0.89 confidence now means "right 89% of the time." There is still *no*
error-free threshold — a few wrong extractions are genuinely high-confidence, which
the harness reports honestly rather than claiming "0 wrong."

*Which signal to trust?* `--compare` puts the critic's **structural** confidence
head-to-head against the model's own **verbalized** confidence (asked per field) over
the same extractions. On DocILE the verbalized signal is marginally better calibrated
in aggregate (**ECE 0.084 vs 0.100**) but operationally useless — the model reports
~0.95 for almost every field, so it can't tell its right answers from its wrong ones
(it only separates at exactly 1.0: 7% coverage). The critic is slightly less
calibrated but actually *discriminates* — a usable operating point at confidence ≥ 0.55
(90% coverage, 10% error). Lower ECE ≠ more useful; you want a signal that both is
honest *and* separates.

*Multi-signal fusion (v4).* The blocker to a zero-error operating point is
*confidently wrong* extractions — and a model's own confidence is blind in exactly
the spots the model is wrong. The fix is a signal that fails *differently*:
**cross-model agreement** (a second, independent vision model reads the same page).
`apverify-eval-fusion` records per field — critic confidence, verbalized confidence,
each deterministic check, and cross-model agreement — against whether the field was
actually correct, then fits an interpretable logistic regression and scores every
signal on a **held-out** split. A fusion model is tied to one extractor's error
distribution, so each row carries its primary extractor and results are reported
per-extractor, never pooled.

On **511 field rows over 143 real DocILE invoices**, 5-fold cross-validated: fusion
scores **AUROC 0.70 (95% CI 0.60–0.80)** and is well-calibrated (**ECE 0.03**). What
carries it is the **arithmetic check (+1.6)**, **verbalized confidence (+1.1)** and the
**critic (+0.7)**. (An earlier single-split n=67 sample read 0.84 — optimistic; the
cross-validated number at 8× the data, with a confidence interval, is the honest one.)

The **per-invoice gate is the sobering, honest result**: at the document level (a
document is correct only if *every* field is) there is **no error-free auto-approval
threshold yet** — at trust ≥0.85, 52% of invoices auto-post but at ~40% document error.
The cause is arithmetic, not a bug: per-field F1 0.92 **compounds** over ~6 fields to
≈0.9⁶ ≈ 50% of full-page invoices having at least one wrong field. So strong per-field
accuracy does **not** equal safe per-invoice automation — the lever is the weakest
fields (vendor 0.71, line items 0.66), and that is the next thing to improve.

**Selective autonomy — the honest "full autonomous" metric.** Auto-approving *every*
invoice is the wrong goal; the right one is auto-approving the most invoices that can
be *proven* safe and escalating the rest. `apverify-eval-fusion` reports a
selective-autonomy curve — the maximum share auto-approvable within an error budget
(`operating_point_at` / `autonomy_curve`). On the robust dataset (Gemini × Groq,
**n=48 held-out**) **no budget ≤5% is met** — the floor is ~40% document error, so
selective autonomy does not yet work with a weak second model and a weakest-field
trust signal. A small underpowered Gemini × **Ollama** split (n=5, treat as a
hypothesis not a result) *does* show a clean ~40%-coverage / 0-error point, suggesting
a **stronger second model unlocks autonomy** — the next thing to test at scale when
quota allows. The metric and the curve are built; the number is reported as-is.

The sharp, honest finding is about the cross-model leg: its value depends not on the
second model's accuracy but on its *error decorrelation* from the primary — so the
harness **measures it directly** (the AUROC of "the two models agree" predicting
correctness). With Groq as the second model against Gemini, that number is **0.30 —
below chance**: Groq is weak enough that it disagrees even when Gemini is right,
flooding false flags, so fusion correctly down-weights it. (In a *Mistral*-primary
run the same Groq leg helped — catching 47% of confidently-wrong fields — proof the
signal lives or dies on the pairing, not raw accuracy.) Swapping in a genuinely
independent second model bears this out: against the same Gemini primary, a **local
Ollama** model moves the decorrelation AUROC from Groq's 0.30 to **0.625** — the
independent leg catching disagreements the correlated one missed (early, on a small
error count, and still accumulating). Both axes (discrimination + calibration) are
always reported, because a single number lies.

*Sampling-based uncertainty — a negative result we kept.* The same question asked of
a *single* model instead of two: resample the page five times and measure
self-agreement — self-consistency (modal-cluster share) and semantic entropy (Shannon
entropy over meaning clusters). It does **not** predict correctness — AUROC **0.55 on
Gemini, 0.57 on Groq**, essentially chance on both a strong *and* a weak extractor.
When the model misreads a field it misreads it *consistently* across resamples, so
agreement cannot flag it. The lesson sharpens the cross-model finding: *independence*
is the signal, not repetition — a different model disagreeing is informative; the same
model agreeing with itself is not. `apverify-eval-uncertainty` reports it; the null is
published because it is the honest boundary of the approach.

*Provider resilience.* Extraction chains independent vision providers
(Gemini → Groq → Mistral, plus an optional local Ollama); if one is rate-limited or
unavailable the next takes over, so the pipeline degrades rather than fails. The same
independence makes any two of them a cross-model pair for fusion — and a local Ollama
leg supplies that second opinion with no rate limit at all.

*Duplicate-fraud detection (v5).* Trust extends from "accurate" to "safe": duplicate
fraud is the largest share of AP fraud — the same invoice resubmitted, OCR noise across
channels, or a small edit to slip past an exact-match check. `apverify-eval-fraud`
matches a candidate against a ledger of prior invoices and returns both a discrete
**tier** (the explainable reason a flag ships with — `EXACT_RESEND`, `OCR_VARIANT`,
`NEAR_DUPLICATE`) and a continuous **score** (swept into a catch-rate-vs-false-positive
curve, the same shape as the calibration view). The hard part is *not* flagging a
legitimate recurring charge — a monthly retainer shares vendor and amount with last
month's — so the date is the discriminator: a resend shares the original date, a
retainer does not. On a synthesized benchmark (real invoices with injected duplicates
**and** legitimate look-alikes, since labelled fraud data is scarce), the matcher
catches **100% of duplicates at 0% false-positive** (AUROC 1.0), and **never flags the
recurring retainer** — the result that shows it is discriminating, not pattern-matching
on vendor+amount. Every flag carries a human-readable reason (*"same vendor + amount +
date; invoice-no INV-1001↔INV-l00l differs only by OCR-confusable characters"*), pure
domain logic with no ML dependency. Wired into the live pipeline behind an invoice
ledger: a confirmed resend of a posted invoice holds the payment (a near-duplicate
routes to a human).

*Vendor-master / bank-change / BEC (v5).* The highest-*loss* fraud vector: business-
email-compromise redirects payment by changing a known vendor's bank account at the
last minute, or impersonating a vendor with a typo-squatted name. `apverify-eval-bec`
checks an invoice against a vendor master and returns a severity-tiered, explainable
flag — `BANK_CHANGE` and `IMPERSONATION` are **HIGH** (they hold the payment),
`NEW_PAYEE` is **LOW** (surfaced for a human but never blocks, since new vendors are
legitimately common). Name matching is deliberately *not* confusable-folded — folding
would collapse a `Stee1` typo-squat onto the real `Steel` and hide the impersonation;
the name-similarity to the nearest known vendor is the score that separates an
impersonation from a genuinely new vendor. On the synthesized benchmark it catches
**100% of bank-change + impersonation at 0% false-positive** (impersonation AUROC 1.0),
and the check is **wired into the live pipeline**: a changed bank account drives
`apverify review` to HOLD with the reason recorded in the audit trace. (Synthetic only —
DocILE labels carry no bank-account data.)

*Anomaly detection (v5).* Flag invoices that are statistically unusual for their
vendor: an **amount spike** (a total far from the vendor's history, via a median/MAD
robust z-score so one historical outlier can't mask the next) or **threshold gaming**
(an amount parked just under a round approval limit). The detector is pure robust
statistics — **no ML dependency** — and ships a reason with every flag, wired into the
pipeline (HIGH → HOLD, MEDIUM → review). The interesting part is the **honest
head-to-head**: `apverify-eval-anomaly` pits it against an optional scikit-learn
Isolation Forest (the `[anomaly]` extra; eval-only, never on the production path). On
the synthesized benchmark the pure detector **beats** Isolation Forest — **AUROC 1.0 /
100% caught / 0% false-positive vs 0.96 / 96% / 0%** — because its engineered
threshold-proximity feature catches the gaming cases a generic outlier model can't see.
ML earns its dependency only if it wins; here it doesn't. (A z-floor gate keeps a
coincidental in-range amount near a round number from being mistaken for gaming.)

*The combined fraud layer (v5).* The three detectors above run as independent stages,
so the question that matters for a CFO is the *system* number, not three separate ones.
`apverify-eval-fraud-suite` synthesizes one stream exercising every fraud type — each
case carrying a full prior-invoice ledger, vendor master, and per-vendor history — and
runs **all three detectors on every case**, flagging it if any fires. This measures the
combined catch-rate **and** the system-wide false-positive rate (a clean invoice flagged
by *any* detector — cross-talk that per-detector benchmarks can't see). On the
synthesized suite the combined layer catches **100% of fraud at 0% false-positive
(precision 100%)**, with each fraud type attributed to its own detector and no clean
invoice flagged by any of them.

*Explainability (v5).* Explainability is non-negotiable for finance, so every fraud
flag carries a structured, ranked explanation, not just a sentence. `apverify-explain`
shows the honest version: the v4 fusion model is *linear*, so each signal's contribution
to the score is exactly `weight × feature` — SHAP for a linear model, in closed form, no
`shap` dependency and nothing approximated. The rule detectors are glass-box, so their
explanation enumerates the conditions that actually fired. The running pipeline attaches
a ranked `Explanation` to every held payment (alongside the free-text reason), so a held
invoice answers *why* in auditable, SOX-style terms.

*Collusion detection (v5).* Collusion hides not in invoice text but in *who approves
what*, so `apverify-eval-collusion` works over an approval log, not a single invoice. It
flags the behavioural fingerprints of a colluding approver-vendor pair: an approver who
**funnels** one vendor, clears amounts parked **just under** their own authorization
limit, and **rubber-stamps** them within seconds of submission. On the synthesized
benchmark it catches **100% of colluding pairs at 0% false-positive (AUROC 1.0)**, and a
one-off approval is never flagged (a minimum-history floor). Behaviour is the real
signal here — text analysis of invoices is deliberately *not* used, because it does not
reveal collusion.

**Throughput.** The verification layer runs on every invoice, so its cost is what
compounds at scale. `apverify-bench --count 1000` sustains **~4,000 invoices/sec
on a single core** (sub-millisecond p95), measuring the critic in isolation; the
real production bottleneck is the I/O-bound extraction call, which the worker pool
is sized for.

## The pipeline

```
document → extract (vision LLM) → critic → 3-way match → approve
                                    │           │           │
                            per-field         invoice vs    AUTO_APPROVE
                            confidence        PO + GRN       HOLD
                                                             HUMAN_REVIEW
```

**Critic** — three independent layers verify the extraction, cheapest first:

| Layer | Question | Catches |
|---|---|---|
| OCR cross-check | Does each value actually appear on the page? | Hallucinated totals, invented GSTINs |
| Arithmetic | Do line items, subtotal, tax, total reconcile? | A single misread/transposed digit |
| Format | Valid GSTIN checksum, parseable date, known currency? | Structurally invalid fields |

The OCR cross-check folds common OCR confusables (`0↔O`, `1↔I/L`, `5↔S`, …) so a
scanner misreading a leading `0` as `O` doesn't hold a perfectly good invoice.

**Matcher** — fuzzy 3-way match of the invoice against the purchase order and
goods-receipt note: vendor (name + GSTIN), PO reference, per-line price and
quantity within tolerance, partial deliveries, and "never bill more than ordered
or received".

**Approver** — combines the two verdicts into one decision and takes the more
cautious: a hallucinated total holds the payment *even when the PO matches*, and a
clean extraction against a mismatched PO routes to a human.

**Auditor (optional, `--audit`)** — a Groq LLM gives a second opinion on the
fields the deterministic critic was least sure about, catching values that are
wrong but plausible and on-page. It can *veto* an auto-approval but never
manufacture one; the deterministic checks remain the floor. (Reserved for
low-confidence fields, since it is the one paid, higher-latency step.)

**Self-consistency (optional, `--cross-check`)** — re-extracts the page with a
second, independent vision model (Mistral Pixtral) and flags any field where the
two disagree; a disagreement on a critical field holds the payment. A flaky
second opinion degrades to "no signal" rather than blocking the pipeline.

### Demo

```
$ apverify review samples/clean_invoice_01.pdf -p samples/procurement.json
  critic   AUTO_APPROVE @ 100%
  match    MATCHED
  approve  AUTO_APPROVE                                  # exit 0

$ apverify review samples/corrupted_total.pdf -p samples/procurement.json
  critic   HOLD @ 30%   (total · arithmetic: subtotal+tax 184200 vs total 184000)
  match    MATCHED
  approve  HOLD — extraction confidence 30%; flagged: total   # exit 1
```

The corrupted invoice's PO *matches* — only the critic catches that the printed
total doesn't reconcile, and its hold overrides the clean match.

## Architecture

Strict **clean / hexagonal** layering — dependencies point inward only, so the
verification logic (the IP) is pure and fully testable with no model, OCR engine,
or network:

```
interface (CLI)  ──▶  application (use cases + ports)  ──▶  domain (entities, critic, matcher, approver)
        │                                                            ▲
        └────────▶  infrastructure (Gemini, Tesseract, PDF, repos)  ─┘
```

- **domain/** — pure, no I/O. `Invoice`/`PurchaseOrder`/`GoodsReceiptNote`
  entities, self-validating value objects (`Money`, `Gstin` with Luhn mod-36),
  the critic (`checks.py`), the matcher (`matching.py`), the approver
  (`approval.py`).
- **application/** — the `ProcessInvoiceUseCase` (v0) and `ReviewPayableUseCase`
  (full pipeline with an audit trace), plus the `Protocol` ports they depend on.
- **infrastructure/** — adapters: Gemini extractor, Tesseract OCR, PDF renderer,
  procurement repository/loader, typed settings. SDK imports live *only* here.
- **interface/** — the `typer` CLI and its composition root.
- **eval/** — the offline benchmark: synthetic generator, corruptor, metrics,
  runner. Deterministic and network-free, so it gates CI.

No SDK import ever appears in `domain` or `application`; adapters are wired
together in exactly one place (`interface/cli/bootstrap.py`).

## Setup

Requires Python 3.11+, plus two system binaries for the live pipeline:

```bash
brew install tesseract poppler      # OCR engine + PDF rasteriser

python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env                # then add your GEMINI_API_KEY
python samples/generate_samples.py  # regenerate the sample invoices
```

> **Security:** `.env` is gitignored and must never be committed. **Rotate any key
> that was ever pasted in plaintext** before the repo goes public.

### Run locally — no API keys

The whole pipeline runs on a local vision model via [Ollama](https://ollama.com), so
you can try it with **no cloud accounts, no API keys, and no per-invoice cost** — the
cloud providers are optional accelerators, not requirements.

```bash
brew install ollama && ollama serve     # or the platform installer
ollama pull qwen2.5vl:7b                 # the local vision model (~6 GB)

# .env needs only these two lines — no GEMINI/GROQ/MISTRAL keys:
echo "OLLAMA_ENABLED=true" >> .env
echo "OLLAMA_MODEL=qwen2.5vl:7b"  >> .env

apverify run invoice.pdf                 # extract + critic, fully local
```

The extractor resolves to Ollama-only when no cloud keys are present; add a
`GEMINI_API_KEY` later and the resilient fallback (Gemini → Groq → Mistral → Ollama)
turns on automatically. On real DocILE invoices the local model is the strongest
non-Gemini option measured here (macro-F1 ≈ 0.81), which makes the keyless path a
genuinely usable default rather than a toy.

## Usage

```bash
apverify run    invoice.pdf                                      # extract + critic
apverify review invoice.pdf -p samples/procurement.json          # full pipeline + 3-way match
apverify review invoice.pdf -p samples/procurement.json --audit  # + Groq LLM auditor
apverify review invoice.pdf -p samples/procurement.json --cross-check  # + Mistral 2nd extraction
apverify-eval   --count 25 --save baseline.json                  # accuracy benchmark (+ snapshot)
apverify-bench  --count 1000 --corrupt-ratio 0.3                 # throughput benchmark
apverify-drift  --baseline baseline.json                         # fail if reliability regressed
apverify-eval-cord --split test --limit 100                      # critic on real CORD-v2 receipts
apverify-eval-docile --dataset-path ./docile --limit 200         # critic on real DocILE invoices
apverify-eval-accuracy --dataset docile --dataset-path ./docile  # per-field + line-item P/R/F1
apverify-eval-accuracy --dataset cord                            # (CORD downloads itself)
apverify-eval-leaderboard --dataset docile --dataset-path ./docile  # rank every provider on real invoices
apverify-eval-calibration --dataset docile --dataset-path ./docile  # ECE + risk-coverage
apverify-eval-fraud --dataset synthetic                          # duplicate catch-rate vs false-positive
apverify-eval-bec --count 25                                     # vendor-master / bank-change / BEC
apverify-eval-anomaly --count 25                                 # amount-spike / threshold-gaming (pure vs Isolation Forest)
apverify-eval-fraud-suite --count 25                             # combined fraud layer: catch-rate vs false-positive
apverify-explain                                                 # ranked linear attribution behind a fused score
apverify-eval-collusion --pairs 6                                # behavioral collusion: catch-rate vs false-positive
```

`apverify-eval-cord` (needs the `datasets` extra: `pip install -e '.[datasets]'`)
runs the critic over **CORD-v2** — real Indonesian receipts with ground-truth
labels — and reports the auto-approve rate plus a breakdown of *why* receipts get
held (e.g. totals that don't reconcile because of unmodelled discounts). This is
the messy long tail; the synthetic set is the controlled benchmark.

`apverify-eval-docile` runs the critic over **DocILE** (the largest public invoice
benchmark — real US/EU invoices, KILE + LIR annotations). On 200 `val` invoices the
critic auto-approves **72%**; the rest are held for genuine reasons (vendor names
that don't match the OCR, dates in formats worth flagging, totals that don't
reconcile where net/tax are labelled). DocILE annotates only a *subset* of fields
per document, so this adapter deliberately evaluates the key-amount cross-check and
format rather than reconciling against partial line-item ground truth — see
`apverify.eval.docile`. (Access is gated and non-commercial; download with your
token per the DocILE repo.)

Exit codes: `0` auto-approve · `1` needs attention (hold/review) · `2` error.

## Collections agent (WhatsApp)

The flip side of AP: chase the receivables *we* are owed. `apverify collect` sends
tiered WhatsApp reminders (gentle → firm → final, by how overdue) for each unpaid
invoice, and the API exposes an inbound webhook that verifies Meta's challenge,
checks the `X-Hub-Signature-256` HMAC, and classifies replies (paid / promise /
dispute / query).

```bash
apverify collect samples/receivables.json --dry-run --as-of 2026-06-30  # prints, sends nothing
apverify collect samples/receivables.json --no-dry-run                  # live (needs WHATSAPP_* keys)
```

`--dry-run` (the default) prints messages instead of sending, so the agent is
demoable without messaging anyone. The reminder tiers and the reply classifier are
pure domain logic; the WhatsApp Cloud API send and the webhook signature check are
adapters, both tested without the network (live send is opt-in only).

## Web UI

A FastAPI backend exposes the pipeline; a Vite + React + TypeScript frontend (an
"audit terminal" — dark ledger, Fraunces display, IBM Plex Mono data) lets you
upload an invoice and watch extract → critic → 3-way match → approve, with a
decision stamp, per-field confidence meters, the match breakdown, and the pipeline
trace. Run the two processes side by side:

```bash
# terminal 1 — API (reads .env for the Gemini key)
uvicorn apverify.interface.api.app:app --reload --port 8000

# terminal 2 — frontend (proxies /api to :8000)
cd frontend && npm install && npm run dev   # http://localhost:5173
```

The API is testable without network: `POST /api/review` takes the invoice plus
`audit` / `cross_check` flags; the use case is built from an injectable provider,
which the API tests replace with in-memory fakes.

Every step is timed (behind a `Tracer` port — `NullTracer` by default), so the
trace carries per-step latency. A typical run makes the point: the LLM extraction
is ~20 s while the whole verification layer (critic + 3-way match) is **~6 ms** —
trust is essentially free; cost lives in the model call. Set `LANGFUSE_PUBLIC_KEY`
/ `LANGFUSE_SECRET_KEY` (with the `langfuse` extra) and the same spans ship to
Langfuse — the adapter depends on a minimal client protocol, so the core never
imports the SDK.

## Development

```bash
ruff check . && ruff format --check .
mypy --strict src tests
pytest --cov
```

The test suite mirrors the layers: pure-domain tests with **zero mocks** (the
domain is at 100% coverage), use cases driven by in-memory fakes, adapter contract
tests that skip when a binary/key is absent, and a hermetic end-to-end CLI test.
CI runs lint + types + tests and a separate **eval gate** that fails the build if
the catch rate regresses.

## Roadmap

Shipped: **v0** (extract + critic) · **v1** (3-way match, approver, orchestrator, eval
harness, CI gate) · **v2** (Groq LLM-as-auditor, Mistral self-consistency, web UI,
WhatsApp collections, throughput) · **v3** (real-data extraction benchmark on CORD +
DocILE, per-field + line-item F1, error taxonomy) · **v4** (calibration / ECE +
temperature scaling, verbalized confidence, multi-signal fusion, sampling-based
uncertainty, k-fold CV, per-invoice auto-approval gate) · **v5** (fraud & anomaly:
duplicate, vendor-master/BEC, statistical anomaly, a cross-detector benchmark, an
explainability layer, and collusion detection — all wired into the pipeline).

Next:

- Scale the real-data results (fusion / calibration to a few hundred DocILE invoices with
  confidence intervals) and report fraud false-positive rates on real legitimate invoices.
- v6: model cascade for cost, a LoRA-tuned open VLM, a public benchmark + leaderboard,
  a hosted demo, and a technical writeup.
- Voice (Whisper) intake for the collections agent, alongside WhatsApp text.

## License

MIT.
