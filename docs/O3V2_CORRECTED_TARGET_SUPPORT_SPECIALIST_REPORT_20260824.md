# O3-v2 corrected target, support, and specialist funnel

## Status

Research-only.  No live bundle, admission rule, portfolio rule, exchange
process, or canonical document was changed.

The conclusion is **do not promote an O3-v2 specialist stack**.  The retained
T3 heads improve raw global-tail diagnostics but do not improve the matched,
dual-MC1, portfolio-constrained downstream stack over its first fully strict
evaluation interval.

## Contract

- Candidate and score receipts are target-free.  Policy outcomes are joined
  only after each held score receipt is written.
- Invalid/incomplete policy paths are excluded from supervised fitting and
  realised-policy metrics; they are never encoded as ordinary economic
  failures.
- Train labels must be resolved before the held fold's reserve boundary.
- Specialist fit: three calendar months ending before a 28-day reserve;
  score ranks use the sampled training prediction distribution only.
- Routing: existing timestamp-local base top 30% route.
- No MDA was run in this funnel.

The corrected semantic population has >99.3% valid label coverage.  The
full audit also verifies that held target, support, specialist, and adapter
receipts do not contain `policy_*` or `semantic_*` outcome fields.

## Corrected target screen

Target outputs use genuine ordinal classes where declared; earlier
continuous-label ordinal experiments are superseded exploratory artifacts.
All figures are mean net bps/trade across October 2025--July 2026 under the
75/25 diagnostic blend.

| Target | Top 1% | Top 2% | Top 5% | Worst Top 5% | Rank IC |
|---|---:|---:|---:|---:|---:|
| T0 current control | 293.26 | 243.86 | 188.24 | 65.87 | .157 |
| T1 economic residual LambdaRank | 362.82 | 302.00 | 223.13 | 26.87 | .162 |
| T2 economic residual ordinal | 349.33 | 300.83 | 201.73 | 70.77 | .172 |
| **T3 pair residual LambdaRank** | **341.13** | **284.25** | **207.47** | **86.89** | **.185** |
| T5 rank-error LambdaRank | 370.06 | 304.54 | 228.65 | 21.25 | .127 |
| **T6 rank-error ordinal** | **357.68** | **301.89** | **227.10** | **55.64** | **.140** |

T3 was retained for its near-tie, pairwise correction semantics and best
stability.  T6 was retained as a distinct ordinal rank-error concept.

## Support-weight selection

Development selection was October--December 2025.  The selected support
schemes were then frozen and evaluated in January--July 2026.

| Retained target/support | Forward Δ vs current O3 control, Top 1/2/5% | Forward worst-month Δ, Top 1/2/5% |
|---|---:|---:|
| T3 + SB1 error × archetype | +61.89 / +54.76 / +32.90 | +73.04 / +40.20 / +40.78 |
| T6 + SB3 error × semantic certainty | +74.37 / +56.12 / +29.66 | +95.09 / +36.15 / −8.40 |

These are raw tail diagnostics, not admission or portfolio metrics.

## Frozen feature screen

Each retained contract selects 40 causal fields: F1=6, F2=4, F3=6,
F4=10, F5=10, F6=4.  They are selected on training-only relevance/coverage
screens, with no outcome field admitted to the held feature panel.

- F1: base/upstream score geometry and disagreement.
- F2: timestamp-local rank geometry.
- F3: resolved-only recent residual/support telemetry.
- F4: causal market-state and cross-sectional fields.
- F5: parent current/BCF score provenance and agreement fields.
- F6: calendar controls.

## Specialist architectures

All results below are strict January--July 2026 OOS global-tail diagnostics.
Each held month has complete feature coverage; no specialist output is an
outcome feature.

| Target | Architecture | Mean Top 1% | Mean Top 2% | Mean Top 5% | Worst Top 5% | Positive months |
|---|---|---:|---:|---:|---:|---:|
| T3 | H1 family median | 72.60 | 57.71 | 45.85 | −9.50 | 6/7 |
| **T3** | **H2 population ensemble** | **339.74** | **273.38** | **181.46** | **5.25** | **7/7** |
| **T3** | **H3 F4/F5 hybrid** | **309.68** | **246.40** | **176.27** | **53.07** | **7/7** |
| T6 | H1 family median | −68.66 | −50.88 | −45.08 | −74.74 | 0/7 |
| T6 | H2 population ensemble | −3.91 | 21.93 | 20.63 | −54.73 | 4/7 |
| T6 | H3 F4/F5 hybrid | −17.60 | 5.78 | 25.50 | −59.11 | 5/7 |

Only T3/H2 and T3/H3 survived.  T6 was rejected: none of its architectures
met the stability gate.

The formal H1 complementarity audit explains why the original family medians
were weak: T3 F5 was the useful standalone family head; T6 F4 was useful in
isolation, but its family median averaged it with adverse heads.  H2/H3 test
whether such useful information can be preserved without the dilution.

## Strict downstream MC1 and portfolio test

The current preserved feature history begins in August 2025.  December 2025
is therefore the first specialist held month with a complete prior three-month
fit window.  June 2026 is the first month with six prior monthly specialist
score receipts (December--May) for the unchanged MC1 six-month prequential
fit.  This yields a fully strict June--July 2026 downstream test, not a
six-month downstream claim.

Conditions:

- current and BCF family maps are refit separately on six strictly prior
  months;
- common canonical rich-policy label ledger;
- dual MC1 admission at 30 or 50 bps;
- one shared chronological constrained portfolio;
- challenger and control evaluated on the exact common candidate identities.

| Input arm | Threshold | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched current-live control | 30 | 1,241 | **151.42** | 187,916.79 | **148.86** | **111.58** | **−0.21** |
| T3 H2 only | 30 | 1,628 | 121.67 | 198,079.08 | 107.82 | 56.90 | −0.26 |
| T3 H3 only | 30 | 1,744 | 115.39 | **201,244.42** | 95.06 | 58.36 | −0.22 |
| T3 H2 + H3 | 30 | 1,711 | 116.44 | 199,237.13 | 94.20 | 49.56 | −0.26 |
| Matched current-live control | 50 | 1,014 | **175.71** | 178,169.55 | **165.08** | **130.40** | −0.21 |
| T3 H2 only | 50 | 1,371 | 139.17 | 190,795.78 | 139.08 | 77.73 | −0.24 |
| T3 H3 only | 50 | 1,493 | 134.65 | **201,037.27** | 119.78 | 68.81 | −0.19 |
| T3 H2 + H3 | 50 | 1,454 | 133.56 | 194,194.40 | 118.68 | 66.65 | −0.24 |

The challenger produces higher total net bps only by admitting materially more
trades.  It loses 35--42 bps/trade, worsens worst-week outcomes by 53--64 bps,
and H2/joint worsen drawdown.  That fails the advancement criteria: the raw
ranking gain does not survive the real admission and portfolio layer with
enough capital efficiency or downside protection.

## Decision

Keep the existing live stack unchanged.  Preserve T3/H2 and T3/H3 as research
signals only.  Do not promote T6 or any O3-v2 specialist composition.

The next valid research question is not another broad target sweep.  It is
whether the T3 H2/H3 information can be used as a conservative *demotion* or
size/risk input inside the existing MC1 envelope, with authority only to
remove weak candidates.  That must be tested on a later untouched block.

## Key artifacts

- Corrected target screen:
  `data_perp/artifacts/strict_r3_o3v2_target_funnel_20260824_v2`
- Selected support forward screen:
  `data_perp/artifacts/strict_r3_o3v2_support_weight_forward_selected_20260824_v1`
- T3 H1 audit:
  `data_perp/artifacts/strict_r3_o3v2_head_quality_t3_h1_20260824_v2`
- Retained H2/H3 target-free adapters:
  `data_perp/artifacts/strict_r3_o3v2_adapter_t3_h2_20260824_v1` and
  `data_perp/artifacts/strict_r3_o3v2_adapter_t3_h3_20260824_v1`
- Matched downstream tests:
  `data_perp/artifacts/strict_r3_o3v2_mc1_t3_h2_junjul_20260824_v1`,
  `data_perp/artifacts/strict_r3_o3v2_mc1_t3_h3_junjul_20260824_v1`, and
  `data_perp/artifacts/strict_r3_o3v2_mc1_t3_h2_h3_junjul_20260824_v2`
- Correctness receipt:
  `data_perp/artifacts/strict_r3_o3v2_correctness_audit_20260824_v1.json`
