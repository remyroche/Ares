# C1-LVA: causal S/R plus value-area MC1 input — superseded research evidence

## Status

`C1_refit_core_plus_causal_sr` with the full causal levels/value-area context
was the principal research challenger to the post-February refit-core MC1 map.
For clarity, this archived evidence document calls that full contract
**C1-LVA**.

C1-LVA is superseded as a research-status document by the canonical no-order
inference contract in
[`C1_LVA_CANONICAL_INFERENCE_20260901.md`](C1_LVA_CANONICAL_INFERENCE_20260901.md).
Its source heads
were fit only on resolved pre-2026 interactions; June–July 2026 was the
initial downstream evaluation and August 1–18 was opened once as a sealed
confirmation.  As a result, a later untouched period is still required before
any promotion decision.  The earlier S/R-only C1 is now a comparator, not the
leading challenger.

**Operational scope:** research-only.  This document does not authorize a
change to the live feature contract, calibration bundles, admission rule,
portfolio auction, policy, or execution stack.

## Exact change

C1-LVA retains the paired BCF/current-v5 score families, source-aligned
parent-policy outcomes, 21-day prior-resolved residual shift, dual BCF/current
admission, BCF-MC1 auction priority, and the controlled long-only portfolio
auction.  It appends independently OOF causal price-structure snapshots and
the retained causal value-area context to the family-specific MC1 mapper:

- support-hold strength;
- resistance-break probability;
- downside-break probability;
- resistance-rejection strength;
- directional structure balance;
- nearest support/resistance distance in ATR;
- causal source-prior strength and reaction-magnitude estimates; and
- explicit snapshot availability.

The retained value-area context is exactly seven fields:

- `profile_poc_distance_atr`;
- `profile_vah_distance_atr` and `profile_val_distance_atr`;
- `profile_hvn_distance_atr` and `profile_lvn_distance_atr`; and
- `profile_inside_value_area` and `profile_value_area_width_atr`.

Snapshots are point-in-time and missingness is a mapper input, never a
candidate filter.  No future interaction label or policy outcome enters the
target-free admission panel.

## Causal contract

- A 1h pivot becomes usable only after three completed 1h bars; a 4h pivot
  only after two completed 4h bars.
- Structural snapshots are built before the scored decision timestamp.  The
  resolved reaction/break labels are retained solely to train the S/R heads;
  every held-month head fit excludes labels unavailable before that month.
- The S/R-to-score merge is one-to-one on candidate identity and timestamp;
  it must preserve the complete target-free candidate matrix exactly.
- The family-local MC1 fit additionally requires the parent rich-policy label
  to have resolved before its monthly held boundary.  The 21-day residual
  shift remains prior-resolved and score-band-local.
- Value-area state uses a trailing 21-day completed-hour profile on a fixed
  25-bps log-price grid.  POC, HVN/LVN, VAH/VAL and value-area geometry are
  snapshots known before the decision; they are neither a future-path label
  nor an eligibility filter.

Focused causal-merge and ontology-contract tests pass for the retained C1
source and for the new structural variants.  This is a research receipt, not
an authorization to modify the live mapper.

## Primary constrained portfolio evidence

All figures use the matched source-aligned parent-policy replay, dual +50-bps
BCF/current admission, BCF-MC1 auction priority, and the controlled long-only
portfolio constraints.  The refit-core is the direct deployable comparator.

| Period | Arm | Accepted trades | Net EV / trade | Net contribution | Worst month | Worst week | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|
| June–July 2026 | refit core | 183 | +264.05 bps | +48,321.51 bps | +227.92 bps | +72.96 bps | -37.27% |
| June–July 2026 | **C1-LVA** | **465** | **+197.74 bps** | **+91,947.72 bps** | +167.77 bps | **+120.76 bps** | **-31.48%** |
| Aug. 1–18, 2026 | refit core | 23 | +555.70 bps | +12,781 bps | n/a | +365.69 bps | -8.14% |
| Aug. 1–18, 2026 | **C1-LVA** | **98** | **+277.61 bps** | **+27,205.70 bps** | n/a | +148.19 bps | **-4.71%** |

C1-LVA increases June–July total contribution by +43,626 bps and improves
maximum drawdown by 5.80 percentage points relative to refit core, while
trading more often.  In sealed August it adds 75 accepted positions and
14,425 bps total contribution, with a 3.43 percentage-point drawdown
improvement.  This is not an indiscriminate expansion: every C1-LVA
acceptance had a causal S/R and value-area snapshot; missingness stays an
explicit mapper input rather than a candidate filter.

The result is broad enough to merit a challenger but is not yet free of
concentration risk.  June--July C1-LVA covers 72 symbols and trades on all 61
days; its largest symbol is 5.16% of entries and its five largest symbol
contributions are 30.78% of total bps.  The partial August confirmation
covers 42 symbols and 13 active days, but its five largest symbols account
for 63.86% of contribution.  Later untouched validation must specifically
test that the August gain is not dependent on that concentration.

## Expanded compatible-panel replay (April--August 18, 2026)

The initial primary receipt intentionally starts flat in June and the sealed
August confirmation intentionally starts flat in August.  The following
separate receipt carries **one common controlled portfolio state** from 1
April through 18 August, and is therefore the appropriate month-by-month
extension.  It retains exactly the frozen BCF/current-v5 score panels,
prior-resolved rich-parent labels, monthly family-local HGB refits, C1-LVA
inputs, dual +50-bps admission and BCF-MC1 auction priority.  It does not use
E2, live execution, post-decision data, or a later score family.

| Month | No-C1 core: trades | No-C1 core: net bps/trade | C1-LVA: trades | C1-LVA: net bps/trade | Delta trades | Delta net bps/trade | Delta total net bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jan.--Mar. | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| April | 616 | +206.28 | 629 | +212.11 | +13 | +5.83 | +6,349.19 |
| May | 521 | +154.58 | 550 | +153.54 | +29 | -1.04 | +3,911.68 |
| June | 132 | +275.72 | 174 | +245.08 | +42 | -30.64 | +6,248.96 |
| July | 48 | +227.92 | 288 | +167.77 | +240 | -60.15 | +37,377.25 |
| Aug. 1--18 | 23 | +555.70 | 98 | +278.14 | +75 | -277.56 | +14,476.46 |
| **Apr.--Aug. 18** | **1,340** | **+199.79** | **1,739** | **+193.26** | **+399** | **-6.53** | **+68,363.55** |

Over the common April--August 18 ledger, C1-LVA improves worst-week quality
from +72.96 to +84.92 bps and maximum drawdown from -37.27% to -31.48%
(+5.80 percentage points), while the worst-month mean changes only -1.04
bps/trade.  The trade-count and total-contribution uplift is therefore broad,
but it comes with a modest aggregate EV/trade trade-off; this is still a
research challenger, not a promotion result.

January--March are intentionally not backfilled: the retained compatible
score history begins in February and the monthly refit requires at least 5,000
strictly prior resolved policy labels, making April the earliest defensible
held month.  August 19--31 are likewise marked unavailable rather than
estimated.  The frozen paired BCF/current-v5 archive ends at 18 August 21:00
UTC; the only later current-score panel has material score deltas on its
overlap (final-score max delta 0.647), so it is a different score-family
contract and cannot be mixed into C1-LVA.  A valid August 19--31 comparison
requires recovery of the frozen current-v5 scorer plus its target-free source
panel, not reuse of the later router/current panel.

The immutable expanded receipt is
`data_perp/artifacts/causal_sr_c1_lva_apr_aug18_continuous_20260901_v2`.

## What the S/R base is doing

These are diagnostic results for the S/R base that C1-LVA retains.  The
value-area fields are an additional, separately validated input block; their
economic evidence appears below.  The mapper change is concentrated where a
live, point-in-time S/R snapshot exists.  Neither C1 nor C1-LVA treats missing
snapshots as a hidden rejection gate: coverage is an explicit model input and
the core map remains usable on the uncovered population.

| Family / covered rows | June 2026 outcome-rank correlation | July 2026 outcome-rank correlation |
|---|---:|---:|
| BCF refit core | 0.167 | 0.227 |
| BCF + C1 | 0.188 | 0.249 |
| current-v5 refit core | 0.227 | 0.158 |
| current-v5 + C1 | 0.316 | 0.201 |

The realized dual-map surface is also directionally calibrated on the covered
population.  In July, C1 rows with minimum(BCF EV, current EV) below zero
averaged -43.2 bps, versus +66.9 bps in the 0--50 bucket, +162.9 bps in
50--100, +226.7 bps in 100--200, and +455.5 bps above 200.  This is a
diagnostic only; the frozen admission threshold remains unchanged.

## Completed decomposition tests

The decomposition was selected on June--July and opened August once for
confirmation.  It shows that both semantic blocks matter: the adverse-break
block is insufficient, while the support/structure-only block is attractive
in selection but loses August quality.  Full C1 therefore remains the
challenger rather than being narrowed after the fact.  Each row is a complete
family-local mapper refit, so the comparison measures deployable component
contracts, not additive feature attribution.

| Arm | Window | Trades | Net EV / trade | Net contribution | Worst month | Worst week | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|
| C1 adverse-break/rejection block | Jun--Jul | 400 | +185.55 bps | +74,219.12 bps | +142.82 bps | +79.48 bps | -34.00% |
| C1 support/structure block | Jun--Jul | 445 | +201.17 bps | +89,519.25 bps | +172.56 bps | +148.45 bps | -34.00% |
| Full C1 | Jun--Jul | 438 | +188.09 bps | +82,385.48 bps | +152.87 bps | +105.49 bps | -31.69% |
| C1 support/structure block | Aug. 1--18 | 107 | +130.64 bps | +13,978.37 bps | n/a | +104.31 bps | -28.00% |
| Full C1 | Aug. 1--18 | 107 | +225.48 bps | +24,126.41 bps | n/a | +129.81 bps | -16.80% |

## Directional OI positioning result

Long-build and short-build OI zones are deliberately separate from price S/R:
they cannot role-reverse, and their labels distinguish defended, failed,
trapped and unwound states.  OI snapshots use only a strictly prior OI
observation (median available source coverage: 97.25%).  The source heads are
predictive in every held month -- roughly 0.55--0.59 AUC for defended/failed
states, with short-build most stable in August -- but they have not earned
downstream use:

| Arm, June--July 2026 | Trades | Net EV / trade | Net contribution | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Full C1 | 438 | +188.09 bps | +82,385.48 bps | +152.87 bps | +105.49 bps | -31.69% |
| C1 + causal directional OI heads | 420 | +191.59 bps | +80,469.43 bps | +156.16 bps | +104.22 bps | -31.69% |

The OI heads remain diagnostics only: their lower participation and lower
total contribution do not clear the predeclared downstream gate.

## Structural ontology screen

The narrow-level variant was rejected at the source stage before any MC1
rerun: its conditional-strength correlation fell from 0.245 to 0.236 and its
accepted-break AUC from 0.626 to 0.620 across June--August 2026.  The
independent-retest variant is also rejected: its accepted-break AUC improves
slightly to 0.631 and Brier improves from 0.197 to 0.192, but its
conditional-strength correlation falls to 0.231.  A mixed source result does
not clear the stability gate, so neither received an MC1 rerun.  The longer
barrier-grid/no-speed variant is rejected too: conditional-strength
correlation falls to 0.207, accepted-break AUC to 0.615, and reaction-
magnitude correlation to 0.201.  No new price-structure ontology cleared the
predeclared source gate; C1 therefore remains the only structural mapper
challenger and no additional MC1 retraining was warranted.

## Prior S/R-only routing control (not the C1-LVA contract)

`R1_C1_when_sr_available_else_C0` is the earlier S/R-only routing control.  It
routed through C1 only when the decision already had a causal S/R snapshot;
all other rows retained the refit-core map.  Across June–August it produced
561 accepted trades, +194.89 bps/trade and +109,330.44 bps total, compared
with C1-all's 545, +195.53 and +106,563.67 bps.  It is retained as a
diagnostic comparator only: it is not a substitute for, and has not been
combined with, the current full C1-LVA contract.

## Required falsification before any promotion

1. Freeze the seven-field value-area block, C1 source-head contract, map
   schedule, dual admission and BCF-priority auction unchanged.
2. Confirm the full C1-LVA contract on a later untouched period, including
   contribution concentration, drawdown, CVaR and worst-week stability.
3. Keep source/head outputs OOF and test only those retained outputs as MC1
   inputs under the same policy labels and portfolio constraints.
4. Do not add channels, time-at-price/balance, OI-at-price, directional-OI or
   selective routing to C1-LVA without a newly predeclared trial.

## Volatility, participation and anchored-VWAP screen — rejected

This late-stage screen asked whether generic, causal market-state fields could
improve the **existing C1-LVA S/R heads**, then improve the downstream paired
BCF/current MC1 maps.  It was deliberately not an eligibility filter, new
ranker, or policy change.  Every candidate continued through the same
source-aligned parent-policy labels, dual +50-bps admission, BCF-EV priority,
and constrained global portfolio replay.

The common source substrate is the repaired `v3` causal profile state: 114
source-ready symbols with explicit field-level missingness, 2025-only source
head training, and June--August 2026 held-month scoring.  The 46 locally
corrupt 15-minute symbol files remain fail-closed; every arm uses the same v3
candidate/feature universe, so the comparison is matched.  This is a
diagnostic branch, not a replacement for the primary C1-LVA receipt above.

The new causal fields were:

- volatility / participation: trailing 21-day ATR and volume percentiles,
  4h-versus-24h realised-volatility and range ratios, relative volume, and
  volume acceleration; and
- anchored VWAP: UTC-session and UTC-week distance and slope in ATR units,
  session cross age, and the last-four-hour aligned close fraction.

The retained C1 interaction-path block was also removed as a negative
control.  It includes approach return/velocity/acceleration, path efficiency,
directional consistency, impulse/pullback structure, sign-flip rate,
largest-bar share, range compression, near-zone close fraction, relative
volume, and volume acceleration.  Those fields are causal and already
available in the base S/R snapshot; the ablation tests whether they are doing
real work rather than merely increasing model width.

### Source-head qualification, Jun--Aug 2026 average

| Context relative to v3 LVA control | Conditional-strength Spearman | Accepted-break AUC | Break Brier ↓ | Reaction-magnitude Spearman | Source decision |
|---|---:|---:|---:|---:|---|
| v3 LVA control | 0.24273 | 0.62375 | 0.19734 | 0.21468 | reference |
| Remove retained interaction-path block | 0.06423 | 0.54860 | 0.20545 | 0.12132 | reject; control is essential |
| Add volatility / participation | 0.24135 | 0.62685 | 0.19698 | 0.22136 | downstream test |
| Add anchored VWAP only | 0.24231 | 0.62336 | 0.19734 | 0.21560 | reject at source gate |
| Add both new blocks | 0.24274 | 0.62536 | 0.19714 | 0.22262 | downstream test |

The source results correctly identify the retained interaction block as
important.  They do **not** establish deployable utility for the new fields;
that decision is made only by the following constrained portfolio replay.

### Matched downstream MC1 result, June--July 2026

All four rows use the same v3 target-free 15-minute cache, pair of retained
score families, monthly refit schedule, parent rich-policy labels, dual
admission, BCF priority, and portfolio constraints.  The summary receipts
were reconstructed only from completed immutable replay ledgers after an
`--only-arms` aggregate-writer defect; the finalization receipts hash every
input ledger and attest zero exchange calls.  No fit, score, admission,
auction, or outcome was recomputed during finalization.

| C1-LVA context | Trades | Dual admissions | Net EV / trade | Net contribution | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v3 LVA control** | **483** | **1,172** | **+196.90 bps** | **+95,101.43 bps** | **+172.60 bps** | **+133.75 bps** | **-31.07%** |
| + volatility / participation | 484 | 1,145 | +194.39 bps | +94,082.56 bps | +167.20 bps | +107.99 bps | -31.48% |
| Remove interaction-path block | 406 | 1,033 | +165.02 bps | +66,996.49 bps | +130.44 bps | -9.68 bps | -47.26% |
| + volatility / participation + anchored VWAP | 466 | 1,087 | +194.94 bps | +90,842.68 bps | +165.10 bps | +96.62 bps | -33.01% |

The closest challenger, volatility/participation alone, loses 2.51 bps per
trade, 1,018.87 bps of total contribution, 5.40 bps of worst-month quality,
25.76 bps of worst-week quality, and 0.40 percentage points of drawdown to
the matched v3 LVA control.  Adding anchored VWAP is worse still on every
predeclared portfolio/risk criterion.  Removing the retained interaction block
is decisively harmful, including a negative worst week.

No variant clears the June--July downstream selection gate, so no new August
confirmation was run.  Opening August after this failure would be post-hoc
feature mining.  **Decision:** retain the existing C1-LVA seven-field
levels/value-area block and its existing causal S/R interaction-path features;
reject these incremental volatility, generic participation, and anchored-VWAP
fields for C1-LVA and for the live/canonical stack.

## Market-profile / channel context screen — rejected

This screen added a strictly causal profile context to the *existing C1 S/R
heads*, rather than using the profile as an independent alpha or eligibility
filter.  The source state is a 21-day rolling completed-hour profile on a
fixed 25-bps log-price grid.  It contains HVN/LVN/POC/VAH/VAL distances,
value-area and time-at-price balance geometry, signed strictly-prior
delta-OI-at-price positioning, and completed-hour Bollinger(20,2),
Keltner(EMA20,2ATR), and Donchian(20) geometry.  Missing profile/OI data is
an explicit source-head missing value and never removes a candidate.

The broad profile bundle is rejected.  C1-LVA below is the deliberately
isolated exception: it retains only the independently downstream-validated
seven-field levels/value-area block, not the broad bundle.

All source-head models were fit only on the frozen **1,021,219 resolved
pre-2026 C1 interactions**.  June, July, and August 2026 were
confirmation-only.  The
profile context improved the C1 source task in every confirmation month:

| Source metric, Jun--Aug average | Frozen C1 | C1 + profile context | Delta |
|---|---:|---:|---:|
| Conditional reaction-strength Spearman | 0.2430 | 0.2448 | +0.0018 |
| Accepted-break AUC | 0.6318 | 0.6355 | +0.0036 |
| Accepted-break Brier | 0.19631 | 0.19597 | -0.00034 |
| Reaction-magnitude Spearman | 0.2154 | 0.2380 | +0.0226 |

The lower-level direct profile utility model was also retained as a negative
diagnostic: its 2026 utility ranking was essentially flat and its adverse
break AUC did not hold past June.  It was not used in the final C1-context
comparison.

Despite the source improvement, the profile context does **not** clear the
downstream MC1 gate under the exact dual-admission, BCF-priority and
portfolio-constrained parent-policy replay.  The two rows below use the same
2025-trained C1 head schedule; only the causal profile context differs.

| Period | Frozen C1 | C1 + profile context | Delta (profile - frozen) |
|---|---:|---:|---:|
| June 2026 contribution | +42,510.76 bps | +41,898.60 bps | -612.16 bps |
| July 2026 contribution | +47,582.64 bps | +47,862.24 bps | +279.60 bps |
| Aug. 1--18 contribution | +25,543.16 bps | +25,169.65 bps | -373.50 bps |
| June--Aug accepted trades | 579 | 578 | -1 |
| June--Aug net EV / trade | +199.72 bps | +198.84 bps | -0.88 bps |
| June--Aug net contribution | +115,636.56 bps | +114,930.49 bps | -706.07 bps |
| Worst month | +158.08 bps | +162.24 bps | +4.16 bps |
| Worst week (within measured windows) | +108.02 bps | +93.92 bps | -14.11 bps |

The profile context therefore remains a documented rejected branch: it is
useful for structural diagnostics and improves level-interaction prediction,
but it does not improve the deployed MC1 economic objective.  It must not be
added to C1, the live stack, or the canonical document without a newly
predeclared source and downstream trial.

### Feature-family decomposition — levels/value-area is the exception

The broad-profile rejection did **not** establish that every profile feature
is unhelpful.  The same causal state was decomposed into four independently
fit source-head contexts.  Each train fit used the identical frozen
pre-2026 interaction population (1,021,219 resolved interactions); June,
July, and August 2026 remain confirmation-only.  Source availability is
proportional to the selected block (at least 70% populated), is an explicit
model field, and never filters a candidate.

| Context family | Conditional-strength Spearman Δ | Break AUC Δ | Break Brier Δ | Magnitude Spearman Δ | Source decision |
|---|---:|---:|---:|---:|---|
| Levels/value area (POC, VAH/VAL, HVN/LVN) | +0.0021 | +0.0003 | +0.00004 | +0.0114 | advance |
| Time-at-price / balance | -0.0006 | +0.0002 | -0.00001 | -0.0011 | reject at source gate |
| ΔOI-at-price / positioning | +0.0013 | -0.0010 | +0.00013 | -0.0010 | reject at source gate |
| Bollinger/Keltner/Donchian | +0.0007 | +0.0031 | -0.00025 | +0.0146 | advance |

The two source-qualified blocks were then sent through the unchanged
family-specific MC1 map, dual +50-bps BCF/current-v5 admission, BCF-MC1
priority, source-aligned parent-policy labels, and controlled global
portfolio auction.  The frozen-C1 rows are the same matched baseline in every
comparison.

| Period | Frozen C1 contribution | Levels/value-area | Δ levels | Channels | Δ channels |
|---|---:|---:|---:|---:|---:|
| June 2026 | +42,510.76 bps | +43,630.44 bps | +1,119.68 | +42,251.55 bps | -259.21 |
| July 2026 | +47,582.64 bps | +48,317.28 bps | +734.64 | +44,307.91 bps | -3,274.73 |
| Aug. 1--18 2026 | +25,543.16 bps | +27,205.70 bps | +1,662.54 | +21,542.71 bps | -4,000.45 |
| Jun.--Aug. total | +115,636.56 bps | **+119,153.42 bps** | **+3,516.85** | +108,102.17 bps | -7,534.39 |

Levels/value-area also improves selected-trade EV from +199.72 to
**+211.64 bps/trade** over the joined windows, while using 563 rather than
579 accepted trades.  Its monthly EV/trade is +246.50, +167.77, and +277.61
bps in June, July, and August respectively (C1: +244.31, +158.08, +245.61).
For June--July, worst week improves from +108.02 to +120.76 bps and max
drawdown is unchanged within 0.01 percentage points; in August worst week
improves from +114.38 to +148.19 bps and max drawdown from -16.80% to -4.71%.

**Decision:** retain only the causal levels/value-area context as the
canonical no-order C1 source contract; see
[`C1_LVA_CANONICAL_INFERENCE_20260901.md`](C1_LVA_CANONICAL_INFERENCE_20260901.md).
It is not an exchange-writing activation.  Channels is rejected despite better
source metrics because it loses constrained MC1 contribution.  Balance,
OI-at-price, and the all-feature bundle are rejected and must not be combined
with the retained levels block without a new predeclared test.

### Channel-family additions to levels/value-area — rejected

The channel bundle was then decomposed *on top of the retained
levels/value-area challenger*, not tested as a standalone replacement.  The
three trials respectively added Bollinger (z-score, width, %B), Keltner
(z-score, width), or Donchian (position, width, upper/lower distance) state
to the same frozen source-head contract.  All improve accepted-break
prediction and reaction-magnitude rank versus levels alone; Keltner has the
largest source lift.  However, each weakens conditional-strength rank and all
three fail the economically decisive downstream test.

| Context, atop levels/value-area | June--July trades | June--July contribution | Aug. 1--18 trades | Aug. contribution | Jun.--Aug. contribution Δ vs levels |
|---|---:|---:|---:|---:|---:|
| Levels/value-area only | 465 | +91,947.72 bps | 98 | +27,205.70 bps | -- |
| + Bollinger | 465 | +86,400.78 bps | 98 | +17,676.58 bps | **-15,076.06 bps** |
| + Keltner | 460 | +88,160.31 bps | 98 | +19,844.37 bps | **-11,148.73 bps** |
| + Donchian | 465 | +89,248.44 bps | 93 | +21,270.12 bps | **-8,634.86 bps** |

This is consistently worse rather than a trade-count artifact: joined
June--August EV/trade falls from +211.64 bps for levels alone to +184.86
(Bollinger), +193.56 (Keltner), and +198.06 (Donchian).  Worst-week and
drawdown measures also worsen in the tested windows.  Therefore no
Bollinger/Keltner/Donchian component is retained for this challenger.

Single-field slicing is intentionally not pursued after these three fully
independent families lose in every held window: selecting individual fields
after this result would be post-hoc feature mining.  Any future field-level
trial must be predeclared and evaluated on a later untouched period.

### Levels/value-area backward ablation — retain all seven fields

The retained levels/value-area block was backward-ablated as four semantic
removals, with the frozen source fit, MC1 architecture, dual +50-bps
admission, BCF priority, policy labels, and controlled auction fixed.  The
comparison covers June, July, and Aug. 1--18 2026.  A removal would be kept
only if it improved joined constrained contribution without weakening the
monthly stability profile.

| Context | Fields excluded | Trades | Net EV/trade | Net contribution | Δ vs full levels |
|---|---|---:|---:|---:|---:|
| **Full levels/value area** | -- | **563** | **+211.64 bps** | **+119,153.42 bps** | -- |
| No POC | `profile_poc_distance_atr` | 570 | +197.79 | +112,742.05 bps | -6,411.36 |
| No VAH/VAL | VAH and VAL distances | 554 | +198.43 | +109,930.17 bps | -9,223.24 |
| No HVN/LVN | HVN and LVN distances | 562 | +198.41 | +111,504.00 bps | -7,649.42 |
| No value-area geometry | inside-value-area and value-area width | 564 | +194.08 | +109,461.57 bps | -9,691.84 |

There are isolated monthly gains that do not survive confirmation: removing
POC adds +1,770.58 bps in June but loses -8,055.31 bps in August; removing
HVN/LVN improves July by +1,437.69 bps but loses -8,473.75 bps in August.
Every deletion therefore fails the joined economic/stability rule.

**Contract retained unchanged:** `profile_poc_distance_atr`,
`profile_vah_distance_atr`, `profile_val_distance_atr`,
`profile_hvn_distance_atr`, `profile_lvn_distance_atr`,
`profile_inside_value_area`, and `profile_value_area_width_atr`.  This
remains a research-only challenger, not a live or canonical alteration.

## Relevant artifacts

### Current C1-LVA receipt

- `data_perp/artifacts/canonical_sr_profile_levels_c1_mc1_junjul_20260831_v2`
- `data_perp/artifacts/canonical_sr_profile_levels_c1_mc1_august_20260831_v1`
- `docs/CAUSAL_PROFILE_VALUE_AREA_GROUP_ABLATION_20260831.md`

### Supporting and rejected research receipts

- `data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_20260831_v5`
- `data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_august_20260831_v3`
- `data_perp/artifacts/causal_sr_selective_router_20260831_v2`
- `scripts/ablate_causal_sr_selective_router.py`
- `data_perp/artifacts/canonical_sr_c1_demotion_component_20260831_v2`
- `data_perp/artifacts/canonical_sr_c1_support_component_20260831_v1`
- `data_perp/artifacts/canonical_sr_c1_support_component_august_20260831_v2`
- `data_perp/artifacts/causal_oi_positioning_2025_train_2026_score_20260831_v1`
- `data_perp/artifacts/causal_oi_positioning_heads_oof_20260831_v1`
- `data_perp/artifacts/canonical_sr_oi_positioning_mc1_20260831_v1`
- `scripts/run_causal_sr_ontology_ablation.py`
- `scripts/materialize_causal_oi_positioning.py`
- `scripts/run_causal_oi_positioning_heads.py`
- `data_perp/artifacts/causal_sr_ontology_ablation_20260831_v1` (narrow levels, rejected)
- `data_perp/artifacts/causal_sr_ontology_ablation_20260831_v2` (independent retests, rejected)
- `data_perp/artifacts/causal_sr_ontology_ablation_20260831_v4` (barrier grid/no speed, rejected)
- `extreme_price_movements/causal_profile_geometry.py`
- `scripts/materialize_causal_profile_geometry.py`
- `scripts/run_causal_profile_geometry_heads.py` (direct-profile negative diagnostic)
- `data_perp/artifacts/causal_profile_geometry_2025_train_2026_score_20260831_v2`
- `data_perp/artifacts/causal_sr_heads_2025train_2026allscore_20260831_v2`
- `data_perp/artifacts/causal_sr_heads_profile_context_2025train_2026allscore_20260831_v2`
- `data_perp/artifacts/causal_sr_heads_profile_levels_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_balance_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_oi_at_price_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_channels_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/canonical_sr_frozen2025_c1_mc1_junjul_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_context_c1_mc1_junjul_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_c1_mc1_junjul_20260831_v2`
- `data_perp/artifacts/canonical_sr_profile_levels_c1_mc1_august_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_channels_c1_mc1_junjul_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_channels_c1_mc1_august_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_levels_bollinger_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_levels_keltner_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_levels_donchian_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_bollinger_c1_mc1_junjul_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_bollinger_c1_mc1_august_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_keltner_c1_mc1_junjul_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_keltner_c1_mc1_august_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_donchian_c1_mc1_junjul_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_donchian_c1_mc1_august_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_levels_without_poc_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_levels_without_vah_val_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_levels_without_hvn_lvn_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_levels_without_value_area_geometry_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_without_poc_c1_mc1_junjul_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_without_poc_c1_mc1_august_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_without_vah_val_c1_mc1_junjul_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_without_vah_val_c1_mc1_august_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_without_hvn_lvn_c1_mc1_junjul_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_without_hvn_lvn_c1_mc1_august_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_without_value_area_geometry_c1_mc1_junjul_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_levels_without_value_area_geometry_c1_mc1_august_20260831_v1`
- `data_perp/artifacts/canonical_sr_frozen2025_c1_mc1_august_20260831_v1`
- `data_perp/artifacts/canonical_sr_profile_context_c1_mc1_august_20260831_v1`
- `data_perp/artifacts/causal_profile_geometry_2025_train_2026_score_20260831_v3_level_conditioned_context`
- `data_perp/artifacts/causal_sr_heads_profile_levels_contextv3_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_levels_without_interaction_contextv3_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_levels_volatility_participation_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/causal_sr_heads_profile_levels_anchored_vwap_2025train_2026allscore_20260831_v1` (rejected at source gate)
- `data_perp/artifacts/causal_sr_heads_profile_levels_volatility_participation_anchored_vwap_2025train_2026allscore_20260831_v1`
- `data_perp/artifacts/canonical_sr_levels_contextv3_control_mc1_junjul_20260831_v1/finalization_receipt.json`
- `data_perp/artifacts/canonical_sr_levels_without_interaction_contextv3_mc1_junjul_20260831_v1/finalization_receipt.json`
- `data_perp/artifacts/canonical_sr_levels_volatility_participation_mc1_junjul_20260831_v1/finalization_receipt.json`
- `data_perp/artifacts/canonical_sr_levels_volatility_participation_anchored_vwap_mc1_junjul_20260831_v1/finalization_receipt.json`
