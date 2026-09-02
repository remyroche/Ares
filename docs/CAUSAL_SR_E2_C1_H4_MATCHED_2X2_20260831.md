# Matched E2/C1 × H4 Continuation Replay — 2026-08-31

## Status

Research-only causal falsification. This document **does not promote** either
continuation controller or change the live/canonical exchange-writing stack.

It reconciles the earlier P8U E2/H4 evidence with the later C1 H4 evidence by
holding the execution policy and portfolio auction fixed and varying only:

1. the target-free entry population; and
2. the continuation-controller contract.

## Fixed contract

Every arm uses the same frozen rich parent policy:

- long-only, exact one-minute path replay;
- entry at decision + five minutes;
- one 100-bps policy cost, applied exactly once;
- the normal chronological constrained portfolio auction and BCF priority;
- H4 action only after a completed 15-minute state and only for the next
  interval.

Routes are persisted before exact paths and policy outcomes are joined only
after this target-free routing step. The parent-policy outcome engine asserts
exact net-bps and exit-timestamp parity for every valid exact path.

## Population / source audit

| Item | E2 original selection | C1 dual-40 |
|---|---:|---:|
| Target-free routed candidates | 1,963 | 1,564 |
| Exact one-minute valid paths | 1,028 | 1,118 |
| Parent-policy portfolio entries | 788 | 487 |
| Cross-population routed-ID overlap | \- | 38 |

The union requested 3,489 target-free paths. Of these, 2,120 had a complete
exact one-minute path and 1,369 failed only *after* routing because their
execution outcome path was unavailable. Invalid paths are excluded from every
compared arm; none is filled or imputed.

This is materially different from the historical E2 continuation study, which
used the older 15-minute source-valid policy panel. That panel produced 1,448
parent and 1,530 H4 portfolio entries, whereas the exact one-minute matched E2
population here has 788 parent entries. Its outcomes and capacity path are not
identical to an exact-minute live replay.

## Controller contracts

| Controller | Target / decision authority | Important behaviour |
|---|---|---|
| Parent | Frozen rich policy | No H4 intervention |
| Archived L1 | `activation50_advantage_bps`; L1 mean | One-interval, activation-only change. This reproduces the old H4 semantics under strict prior-resolved refits. |
| Repaired L2 | `latched_activation50_giveback20`; L2 regression | MFE-ready, permanent latch; earlier activation plus 20% tighter giveback. |

The legacy L1 controller is much more active in this exact replay: it enables
44.2% of evaluated E2 states and 55.5% of C1 states. The repaired L2 latch is
eligible/enabled on 14.5% and 7.1%, respectively. That difference is intended
by the newer safety-focused target and must not be confused with a model
improvement.

## Matched exact-one-minute results — June through August 2026

| Population | Controller | Entries | Net EV/trade | Total net bps | Δ EV/trade vs own parent | Δ total bps | Sortino | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E2 original | Parent | 788 | -7.76 | -6,115.61 | \- | \- | -0.031 | -78.64% |
| E2 original | Archived L1 | 801 | +0.27 | +217.78 | **+8.03** | **+6,333.38** | -0.005 | -70.00% |
| E2 original | Repaired L2 latch | 800 | -8.19 | -6,554.04 | -0.43 | -438.43 | -0.032 | -77.81% |
| C1 dual-40 | Parent | 487 | +126.88 | +61,789.40 | \- | \- | 0.658 | -10.82% |
| C1 dual-40 | Archived L1 | 487 | **+128.49** | **+62,575.28** | **+1.61** | **+785.88** | **0.689** | -10.81% |
| C1 dual-40 | Repaired L2 latch | 487 | +126.76 | +61,733.05 | -0.12 | -56.35 | 0.668 | -10.81% |

### Monthly exact-one-minute EV per trade

| Population / controller | June | July | August |
|---|---:|---:|---:|
| E2 parent | +23.80 | -53.46 | +1.50 |
| E2 archived L1 | +33.73 | -40.75 | +5.50 |
| E2 repaired L2 | +22.39 | -51.98 | +0.35 |
| C1 parent | +191.72 | +116.94 | +70.62 |
| C1 archived L1 | +192.27 | +116.70 | +76.54 |
| C1 repaired L2 | +192.69 | +116.80 | +69.33 |

## What explains the earlier good H4 result?

The prior +23.21 bps/trade H4 activation result (and +25.31 bps/trade for
H4 + Giveback-20) was real **within its older E2/15-minute source-valid
replay**, but it is not an exact-minute live-equivalent estimate. The matched
test isolates four reasons its headline was larger:

1. **Execution substrate changed.** The old result used the 15-minute policy
   source panel; the new result evaluates the frozen exact one-minute path with
   a decision + five-minute entry. The old panel necessarily realizes exits on
   15-minute boundaries, whereas the exact engine can trigger between them.
   These outcomes are not interchangeable.
2. **The evaluated E2 population was much larger.** The old source-valid study
   accepted 1,448 parent / 1,530 H4 positions. The exact-path matched E2 study
   can evaluate 788 / 801. The missing exact paths are excluded consistently,
   but the surviving cohort is economically different.
3. **Part of the old gain was capacity recycling.** On E2, L1 changes 175
   candidate exits and increases accepted entries from 788 to 801 in the exact
   replay. The historical proxy study had an even larger entry-count change
   (1,448 to 1,530). H4 was therefore improving both individual exit timing
   and portfolio capacity, not just a fixed set of exits.
4. **The controller semantics changed.** Archived L1 is a broad, repeatedly
   active one-interval activation model. The repaired L2 contract is an
   MFE-ready, conservative latched controller. It acts much less frequently;
   it should be judged as a different strategy, not a repair that must preserve
   the old L1 uplift.

Under the exact common policy, the archived controller retains a modest,
directionally positive effect (+8.03 bps/trade on E2 and +1.61 on C1). The
repaired L2 controller has no measured portfolio benefit in either population.
Thus the old +23/+25-bps headline cannot be carried forward as production
evidence for the new C1/L2 design.

## Causal / integrity evidence

- Target-free source selections are immutable and hashed before outcome access.
- H4 training rows are restricted to prior-resolved labels for each held month.
- The same exact one-minute arrays are reused after a candidate-ID, timestamp,
  symbol, side, and array-identity audit.
- The valid and invalid exact-path sets partition the target-free request set;
  all outcomes are joined downstream.
- Each portfolio arm uses an identical auction implementation for its own fixed
  routed population.
- No exchange calls, model-promotion action, or live artifact mutation occurs.

## Artifacts

- [Run manifest](/Users/remyroche/Documents/Ares/data_perp/artifacts/causal_sr_e2_c1_h4_matched_2x2_20260831_v4/run_manifest.json)
- [Portfolio summary](/Users/remyroche/Documents/Ares/data_perp/artifacts/causal_sr_e2_c1_h4_matched_2x2_20260831_v4/portfolio_summary.parquet)
- [Exit-change attribution](/Users/remyroche/Documents/Ares/data_perp/artifacts/causal_sr_e2_c1_h4_matched_2x2_20260831_v4/exit_change_attribution.parquet)
- [Strict prequential support](/Users/remyroche/Documents/Ares/data_perp/artifacts/causal_sr_e2_c1_h4_matched_2x2_20260831_v4/strict_prequential_training_support.parquet)
- [Target-free population routes](/Users/remyroche/Documents/Ares/data_perp/artifacts/causal_sr_e2_c1_h4_matched_2x2_20260831_v4/target_free_population_routes.parquet)

The producer is [run_causal_sr_e2_c1_h4_matched_2x2.py](/Users/remyroche/Documents/Ares/scripts/run_causal_sr_e2_c1_h4_matched_2x2.py).
