# Ten-head conditional-usefulness and K9 consensus integration

**Date:** 2026-08-10  
**Side:** long only  
**Status:** ranking challenger advances; executable incumbent is not replaced  
**Development:** January–July 2025  
**Frozen transport:** January–July 2026

## 1. Question and matched architecture

This ablation tests whether the frozen head architecture selected in
`TEN_HEAD_CONDITIONAL_USEFULNESS_FUNNEL_20260810.md` improves the current
D2-base stack when retrained on the current optimized-policy residual, and
whether fixed K9 state fields add value inside those consensus residual heads.

Every candidate uses the same strict-prequential D2 R3 base, causal train-only
policy-net map, optimized SimplePolicyOptimiser outcome, 75/25 base-consensus
blend, correctness-only downstream layer, prior-42-day CDF, hierarchical
21/42/84-day admission, and constrained long-only portfolio. Severe-200 is
outside this ablation and remains exact H12 TP6/SL4 net <= -200 bps.

The challenger imports the funnel's frozen per-head target, query, MDA subset,
weighting, and ranker parameters. Its target is the ordinal policy residual
with edges `[-150, -50, +50, +150]` bps. Six heads use exact timestamp × side
queries; the others use 4-hour × side. Accepted MDA subsets are 15 fields for
cap60 ordinary, 30 for cap60 equal-month, and 51 for cap120 equal-month; all
other heads retain their cap. No HPO challenger was promoted in the original
funnel, so all heads retain the frozen default ranker. Unlike its historical
stored scores, every head here is refit on the current optimized-policy
residual and current D2 base anchor.

## 2. K9 contract

K9 is fit once on the target-free point-in-time surface from 2024-10-01 through
2024-12-31: 265,139 rows before cap, 100,000 equal-month fit rows, nine
clusters, and no outcome input. Its fixed bundle hash is
`f0084c25e35170747464bff1e0aba83d68749e0c1270d712f6ddb134f478881b`.
It is never refit monthly, so cluster positions retain one meaning across all
consensus folds.

The screen compared no K9; entropy/Top-2-margin/OOD summaries; all nine soft
memberships; and the summaries plus three memberships selected by train-only
binned conditional MI given base-rank decile. Clusters 01, 02, and 05 were
each selected in six of seven folds. Held outcomes never entered selection.

## 3. K9 development screen

This used an 80,000-row fold cap. Values are optimized-policy net bps/trade
before downstream correctness.

| Consensus arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Top-2 portability | Worst month | Positive months |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Incumbent ordinary | +76.45 | +54.33 | +41.82 | +30.28 | +13.81 | -132.50 | -126.20 | 5/7 |
| **Conditional, no K9** | +92.87 | +66.90 | **+52.51** | **+24.39** | **+14.32** | **-3.51** | **-38.65** | **6/7** |
| Conditional + summaries | **+94.65** | +66.28 | +47.39 | +22.13 | +13.06 | -23.56 | -44.49 | 5/7 |
| Conditional + all memberships | +93.16 | **+68.96** | +46.42 | +21.48 | +13.34 | -22.90 | -50.83 | 5/7 |
| Conditional + CMI-3 | +82.34 | +62.32 | +42.52 | +21.69 | +13.45 | -18.55 | -46.04 | 5/7 |

K9 provides a narrow Top-0.5/1% gain in two variants, but every K9 arm loses
Top-2 EV, Top-5 EV, worst-month EV, and positive-month coverage versus the
same conditional heads without K9. No K9 consensus arm advances.

## 4. Full-cap conditional consensus

Only the screen winner was expanded to 240,000 training rows and frozen for
2026.

### Before downstream correctness

| Year | Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Worst Top-2 month | Positive months |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 | Incumbent ordinary | +76.45 | +54.33 | +41.82 | +30.28 | +13.81 | -126.20 | 5/7 |
| 2025 | Conditional | **+98.01** | **+76.73** | **+68.03** | **+46.33** | **+23.13** | **-42.53** | **6/7** |
| 2026 | Incumbent ordinary | +105.86 | +80.47 | +55.57 | +18.39 | -6.84 | +7.36 | 7/7 |
| 2026 | Conditional | **+125.74** | **+92.42** | **+65.12** | **+28.34** | **+0.77** | **+10.32** | 7/7 |

The conditional contract improves every pooled upstream tail in both years.

## 5. Complete stack results

| Year | Stack | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Top-2 portability | Worst Top-2 month |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 | Incumbent ordinary consensus | +156.97 | +127.38 | +108.38 | +74.96 | +40.44 | +107.44 | +56.38 |
| 2025 | **Conditional consensus** | **+177.56** | **+147.35** | **+123.77** | **+88.02** | **+52.06** | **+118.18** | **+66.93** |
| 2026 | Incumbent ordinary consensus | **+171.19** | +129.38 | +88.99 | +39.84 | +6.59 | +76.98 | **+44.08** |
| 2026 | **Conditional consensus** | +168.65 | **+133.73** | **+93.72** | **+42.11** | **+11.37** | **+101.94** | +32.30 |

The only pooled-tail regression is 2026 Top 0.5% (-2.53 bps). The challenger
improves Top 1/2/5/10 and portability, with 7/7 positive Top-2 months in both
years. The exact/15-minute Top-2 subset also improves from +100.46 to +118.33
in 2025 and +67.57 to +71.45 in 2026, so the result is not created solely by
the hourly backfill.

## 6. Admission and portfolio

| Year | Stack | Trades | Trades/day | Net bps/trade | Positive rate | Max drawdown |
|---:|---|---:|---:|---:|---:|---:|
| 2025 | Incumbent | 3,128 | 14.75 | +142.51 | 64.39% | -92.67% |
| 2025 | Conditional | **3,336** | **15.74** | **+145.80** | **64.66%** | **-79.88%** |
| 2026 | Incumbent | 1,942 | 9.16 | **+137.82** | 58.19% | **-59.78%** |
| 2026 | Conditional | **2,185** | **10.31** | +135.16 | **59.77%** | -68.01% |

The challenger improves participation and hit rate in both years and improves
2025 portfolio EV/drawdown. In 2026 it loses 2.66 bps/trade and worsens
drawdown by 8.23 percentage points. That prevents automatic executable
promotion despite stronger ranking evidence.

## 7. Decision

1. The conditional-usefulness ten-head contract advances as the versioned
   ranking challenger.
2. The ordinary consensus remains the executable control pending a later
   frozen period and risk-layer comparison.
3. K9 should not be fed directly to the consensus residual heads. K9 remains
   useful as downstream aggregate geometry/OOD/leaf context and diagnostics;
   raw memberships remain out of live consensus and correctness inputs.
4. Severe-200 remains unchanged and separate.

## 8. Implementation and artifacts

- Runner: `scripts/run_ten_head_k9_consensus_ablation.py`
- Tests: `tests/test_ten_head_k9_consensus_ablation.py`
- 2025 K9 screen: `data_perp/artifacts/strict_r3_ten_head_k9_consensus_screen_long_2025_janjul_20260810_v1`
- 2025 full-cap upstream: `data_perp/artifacts/strict_r3_ten_head_conditional_fullcap_long_2025_janjul_20260810_v1`
- 2026 frozen upstream: `data_perp/artifacts/strict_r3_ten_head_conditional_fullcap_long_2026_janjul_20260810_v1`
- 2025 complete stack: `data_perp/artifacts/strict_r3_ten_head_conditional_correctness_fullstack_long_2025_janjul_20260810_v1`
- 2026 complete stack: `data_perp/artifacts/strict_r3_ten_head_conditional_correctness_fullstack_long_2026_janjul_20260810_v1`

The comparison is causal and matched, but not untouched: both years have now
influenced research decisions. Promotion requires a later frozen period.
