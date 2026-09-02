# Enhanced-base live-stack challenger — matched audit

## Scope

Offline, long-only research.  No live configuration, exchange action, or
production artifact was changed.  The challenger replaces only the upstream
base score with the strict-OOS 50/50 direct-efficiency/timing score; every
downstream component is refit from that score:

1. 10 canonical residual LambdaRank heads;
2. 75/25 base/consensus blend;
3. residual-correctness demotion;
4. separate current and BCF-like prequential MC1 maps;
5. dual MC1 admission at +30 bps and BCF-like mapped-EV auction priority;
6. canonical rich policy labels and the normal constrained portfolio replay.

The comparison delta is only on exact common candidate IDs.  The broader
enhanced route is retained separately as coverage evidence, not as a claimed
uplift over the control.

## Matched constrained portfolio result

| Period | Arm | Accepted | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 Q4 | live-like control | 2,071 | +140.65 | +291,295 | +98.20 | +30.93 | -29.72% |
| 2025 Q4 | enhanced matched | 1,984 | +145.54 | +288,742 | +100.40 | +45.87 | -27.65% |
| 2026 Apr-Jul | live-like control | 2,366 | +139.50 | +330,048 | +113.66 | +88.94 | -21.61% |
| 2026 Apr-Jul | enhanced matched | 2,756 | +124.34 | +342,680 | +92.03 | +75.42 | -25.04% |

The challenger improves 2025 Q4 precision by +4.88 bps/trade, but is lower by
-15.16 bps/trade in 2026 Apr-Jul.  Across both matched periods it adds 303
accepted entries and +10,078 total bps, while reducing mean EV/trade by 6.83
bps.  It is therefore a challenger, not a promotion candidate.

## Consensus-head strict-OOS audit

Population: 348,330 valid, enhanced-base-routed rows from October 2025 through
July 2026.  Every fold is scored before policy outcomes are joined.  `Top-x%`
is a global diagnostic using the individual head's own OOS rank; it is not an
admission or portfolio result.  LOO is the change in full-median-consensus
top-5 EV after excluding the named head: positive means that head helps.

| Head | IC | Top-1 | Top-2 | Top-5 | Positive top-5 months | Mean pairwise corr. | LOO Δ top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cap100 ordinary | 0.029 | +333.82 | +226.62 | +104.97 | 8/10 | 0.796 | +7.02 |
| cap80 ordinary | 0.023 | +307.55 | +213.55 | +100.31 | 9/10 | 0.806 | +4.15 |
| cap120 equal-month | 0.053 | +298.34 | +184.19 | +80.01 | 9/10 | 0.757 | +3.02 |
| cap40 equal-month | 0.013 | +334.53 | +204.66 | +78.60 | 8/10 | 0.814 | +0.63 |
| cap60 equal-month | 0.011 | +283.37 | +185.12 | +75.49 | 9/10 | 0.815 | +0.98 |
| cap40 ordinary | 0.014 | +262.76 | +166.67 | +64.91 | 8/10 | 0.818 | -0.05 |
| cap120 ordinary | 0.030 | +208.70 | +140.19 | +62.44 | 8/10 | 0.742 | -1.16 |
| cap80 equal-month | -0.002 | +208.17 | +126.55 | +54.52 | 6/10 | 0.757 | -0.37 |
| cap100 equal-month | -0.004 | +101.73 | +58.48 | +13.20 | 3/10 | 0.719 | -2.30 |
| cap60 ordinary | -0.090 | -89.36 | -81.12 | -77.03 | 0/10 | 0.627 | -10.15 |

`cap100_ordinary`, `cap80_ordinary`, and `cap120_equal_month` are the useful,
non-identical contributors.  `cap60_ordinary` is consistently harmful on this
diagnostic; `cap100_equal_month` and `cap80_equal_month` are also weak.
These findings identify focused removal/downweighting ablations; they do not
license editing the frozen live consensus contract without a fresh downstream
MC1-and-portfolio comparison.

## Research successor contract

Subsequent enhanced-base research uses only the five heads with positive,
stable standalone economics and non-negative leave-one-out contribution:

`cap100_ordinary`, `cap80_ordinary`, `cap120_equal_month`,
`cap40_equal_month`, and `cap60_equal_month`.

The successor selector is
`config/strict_r3_enhanced_base_consensus_top5_v1.json`.  It hash-binds the
unchanged frozen ten-head parent and selects only these names.  It is scoped to
the enhanced-base research runner and audit; it does not alter the live
ten-head contract.

## Aggregate-layer diagnostic

| Layer | Rank IC to policy net | Top-1 | Top-2 | Top-5 |
|---|---:|---:|---:|---:|
| Median ten-head consensus | 0.014 | +330.41 | +200.98 | +74.73 |
| Ordinary-head shadow consensus | 0.014 | +305.94 | +187.43 | +72.60 |
| 75/25 base + consensus upstream | 0.186 | +415.01 | +350.80 | +235.36 |
| Residual-correctness demotion | 0.007 | +274.38 | +187.08 | +85.28 |
| Current pre-MC1 final score | 0.036 | +391.30 | +304.22 | +155.29 |

The enhanced base/upstream remains the main ranking signal.  The ten-head
median is a modest correction layer, not a substitute for it; this explains
why a large base-only uplift need not translate into a proportional
portfolio-level uplift after consensus and dual MC1 admission.

## Receipts

- `data_perp/artifacts/strict_r3_enhanced_base_live_stack_challenger_20260823_v10/`
- `data_perp/artifacts/strict_r3_enhanced_base_consensus_head_audit_20260823_v2/`
- `scripts/run_strict_r3_enhanced_base_live_stack_challenger.py`
- `scripts/audit_strict_r3_enhanced_base_consensus_heads.py`
