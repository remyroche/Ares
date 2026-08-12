# N5 longer-period audit on canonical Cell-day admission

Date: 2026-08-12  
Side: long only  
Decision: N5 does not advance beyond research challenger

## Why the earlier uplift was too large

Two problems inflated the original headline. First, tail percentages were
computed independently inside each arm's admitted population. When N5
demoted candidates below +50 bps, its Top-5 could contain fewer rows than the
control's Top-5. Second, that experiment used the reserve-seeded 21-day map,
not the canonical exact-producer Cell-day trim-15% map.

The evaluator now persists `matched_cardinality_audit.parquet` with two fair
comparisons: challenger-count matched, and fixed canonical population. The
portfolio replay now accepts explicit expected-EV/admission columns and
multiple immutable score/selection partitions.

## Correct contract

- N5: `N5_ldf_support_l110_meanrisk`, 64-tree Local Distribution Forest
  Proxy, depth 8, minimum leaf 120, 70% features, 75% bootstrap sample,
  local-support shrinkage with prior strength 300.
- Target: frozen SimplePolicyOptimiser policy net bps.
- Training: strict chronological, labels resolved before cutoff, timestamp
  Top-30% training domain, equal-month cap 60,000.
- Inputs: 66 causal fields, including ten latest-conversion-fit support/OOD/
  covariance fields; raw K9 memberships and Cell-day map columns are excluded.
- EV basis: exact-producer Cell-day trim 15%, common bps, +50-bps admission.
- Overlay: 25% demotion only:
  `cell_day_ev + .25 * min(n5_posterior_ev - cell_day_ev, 0)`.
- Portfolio: long only; two entries per 15-minute bar; eight concurrent; one
  position per asset; 80% margin cap; 7x leverage; 10% wallet margin slots;
  same-producer final score only breaks exact EV ties.
- Exit outcome: the frozen pre-2025 SimplePolicyOptimiser winner, SL
  4.152000643 ATR, trailing activation 2.326224920 ATR, giveback
  0.102371990 ATR, H12 timeout, and 100 bps cost once.

## Continuous constrained portfolio results

| Period | Arm | Trades | Trades/day | Net bps/trade | Sum net bps | Positive | Max DD | Sortino | Worst week |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 Feb-Jul | Cell-day control | 3,340 | 18.45 | +149.93 | 500,766 | 65.45% | -75.15% | 0.351 | -22.64% |
| 2025 Feb-Jul | N5 demotion | 3,310 | 18.29 | +154.92 | 512,773 | 65.11% | -79.44% | 0.352 | -5.44% |
| 2026 May-Jul | Cell-day control | 697 | 7.58 | +67.41 | 46,988 | 50.50% | -54.45% | 0.508 | -22.50% |
| 2026 May-Jul | N5 demotion | 648 | 7.04 | +74.97 | 48,583 | 52.16% | -45.88% | 0.567 | -14.93% |

Across both windows, control is 4,037 trades at +135.68 bps/trade. N5 is
3,958 trades at +141.83 bps/trade: a trade-weighted +6.15-bps uplift and
+13,603 summed net bps. This is useful but far below the original headline.

## Monthly portfolio result

| Month | Control trades | Control net | N5 trades | N5 net | N5 uplift |
|---|---:|---:|---:|---:|---:|
| 2025-02 | 688 | +168.76 | 684 | +167.36 | -1.40 |
| 2025-03 | 613 | +138.49 | 608 | +143.62 | +5.14 |
| 2025-04 | 428 | +213.64 | 428 | +211.88 | -1.76 |
| 2025-05 | 567 | +136.67 | 563 | +136.09 | -0.58 |
| 2025-06 | 509 | +92.93 | 504 | +105.08 | +12.15 |
| 2025-07 | 535 | +156.14 | 523 | +173.44 | +17.30 |
| 2026-05 | 627 | +77.33 | 629 | +80.34 | +3.01 |
| 2026-06 | 69 | -26.40 | 19 | -102.69 | -76.29 |
| 2026-07 | 1 | +321.38 | 0 | n/a | n/a |

The June drought is fixed by Cell-day admission, but N5 then removes most
June trades and retains a much worse subset. It also rejects the sole July
admission. This is a material portability failure.

## Stability and uncertainty

In 2025, N5 improves per-trade weekly EV in 12 of 27 weeks. Median weekly
uplift is 0.0 bps and mean uplift is +5.15 bps; the paired week bootstrap 95%
interval is `[-1.97, +13.47]`, crossing zero. Thirteen of 27 weeks improve
summed net bps.

In May-July 2026 only six weeks have trades in both arms. Three improve.
Median weekly uplift is -1.75 bps and the paired week-bootstrap interval is
`[-40.62, +8.89]`. Aggregate improvement is driven by May and by avoiding
some exposure, while June selection quality is worse.

Top-tail diagnostics remain secondary. Cell-day has only 20 monotone score
cells; N5 greatly changes within-cell ordering, producing large fixed-
population Top-2/5 uplifts. Those tail numbers are not reliable evidence of
a comparable executable gain because the portfolio auction, concurrency and
asset constraints compress the effect substantially.

## Decision

1. Keep Cell-day trim 15% as the canonical EV-map/admission basis.
2. Do not make N5's 25% demotion canonical.
3. Keep N5 in shadow as an ordering/risk diagnostic.
4. Before another promotion attempt, constrain N5 authority by causal support
   and explicitly require no degradation in a shock month such as June.
5. Select future trust overlays on constrained portfolio delta, worst-month
   delta and weekly paired uncertainty—not unconstrained tail EV alone.

## Code and artifacts

- N5 fold runner: `scripts/run_strict_r3_current_exact_b5_fold.py`
- Fair admission evaluator: `scripts/evaluate_strict_r3_trust_posterior_admission.py`
- Portfolio runner: `scripts/replay_strict_r3_tail_health_portfolio.py`
- 2025 portfolios:
  `strict_r3_cellday15_latestfit_N5_portfolio_{control,demotion_a25}_long_2025_febjul_20260812_v3`
- 2026 portfolios:
  `strict_r3_cellday15_latestfit_N5_portfolio_{control,demotion_a25}_long_2026_mayjul_20260812_v2`
- Fold audits:
  `strict_r3_cellday15_latestfit_N5_audit_long_*_20260812_v2`
- Tests: 26 focused tests passed, including a regression guard that the
  Cell-day map fields cannot enter N5's model feature set.
