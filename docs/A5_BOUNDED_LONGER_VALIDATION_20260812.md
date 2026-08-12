# A5 bounded correction — longer-period validation

## Decision

> **Superseded decision note — 2026-08-12.** The analysis below originally
> recommended retaining `F0_A0_top15` and keeping A5 as a challenger. A later
> explicit decision promotes `F2_blend_a10_fixed_A0_top15` instead. The
> measurements in this report are unchanged; the active contract and sealed
> inference lineage are defined in
> `docs/TP6_SL4_BASE_CONSENSUS_CANONICAL_20260807.md`.

The April-2025 through July-2026 replay is causal and uses one frozen
October-December-2024 geometry/K9 bundle.  April-September 2025 are a labelled
expanding-history robustness block (3-8 months of post-geometry training).
October-2025 onward is the matched full nine-month contract.

## Contract

```text
admission = A0 admitted AND timestamp-local top 15%
score_20 = A0 expected policy net bps
         + 0.20 * (causally calibrated A5 expected bps - A0 expected bps)
```

A5 calibration for month `t` uses only earlier OOS predictions whose policy
labels resolved before month `t`.  The frozen SimplePolicyOptimiser outcome is
used: 12-hour timeout, 100-bps cost once, stop 4.1520006 ATR, trailing
activation 2.3262249 ATR, giveback 0.1023720 ATR.  Portfolio constraints are
long-only, 7x leverage, eight concurrent positions, two entries per 15-minute
bar, one position per asset, 10% margin slots, and 80% total margin.

## Raw admitted-pool ranking

| Arm | Top 1% | Top 2% | Top 5% | All admitted |
|---|---:|---:|---:|---:|
| A0 top-15 | +200.95 | +202.93 | +180.19 | +121.05 |
| A5 blend 10% | +190.97 | +154.34 | +160.41 | +121.05 |
| A5 blend 15% | +202.14 | +160.39 | +147.44 | +121.05 |
| A5 blend 20% | +204.04 | +163.05 | +138.47 | +121.05 |
| A5 blend 25% | +215.09 | +164.49 | +123.73 | +121.05 |

The correction improves only the extreme Top-1 tail. It materially damages
Top-2 and Top-5 ranking, so it is not a generally better score.

## Portfolio comparison

| Arm | Trades | Trades/day | Net bps/trade | Sortino | Max DD | Worst week |
|---|---:|---:|---:|---:|---:|---:|
| A0 top-15 | 6,651 | 13.66 | +150.57 | 0.469 | -48.52% | -31.51% |
| A5 blend 10% | 6,652 | 13.66 | +152.35 | 0.471 | -48.53% | -31.51% |
| A5 blend 15% | 6,649 | 13.65 | +152.49 | 0.462 | -49.40% | -31.51% |
| A5 blend 20% | 6,641 | 13.64 | +153.31 | 0.463 | -49.40% | -31.51% |
| A5 blend 25% | 6,645 | 13.64 | +152.72 | 0.464 | -49.40% | -31.51% |
| A5 20% demotion-only | 6,643 | 13.64 | +152.47 | 0.463 | -49.40% | -31.51% |
| A5 20% capped +/-50 bps | 6,646 | 13.65 | +152.54 | 0.462 | -49.40% | -31.51% |

The 20% symmetric blend adds +2.74 bps/trade overall. A paired calendar-day
bootstrap gives a 95% interval of approximately [-0.05, +5.52] bps and
P(uplift > 0) = 97.3%; this is borderline rather than decisive because zero is
inside the interval. It raises the expanding-history block by +5.23 bps/trade,
but the exact nine-month confirmation block by only +0.98 bps/trade
(95% interval [-1.68, +3.69]).

The 20% arm is positive in all 16 months and improves 12/16 monthly means, but
its Sortino and drawdown are worse than A0. Across 68 active weeks, 67 are
positive; 37 improve on A0, 50 are non-negative uplift, and worst uplift is
-44.19 bps/trade. Both A0 and A5 retain the same two zero-trade weeks,
2026-W24 and 2026-W25. A5 creates no additional drought.

## Interpretation

The longer chronology confirms that A5 carries weak complementary information,
especially for the extreme tail and portfolio tie/order decisions. It does not
support replacing A0's ranking: the apparent constrained uplift comes from a
small number of auction substitutions, while raw Top-2/5 ordering, Sortino, and
drawdown deteriorate. Conservative capping and demotion-only use do not repair
that trade-off.

## Artifacts

- `scripts/assemble_a5_longer_validation.py`
- `data_perp/artifacts/strict_r3_a5_longer_prequential_long_2025apr_2026jul_20260812_v1`
- `data_perp/artifacts/strict_r3_a5_bounded_integration_long_2025apr_2026jul_20260812_v1`
- `data_perp/artifacts/strict_r3_a5_bounded_integration_summary_long_2025apr_2026jul_20260812_v1`
- `data_perp/artifacts/strict_r3_a5_bounded_long_portfolio_<arm>_2025apr_2026jul_20260812_v1`

Correctness tests: 11 passed across bounded integration, prequential
calibration, and Cell-day residual walk-forward modules.

## Matched comparison with ungated R5 posterior

Here `R5 posterior` means its own `expected policy net >= +50 bps` admission.
`A0 top-15` adds the timestamp-local top-15 domain gate to that same posterior.

| Arm | Full-period trades | Trades/day | Full net bps/trade | Exact-9m net bps/trade | Worst active week | Worst month |
|---|---:|---:|---:|---:|---:|---:|
| R5 posterior | 6,704 | 13.77 | +149.84 | +161.08 | 2026-W30: -271.40 | 2025-12: +102.96 |
| A0 top-15 | 6,651 | 13.66 | +150.57 | +162.41 | 2026-W30: -271.40 | 2025-12: +106.15 |
| A5 blend 10% | 6,652 | 13.66 | +152.35 | +162.31 | 2026-W30: -271.40 | 2025-12: +104.13 |
| A5 blend 20% | 6,641 | 13.64 | +153.31 | +163.39 | 2026-W30: -271.40 | 2025-12: +104.57 |

All four arms have the same two-week drought: 2026-W24 and 2026-W25. The
following two weeks contain only three and two trades respectively. The common
worst active week, 2026-W30, has 15 trades at -271.40 net bps/trade and a
-30.49% wallet return. This unchanged failure shows that neither the top-15
domain gate nor bounded A5 score correction addresses the July loss cluster.

Across the exact nine-month October-2025 through July-2026 block, A0 improves
on ungated R5 by +1.32 bps/trade, and the 20% A5 score adds another +0.98
bps/trade. All monthly mean EVs remain positive; the worst is December 2025.

## Compounded and relative-stability diagnostics

Fresh-start October-2025 through July-2026 portfolios produce the following
theoretical, fully reinvested results from 1,000 initial wallet units. These are
not capacity-adjusted forecasts; repeated 7x reinvestment makes the absolute
wallet values explode.

| Arm | Final wallet | Sortino | Colmar | Return profit factor | Ulcer | Pain index | Pain-to-gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| R5 posterior | 1.033e20 | 0.490 | 6.42e20 | 2.735 | 7.84% | 2.80% | 0.576 |
| A0 top-15 | 9.559e19 | 0.492 | 5.85e20 | 2.765 | 7.57% | 2.53% | 0.567 |
| A5 top-15 blend 10% | 8.503e19 | 0.487 | 5.08e20 | 2.766 | 7.57% | 2.53% | 0.566 |
| A5 top-15 blend 20% | 1.006e20 | **0.493** | 6.22e20 | **2.778** | **7.56%** | 2.56% | **0.563** |

Colmar is numerically huge because its CAGR numerator inherits the unrealistic
unbounded compounding assumption. Sortino, profit factor, Ulcer, Pain Index,
and the distributional stability statistics are more interpretable here.
Profit factor is computed on normalized per-trade returns; Pain-to-gain is
absolute negative-return mass divided by net return mass, so lower is better.

Across the exact nine-month block, all arms have 10/10 positive months and
41/42 positive active weeks; the one negative active week is 2026-W30. There
are 44 calendar weeks, of which two are zero-trade drought weeks.

| Arm | Median month EV | Worst month | Month SD | Month IQR | Median week EV | Week SD | Weekly trade-count CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| R5 posterior | +143.99 | +102.96 | 45.14 | 75.20 | +176.08 | 126.37 | 0.478 |
| A0 top-15 | +144.84 | +106.15 | 43.08 | 71.57 | +176.08 | 125.55 | 0.479 |
| A5 top-15 blend 10% | +143.47 | +104.13 | 42.00 | 65.67 | +175.92 | 125.78 | 0.479 |
| A5 top-15 blend 20% | **+146.83** | +104.57 | **41.69** | **60.66** | +175.20 | 125.89 | **0.477** |

Relative to R5, the 20% blend improves 8/10 monthly means and 18/42 active
weekly means. Its median monthly uplift is +1.66 bps and its worst monthly
uplift is -5.46 bps. Weekly uplift is much less stable: median zero, worst
-23.36 bps, and weekly uplift standard deviation 9.60 bps. This supports weak
monthly complementarity, not a consistently superior weekly controller.
