# Cell-day residual trust continuation — 28-day map

## Decision

The exact-producer Cell-day map now uses the preceding **28 calendar days** of
resolved policy outcomes.  The previous 42-day artifacts remain historical
controls only.

The selected canonical executable-research configuration is:

```text
strict-R3 exact-producer score
→ causal 28-day Cell-day trim-15% expected-net map
→ R5 Local Distribution Forest Proxy, trained on the preceding 9 months
→ posterior expected policy net >= +50 bps
→ portfolio auction and frozen SimplePolicyOptimiser outcome
```

It is the canonical executable-research arm by explicit decision on
2026-08-12, but it remains shadow-only and is not production-approved. The
3/6/9/12-month comparison and the choice of posterior admission used May--July
2026. A subsequent backward walk-forward confirmation over October 2025--April
2026 passed strongly, but promotion still needs a later untouched forward
period. Promotion here means selection of the frozen research/inference
contract, not authorization to place exchange orders.

The inference-ready August-1 model bundle is persisted at
`data_perp/artifacts/strict_r3_cell_day_residual_trust_bundle_long_20260801_28d_r5_9m_posterior_v2`.
Its manifest records a 2025-11-01 training start, nine calendar months,
1,049,296 eligible prior-resolved rows before selection, a 60,000-row
equal-month fit, the ordered 66-field model contract, four stable-CMI
interactions, posterior admission ownership, and fail-closed missing-posterior
semantics.

## What changed

- `scripts/ablate_strict_r3_cell_day_bayesian_ev_mapping.py` now has an explicit
  `--window-days` contract and defaults to 28.
- New 28-day maps were materialised for 2025 January--March, 2025 April--July,
  2025 August--2026 March, and 2026 January--July.
- The missing August 2025--March 2026 latest-model-fit, 66-field trust substrate
  was materialised with nine strict 28-day monthly producers.
- R5 was refitted for May, June, and July 2026 with 3/6/9/12-month training
  windows.  Every held month is predicted by a model fitted before that month.
- The matched evaluator now accepts the explicit 28-day map sidecar.  It does
  not require legacy embedded map columns.
- Candidates without a valid trust posterior fail closed for posterior-only
  admission; frozen-admission overlays retain the causal Cell-day EV unchanged.

The internal names `causal_21d_*` in the evaluator are legacy wire aliases.
For the artifacts named `window28`, their values and admission bits come from
the explicit 28-day map sidecar recorded by hash in each manifest.

## Map construction

For decision day `t`, the map uses rows satisfying:

```text
t - 28 days <= decision_ts < t
label_available_ts < t 00:00 UTC
```

It creates twenty score cells, estimates equal-day expected policy net after a
symmetric 15% day trim, and admits at expected net >= +50 bps.  Outcomes use the
frozen SimplePolicyOptimiser policy and include the 100-bps cost exactly once.

### Raw admission comparison: 28 versus historical 42 days

These are pre-portfolio, all-admitted results on identical populations.

| Period | Window | Admitted | Share | Net bps/trade |
|---|---:|---:|---:|---:|
| 2025 Jan--Mar | 42d control | 23,617 | 7.55% | +132.06 |
| 2025 Jan--Mar | 28d | 25,851 | 8.26% | +116.87 |
| 2025 Apr--Jul | 42d control | 20,874 | 4.69% | +86.22 |
| 2025 Apr--Jul | 28d | 24,185 | 5.44% | +54.69 |
| 2026 Jan--Jul | 42d control | 29,560 | 3.58% | +52.96 |
| 2026 Jan--Jul | 28d | 28,554 | 3.46% | +58.73 |

The requested 28-day map is more adaptive but noisier in 2025.  It improves raw
2026 EV and is the only substrate used by the continuation below.

## R5 target, features, and training

Target:

```text
clip(policy_net_bps - causal_28d_cell_day_expected_net_bps, -500, +500)
```

Model: 64-tree Local Distribution Forest Proxy, depth 8, minimum leaf 120,
feature fraction 0.70, bootstrap fraction 0.75.  Training uses the top 30%
within decision timestamp, equal-month sampling, a 60,000-row cap, and only
labels resolved before the held cutoff.

The 66 inputs comprise upstream strict-R3/base/consensus/correctness fields and
causal support, OOD, Mahalanobis, covariance/correlation-break, K9-role, and
latest-model-fit active-rule state.  Cell-day map values are target anchors but
are not model inputs.  Raw K9 memberships are excluded.  Geometry is the frozen
October--December 2024 bundle and is never refit monthly.

For the 9-month May model, train-only stable-CMI retained six cross-family
interactions, principally geometry/active-rule OOD, OOD/support, and
geometry/Mahalanobis combinations.  Exact edges and recurrence are persisted
in each fold manifest; no held outcome participates in their discovery.

## Matched pre-portfolio results

All rows use the same May--July 2026 candidate and exact policy-outcome
population.  `All admitted` is the causal absolute +50-bps rule.  Tails are
pooled globally within that admitted population and are diagnostics, not live
thresholds.

| Arm | Admitted | All admitted | Top 0.5% | Top 1% | Top 2% | Top 5% |
|---|---:|---:|---:|---:|---:|---:|
| 28d Cell-day control | 8,622 | +22.67 | +141.46 | +97.96 | +62.33 | +95.02 |
| R5 3m posterior admission | 5,284 | +58.69 | +420.04 | +385.97 | +358.83 | +252.89 |
| R5 6m posterior admission | 5,121 | +54.77 | +455.07 | +365.85 | +346.90 | +261.69 |
| R5 9m posterior admission | 4,847 | +69.10 | +475.33 | +458.67 | +394.66 | +269.94 |
| R5 12m posterior admission | 4,713 | +72.82 | +524.42 | +437.98 | +363.58 | +263.40 |

The 9-month arm is preferred over 12 months for the frozen forward configuration:
it wins Top 1/2/5%, wins constrained EV, and has the best constrained worst
month.  The 12-month arm has slightly higher all-admitted raw EV and lower
portfolio drawdown, so it remains the stability alternative.

## Portfolio-constrained results

Portfolio contract: global auction, eight concurrent positions, two new entries
per 15-minute bar, one position per asset, 80% margin cap, 7x leverage, initial
wallet 1,000.  The outcomes are already materialised from the frozen
SimplePolicyOptimiser winner; they are not recomputed or reoptimized here.

| Arm | Trades | Trades/day | Net bps/trade | Positive | Sortino | Max DD | Worst week |
|---|---:|---:|---:|---:|---:|---:|---:|
| 28d Cell-day control | 871 | 9.47 | +72.30 | 60.16% | 0.256 | -56.63% | -52.75% |
| R5 3m posterior admission | 888 | 9.65 | +84.92 | 58.11% | 0.232 | -65.98% | -50.49% |
| R5 6m posterior admission | 772 | 8.39 | +104.34 | 58.29% | 0.292 | -40.51% | -29.31% |
| **R5 9m posterior admission** | **784** | **8.52** | **+114.36** | **58.80%** | **0.347** | **-40.79%** | **-31.51%** |
| R5 12m posterior admission | 745 | 8.10 | +110.54 | 58.79% | 0.329 | -36.79% | -20.55% |

The 9-month configuration improves constrained EV by **+42.06 bps/trade** versus
the 28-day control, lowers maximum drawdown by **15.84 percentage points**, and
raises Sortino by **0.092**.  Compounded wallet PnL is intentionally omitted as
a promotion statistic because 7x leverage and path-dependent compounding make
it extremely sensitive to this already-used development interval.

### Monthly portfolio economics

| Month | 28d control trades | Control net | R5 9m trades | R5 9m net |
|---|---:|---:|---:|---:|
| 2026-05 | 661 | +84.99 | 638 | +111.05 |
| 2026-06 | 0 | n/a | 64 | +123.98 |
| 2026-07 | 210 | +32.37 | 82 | +132.62 |

The posterior correction alters the admission map, as explicitly requested.
It repairs the 28-day control's June drought instead of merely reordering an
empty admitted set.

### Weekly stability of the 9-month configuration

- 12 active weeks out of 14 calendar weeks; 11 positive and one negative.
- Median active-week EV: +146.92 bps/trade.
- Worst active week: -271.40 bps/trade on 15 trades.
- Control: 9 active weeks, 8 positive and one negative; median +76.94 and worst
  -137.55 bps/trade.

The selected configuration improves coverage and typical-week EV, but its one negative
week is more severe.  This is the main forward risk to monitor.

## Causality and coverage audit

- Candidates and features are target-free at scoring time.
- Each monthly R5 fit uses only rows whose policy label resolves before cutoff.
- A fold must contain the complete declared nine-calendar-month mapped-history
  window. Earlier folds remain in the ledger but fail closed; they are never
  presented as nine-month fits built from a partial window.
- May/June/July models train over the preceding 9 months, respectively starting
  2025-08-01, 2025-09-01, and 2025-10-01.
- Held outcomes are joined only after scores and admission are frozen.
- Trust-prediction coverage is 97.99%--99.46%.  Missing posterior rows fail
  closed for posterior admission.
- Geometry/K9 semantics use one frozen bundle; no monthly K9 refit occurs.

## Sealed shadow verification

The complete schema-v4 hourly orchestrator was exercised at 2026-08-12 09:00
UTC. It built all 170 frozen-universe rows before filtering, rejected seven by
the contemporaneous actionability gate, regenerated features from the declared
hourly/15-minute sources, and passed 162/163 actionable rows as completely
finite on all 120 frozen fields (99.39%, above the sealed 90% cycle threshold).
It verified all nineteen artifact hashes and passed every runtime invariant:
target-free scoring, no held-window percentiles, same-bundle reference/held
scoring, frozen geometry, no current outcomes, cost applied once, complete
mapping, shadow-only mode, and zero exchange calls.

No candidate passed admission at that hour. The 28-day Cell-day estimates
ranged from -134.93 to -0.02 net bps and R5 posterior estimates from -182.69
to -46.35 net bps, all below +50. This validates fail-closed execution parity;
it is not a forward return observation. Authoritative orchestration receipt:
`data_perp/artifacts/strict_r3_hourly_shadow_r5_9m_posterior_20260812T090000Z_v3_featurefixed/run_manifest.json`.
- The 28-day map sidecar path and SHA-256 are persisted in every fit/evaluation
  manifest.

## Required next validation

1. Keep the 9-month R5 posterior-admission configuration frozen without changing
   features, target, thresholds, map, or portfolio policy.
2. Run it on August 2026 onward as untouched forward evidence.
3. Compare 9 versus 12 months only as a predeclared stability pair; do not
   reopen the 3/6-month or blend grids.
4. Promote only if the 9-month arm preserves positive constrained EV, does not
   worsen the negative-week tail materially, and maintains sufficient causal
   admission coverage.

## Earlier-period walk-forward confirmation

> **Superseded history boundary — 2026-08-12.** This paragraph described the
> input ledger available to the original R5 study. The canonical regeneration
> now supplies April--December 2024 training/calibration history, so the first
> January 2025 producer has a complete nine-month window. The authoritative
> artifact is
> `data_perp/artifacts/strict_r3_canonical_a5_long_2025_jul2026_2024warmup_20260812_v1`;
> its 21 January-2025--July-2026 folds all fit. The frozen October--December
> 2024 Geometry/K9 bundle is defined once at 2025-01-01 and never refit monthly.

Seven additional monthly fits were produced for October 2025--April 2026. Each
fit retained the same target, 66-field contract, top-30% timestamp-local
training population, 60,000-row equal-month sample, model parameters, 28-day
map, +50-bps hurdle, policy, and portfolio constraints. Only the nine-month
calendar window rolled forward.

### Portfolio comparison, October 2025--March 2026

This block predates the May--July window-selection interval.

| Arm | Trades | Trades/day | Net bps/trade | Positive | Sortino | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| 28d Cell-day control | 3,171 | 17.42 | +116.44 | 59.35% | 0.347 | -60.55% |
| R5 9m posterior admission | 2,678 | 14.71 | **+167.10** | **63.82%** | **0.477** | **-43.81%** |

The earlier block confirms an uplift of **+50.67 bps/trade**, with 16.75
percentage points less drawdown.

### Contiguous portfolio comparison, October 2025--July 2026

| Arm | Trades | Trades/day | Net bps/trade | Positive | Sortino | Max DD | Worst week |
|---|---:|---:|---:|---:|---:|---:|---:|
| 28d Cell-day control | 4,588 | 15.09 | +115.68 | 59.63% | 0.370 | -60.55% | -52.75% |
| **R5 9m posterior admission** | **3,972** | **13.07** | **+161.22** | **62.19%** | **0.490** | **-43.81%** | **-31.51%** |

The ten-month constrained uplift is **+45.53 bps/trade**. R5 is positive in
all ten months and improves nine; November 2025 is effectively flat versus the
control (+124.61 versus +125.27 bps/trade).

| Month | Control trades | Control net | R5 9m trades | R5 9m net | Uplift |
|---|---:|---:|---:|---:|---:|
| 2025-10 | 563 | +132.35 | 487 | +217.69 | +85.34 |
| 2025-11 | 420 | +125.27 | 410 | +124.61 | -0.66 |
| 2025-12 | 580 | +60.45 | 572 | +102.96 | +42.51 |
| 2026-01 | 615 | +119.00 | 388 | +217.01 | +98.01 |
| 2026-02 | 402 | +98.73 | 322 | +155.36 | +56.63 |
| 2026-03 | 591 | +159.32 | 499 | +194.95 | +35.63 |
| 2026-04 | 548 | +179.81 | 514 | +200.80 | +20.99 |
| 2026-05 | 659 | +85.27 | 634 | +111.71 | +26.44 |
| 2026-06 | 0 | n/a | 64 | +123.98 | n/a |
| 2026-07 | 210 | +32.37 | 82 | +132.62 | +100.25 |

Across 39 weeks in which both arms trade, R5 improves 31 and loses eight. Mean
paired-week uplift is +50.08 bps/trade; a 20,000-resample paired-week bootstrap
gives a 95% interval of **+23.72 to +80.57 bps/trade**. Across the entire
calendar, R5 has 41 positive and one negative active week. The single negative
week remains the main caveat: -271.40 bps/trade on 15 trades, worse than the
control's -137.55 on 69 trades that week.

### Pooled-global admission diagnostics, October 2025--July 2026

| Arm | Admitted | All admitted | Top 0.5% | Top 1% | Top 2% | Top 5% |
|---|---:|---:|---:|---:|---:|---:|
| 28d control | 49,217 | +52.80 | +12.73 | +40.43 | +50.08 | +88.04 |
| R5 9m posterior | 22,918 | **+127.47** | **+556.85** | **+493.91** | **+421.45** | **+354.43** |

The admission sets agree on 97.55% of all candidate rows. R5 retains 21,708
control admissions, rejects 27,509, and adds 1,210 rows whose posterior clears
the hurdle. Trust-prediction coverage is 99.45% over the ten-month population.

### Interpretation

This is a strong **chronological OOS-by-fold backward confirmation** of the
architecture. It is not an untouched final test: the broader 2025--26 period
has informed earlier research, and the 9-month choice was made after observing
May--July. The test does show that the May--July result is not isolated to that
regime. It does not authorize live promotion without later forward evidence.

## Reproducibility

- 28-day map producer: `scripts/ablate_strict_r3_cell_day_bayesian_ev_mapping.py`
- trust fold producer: `scripts/fit_strict_r3_cell_day_residual_trust.py` and
  `extreme_price_movements/strict_r3_cell_day_trust.py`
- matched evaluator: `scripts/evaluate_strict_r3_trust_posterior_admission.py`
- portfolio replay: `scripts/replay_strict_r3_tail_health_portfolio.py`
- generic immutable sidecar slicer: `scripts/subset_candidate_sidecar_by_time.py`
- leading fold artifacts:
  `data_perp/artifacts/strict_r3_cellday28_latestfit_R5C500_window9m_long_2026{may,jun,jul}_20260812_v1`
- leading evaluation artifacts:
  `data_perp/artifacts/strict_r3_cellday28_latestfit_R5C500_window9m_audit_long_2026{may,jun,jul}_20260812_v1`
- leading portfolio artifact:
  `data_perp/artifacts/strict_r3_cellday28_latestfit_R5C500_window9m_portfolio_long_2026mayjul_20260812_v1`
- earlier confirmation folds:
  `data_perp/artifacts/strict_r3_cellday28_latestfit_R5C500_window9m_confirm_long_2025oct_20260812_v1`
  through
  `data_perp/artifacts/strict_r3_cellday28_latestfit_R5C500_window9m_confirm_long_2026apr_20260812_v1`
- contiguous ten-month portfolio artifact:
  `data_perp/artifacts/strict_r3_cellday28_latestfit_R5C500_window9m_confirm_portfolio_long_2025oct_2026jul_20260812_v1`
