# Strict-R3 outcome, admission, and conversion repair — long-only handover

**Date:** 2026-08-10  
**Status:** completed research repair; not production-approved  
**Evaluation:** January–July 2025 development and January–July 2026 confirmation  
**Execution target:** SimplePolicyOptimiser winner, 12-hour timeout, 100-bps cost once

## 1. Decision

The selected research stack is:

```text
target-free point-in-time candidates
→ strict-prequential R3 D2 base
→ prior-42-day rank and prior-prequential policy-net map
→ ten ordinary policy-residual LambdaRank heads
→ 75/25 base-consensus blend
→ rolling three-month C3 geometry, aligned between refits
→ policy-residual correctness model, without raw K9 memberships
→ same-model prior-42-day CDF
→ causal 21/42/84-day hierarchical tail EV map at +50 bps
→ long-only constrained portfolio auction
```

Severe-200 is **not** part of the selected live score. It remains a shadow
diagnostic whose target is frozen to:

```text
exact H12 TP6/SL4 net bps <= -200
```

It was deliberately not retargeted to the optimized trailing policy. The
policy-aligned correctness model is a distinct conversion arm; it must not be
described as Severe-200.

## 2. Outcome-substrate repair

The original policy-outcome surface was incomplete, and 27 of 154 symbols had
no valid optimized-policy outcome in the January–July 2025 interval. Selecting
tails from all scores and then evaluating only rows with a path was causally
correct, but it left unstable support and prevented a complete assessment of
the missing symbols.

The repaired surface preserves exact and 15-minute outcomes first, then fills
only missing paths with a declared hourly-OHLC replay. The replay uses a causal
hourly ATR, next-bar entry, the frozen optimized policy, a 12-hour timeout, and
the fixed 100-bps cost exactly once.

| Coverage audit | Before | After |
|---|---:|---:|
| Valid rows | 1,792,690 | 2,174,248 |
| Coverage | 82.45% | 99.997% |
| Newly filled rows | — | 381,558 |
| Jan–Jul 2025 valid rows | incomplete | 757,733 / 757,733 |
| Jan–Jul 2025 symbols with zero valid rows | 27 | 0 |

On 296,203 rows where both sources exist, the hourly replay has 0.945
Spearman correlation, 96.50% sign agreement, 34.50-bps mean absolute error,
and +6.02-bps bias. The 2025 monthly overlap is materially better: Spearman
is 0.979–0.989, sign agreement is 98.8–99.1%, and MAE is 15.0–23.8 bps.
The source is useful for 2025 coverage, but it is not interchangeable with
finer paths. All final results retain a source split.

## 3. Future-path selection repair

Global tails are selected from every finite-score candidate before outcome
validity is checked. The portfolio auction also no longer removes candidates
because their future path is missing. An admitted unresolved row reserves a
conservative H12 slot and remains outcome-unavailable; it cannot be replaced
by a lower-ranked candidate using future knowledge. The final matched replays
accepted zero unresolved rows, but the invariant is now enforced in tests.

## 4. Conversion overlay ablation

All figures below are optimized-policy net bps/trade. The H12 arms use the
unchanged Severe-200 TP6/SL4 target; `correctness` uses policy-residual
correctness. `K9` means the nine raw soft cluster memberships are included.
Aggregate leaf/geometry support fields remain available in every correctness
arm.

### 2025 development screen

| Overlay | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Top-2 portability | Worst month |
|---|---:|---:|---:|---:|---:|---:|---:|
| Correctness, no raw K9 | +162.93 | +124.49 | +92.82 | +57.42 | +30.32 | +55.55 | +43.90 |
| Correctness + raw K9 | +161.11 | +117.22 | +86.79 | +52.01 | +28.36 | +47.30 | +29.26 |
| Severe H12, K9, alpha 0.10 | +170.02 | +133.64 | +96.30 | +62.03 | +36.56 | +72.10 | +41.47 |
| Severe H12, K9, alpha 0.25 | +172.65 | +138.28 | +103.70 | +68.57 | +39.92 | +92.16 | +55.59 |
| Severe H12, K9, alpha 0.50 | +171.97 | +144.61 | +115.13 | +77.96 | +45.28 | +106.08 | +72.63 |
| Severe H12, no K9, alpha 0.50 | +166.91 | +140.17 | +115.31 | +81.93 | +46.54 | +97.14 | +74.69 |

### 2026 transport check

| Overlay | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Top-2 portability | Worst month | Positive months |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Correctness, no raw K9** | **+157.01** | **+112.63** | **+73.62** | **+27.80** | +0.55 | **+72.74** | **+30.84** | **7/7** |
| Correctness + raw K9 | +150.06 | +107.09 | +66.51 | +19.39 | -3.02 | +43.05 | +4.95 | 7/7 |
| Severe H12, K9, alpha 0.10 | +99.92 | +84.66 | +55.13 | +24.03 | +1.73 | +47.61 | +17.41 | 7/7 |
| Severe H12, K9, alpha 0.25 | +92.64 | +76.98 | +55.78 | +24.25 | +1.99 | +52.98 | +4.28 | 7/7 |
| Severe H12, K9, alpha 0.50 | +82.37 | +65.51 | +53.48 | +24.93 | +2.45 | +31.24 | -17.97 | 5/7 |
| Severe H12, no K9, alpha 0.25 | +132.91 | +88.79 | +61.09 | +27.35 | +3.68 | +45.72 | +1.90 | 7/7 |
| Severe H12, no K9, alpha 0.50 | +113.45 | +86.74 | +58.09 | +28.10 | +4.47 | +25.31 | -23.26 | 5/7 |

The high-alpha Severe arms won in 2025 but did not transport to 2026. Periodic
refitting did not remove that regime dependence. Raw K9 memberships also
reduced the correctness arm in both years. The robust cross-era choice is
therefore correctness-only without the raw K9 vector. Severe remains a
periodically refit, exact-H12 shadow monitor; raw K9 memberships remain a
diagnostic rather than model inputs.

## 5. Full-cap ranking results

| Year | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Top-2 worst month | Positive Top-2 months |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 | +156.97 | +127.38 | +108.38 | +74.96 | +40.44 | +56.38 | 7/7 |
| 2026 | +171.19 | +129.38 | +88.99 | +39.84 | +6.59 | +44.08 | 7/7 |

The 2026 drop is concentrated in broader tails rather than the extreme tail:
Top 0.5% and Top 1% are slightly higher than in 2025, while Top 2/5/10 decline
by 19.39, 35.12, and 33.85 bps. This indicates a thinner profitable score
surface and weaker conversion below the best candidates, not a total loss of
top-tail opportunity.

The source split confirms that the result is not created by the coarse fill:

| Year | Source at global Top 2% | Trades | Share | Net bps/trade |
|---:|---|---:|---:|---:|
| 2025 | Existing exact/15-minute | 10,738 | 70.85% | +100.46 |
| 2025 | Hourly proxy | 4,417 | 29.15% | +127.65 |
| 2026 | Existing exact/15-minute | 10,721 | 63.52% | +67.57 |
| 2026 | Hourly proxy | 6,158 | 36.48% | +126.28 |

The hourly subset is more profitable, so the combined estimate is optimistic
relative to finer paths. The conservative finer-path-only evidence remains
positive, including +67.57 bps at 2026 Top 2%.

## 6. Hierarchical causal admission

The legacy 21-day, 20-bin map was too coarse in the only score region used by
the portfolio. The replacement keeps the causal 21-day response but uses
uneven bins with more resolution near the upper tail and shrinks estimates
toward 42- and 84-day side-local parents. Evaluation begins only after 84 days
of prior scored history. Admission remains mapped expected net >= +50 bps and
fails closed without support.

| Year | Admission map | Admission rate | Top-2 recall | Admitted net bps/trade |
|---:|---|---:|---:|---:|
| 2025 | Legacy 21-day / 20-bin | 6.43% | 76.11% | +52.61 |
| 2025 | **Hierarchical tail** | **5.75%** | **78.37%** | **+65.26** |
| 2026 | Legacy 21-day / 20-bin | 1.95% | 43.24% | +63.13 |
| 2026 | **Hierarchical tail** | **1.74%** | **57.58%** | **+86.81** |

The new map admits fewer candidates but recovers more of the true Top 2%,
especially in 2026. This removes the apparent recent-admission drought.

## 7. Portfolio replay

| Year | Map | Trades | Trades/day | Net bps/trade | Positive rate | Max drawdown |
|---:|---|---:|---:|---:|---:|---:|
| 2025 | Legacy | 3,054 | 14.41 | +95.03 | 60.61% | -89.79% |
| 2025 | **Hierarchical tail** | **3,128** | **14.75** | **+142.51** | **64.39%** | -92.67% |
| 2026 | Legacy | 1,406 | 6.63 | +67.10 | 52.28% | -79.31% |
| 2026 | **Hierarchical tail** | **1,942** | **9.16** | **+137.82** | **58.19%** | **-59.78%** |

Monthly hierarchical-tail portfolio net EV is positive throughout:

| Month | Trades | Net bps/trade | Month | Trades | Net bps/trade |
|---|---:|---:|---|---:|---:|
| 2025-01 | 629 | +145.78 | 2026-01 | 404 | +104.61 |
| 2025-02 | 540 | +107.48 | 2026-02 | 234 | +165.31 |
| 2025-03 | 339 | +153.28 | 2026-03 | 304 | +125.77 |
| 2025-04 | 561 | +176.38 | 2026-04 | 390 | +177.20 |
| 2025-05 | 483 | +91.89 | 2026-05 | 414 | +146.46 |
| 2025-06 | 139 | +161.49 | 2026-06 | 27 | +383.08 |
| 2025-07 | 437 | +179.15 | 2026-07 | 169 | +49.65 |

The exact/15-minute-only accepted subset remains positive at +130.02 bps in
2025 and +118.69 bps in 2026. The policy-level repair therefore survives the
conservative source check.

The unresolved blocker is drawdown. Positive trade-level EV is not equivalent
to a production-ready portfolio under the present correlated exposure and
wallet sizing. The next workstream should alter portfolio risk controls, not
the frozen Severe target: cluster-correlated exposure caps, volatility-scaled
margin, loss-streak/regime throttles, and a drawdown-aware daily risk budget.

## 8. Causal and model-contract invariants

- Candidates and cross-sectional features are generated without future-path
  validity or outcome completeness.
- Base/map/residual/correctness inputs are strict prequential and resolved
  before each four-week cutoff.
- Geometry uses a three-month burn-in and is aligned between refits; one
  downstream model consumes only one bundle's representation.
- Severe-200 always uses exact TP6/SL4 H12 net <= -200 bps. No policy-net
  Severe configuration is accepted.
- Correctness uses selected-policy residual economics and is a separate model.
- No raw held-window percentile operation is used.
- Admission uses prior-resolved 21/42/84-day evidence and fails closed.
- Outcome availability cannot alter candidate identity, tail membership, or
  auction replacement.
- The fixed 100-bps cost is applied once.

## 9. Implementation and artifacts

### Code

- `extreme_price_movements/strict_r3_frozen_policy_labels.py`
- `extreme_price_movements/stage_i_causal_admission.py`
- `extreme_price_movements/strict_r3_canonical_v2.py`
- `scripts/backfill_strict_r3_policy_outcomes_hourly.py`
- `scripts/run_strict_r3_c3_window_cadence_ablation.py`
- `scripts/run_causal_geometry_k9_c3_ablation.py`
- `scripts/run_ten_head_c3_full_stack_replay.py`
- `scripts/replay_strict_r3_forward_portfolio.py`
- `scripts/compare_strict_r3_admission_maps.py`

### Main immutable evidence

- `data_perp/artifacts/strict_r3_policy_outcomes_hourly_backfill_long_2025_jul2026_20260810_v1`
- `data_perp/artifacts/strict_r3_conversion_overlay_h12_hier_tail_screen_long_2025_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_conversion_overlay_h12_hier_tail_screen_long_2026_janjul_20260810_v2`
- `data_perp/artifacts/strict_r3_conversion_correctness_nok9_d2_fullcap_long_2025_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_conversion_correctness_nok9_d2_fullcap_long_2026_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_admission_map_comparison_correctness_nok9_d2_long_2025_janjul_20260810_v2`
- `data_perp/artifacts/strict_r3_admission_map_comparison_correctness_nok9_d2_long_2026_janjul_20260810_v1`

## 10. Promotion status

This work repairs the outcome substrate, removes future-path portfolio
selection, improves causal tail admission, and selects the most portable
conversion overlay. It does not constitute untouched validation: both 2025
and 2026 influenced development. Freeze the stack and evaluate a later period
before promotion. Production approval additionally requires acceptable
drawdown under realistic wallet and correlated-exposure constraints.
