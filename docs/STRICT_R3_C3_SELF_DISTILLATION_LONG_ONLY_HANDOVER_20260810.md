# Strict-R3 C3 and Self-Distillation — Long-Only Handover

> **2026-08-10 follow-on:** outcome coverage, future-path-safe portfolio
> selection, conversion overlays, and hierarchical tail admission are audited
> in `docs/STRICT_R3_OUTCOME_ADMISSION_CONVERSION_REPAIR_20260810.md`. That
> report supersedes this document for the selected conversion/admission layer.
> Severe-200 remains frozen to exact H12 TP6/SL4 net <= -200 bps and was not
> retargeted to the optimized trailing policy.

**Status:** completed causal research funnel; D2 base curriculum advances as the canonical research challenger.  
**Scope:** long side only.  
**Development:** January–July 2025.  
**Later confirmation:** January–July 2026.  
**Production status:** not approved; a later untouched frozen period remains required.

## 1. Final architecture

```text
target-free point-in-time long candidates
→ strict-R3 three-class base with D2 robust-clear curriculum
→ same-model prior-42-day base rank
→ prior-prequential selected-policy net map
→ ten ordinary policy-residual LambdaRank heads
→ 75% base rank + 25% median residual-head rank
→ C3 rolling three-month raw-market geometry/K9 bundle
→ current-base leaf/path state
→ Severe-200 demotion
→ +100-bps policy-residual correctness ranker
→ same-model prior-42-day CDF
→ causal side-local 21-day expected-net admission at +50 bps
→ constrained long-only portfolio auction
```

No held-window percentile is used. Global top-k is diagnostic only. The executable path uses the causal 21-day expected-net map.

Every diagnostic tail is selected from the complete finite-score population before future-outcome coverage is inspected. Missing or invalid policy paths therefore never improve the selected-set denominator. Net EV and hit rate are computed only on the selected rows whose outcomes are valid, and coverage is reported explicitly. This distinction is important in 2025, where top-tail policy-outcome coverage is only about 61–75%; 2026 coverage is materially fuller.

## 2. Selected downstream window and cadence

### Training-window screen

The downstream C3 safety/correctness layer was initially refit monthly. The raw geometry window ends before the downstream training window begins, so one downstream fit never consumes state values from two geometry bundles.

| Training window | Top-2% net | Portability score | Worst month | Positive months | Portfolio net | Trades/day |
|---:|---:|---:|---:|---:|---:|---:|
| 1 month | +81.68 | +127.23 | +30.92 | 7/7 | +64.82 | 9.56 |
| 2 months | +77.97 | **+145.32** | +10.17 | 7/7 | +74.43 | 9.04 |
| 3 months | +92.64 | +130.04 | +30.80 | 7/7 | +85.09 | 10.25 |
| 4 months | +113.44 | +89.22 | +62.20 | 7/7 | +85.24 | 11.16 |
| 5 months | +119.66 | +92.42 | **+69.81** | 7/7 | +72.02 | 11.71 |
| **6 months** | **+120.06** | +134.18 | +61.04 | **7/7** | +84.35 | **12.85** |

Six months is retained because it has the strongest pooled top-2 EV among the window arms while maintaining seven positive months. The two-month arm's high portability score is driven by the median/MAD form despite much weaker pooled EV and a +10.17-bps worst month; it does not displace six months.

### Refit cadence

| Cadence | Top-2% net | Portability score | Worst month | Portfolio net | Trades/day |
|---:|---:|---:|---:|---:|---:|
| **2 weeks — 2025 winner** | **+155.63** | **+148.30** | **+92.46** | +95.56 | **13.64** |
| **4 weeks — retained** | +128.37 | +141.88 | +66.66 | **+96.08** | 13.04 |
| 8 weeks | +125.09 | +121.26 | +56.17 | +83.79 | 13.86 |

Two weeks won the 2025 development screen, including stability. It did not transport: on January–July 2026 its top-2 pooled EV was +68.05 bps, but June contributed −157.20 bps, leaving 6/7 positive months and a −137.61 portability score. The matched four-week D0 control delivered +68.77 bps pooled, a +20.72-bps worst month, 7/7 positive months, and +61.97 portability. Four weeks is therefore retained; eight weeks was already inferior to four weeks in 2025.

### Geometry burn-in

The full-cap matched comparison selects three months:

| Burn-in | Top-2% net | Portability | Portfolio net |
|---:|---:|---:|---:|
| 2 months | +154.90 | +133.46 | +88.45 |
| **3 months** | **+155.63** | **+148.30** | **+95.56** |

Each four-week downstream fit receives exactly one newly fitted three-month raw-market K9 bundle. Cluster identities are aligned to the preceding bundle; every transform, reference row, training row, and held row in that fit uses the same bundle hash. Current-base leaf/path state is refit separately. This is C3 rolling geometry, not the older permanently frozen October–December 2024 K9 contract.

## 3. Execution policy and live admission

The SimplePolicyOptimiser winner was selected only on strict-prequential pre-2025 data:

| Parameter | Value |
|---|---:|
| Stop | 4.1520006 ATR |
| Trailing activation | 2.3262249 ATR |
| Giveback | 0.1023720 ATR |
| Timeout | 12 hours |
| Cost | 100 bps exactly once |

The 2026 outcome substrate uses exact resampled 15-minute paths wherever minute data exist. Timestamp-complete but stale/flat coarse paths are rejected rather than treated as valid zero-return outcomes.

Live admission is side-local and causal:

1. take the final CDF42 score;
2. use only fully resolved outcomes from the preceding 21 calendar days;
3. fit 20 bins with 5% trimming and pooled-parent shrinkage;
4. admit mapped expected net of at least +50 bps;
5. fail closed when support is insufficient;
6. apply concurrency, exposure, asset, entry-rate, and 80%-margin constraints.

## 4. Base self-distillation

### Teacher and weights

The teacher is each row's prior strict-prequential, side-local base `rank42`; it is never a held-month or timestamp-local rank.

The winning D2 arm boosts only R3 robust-clear rows that fall in the teacher's global top 20%:

```text
raw_weight = existing_weight × 1.5
    if R3 class is robust-clear and teacher_rank >= 0.80
raw_weight = existing_weight otherwise

final_weight = bounded mean-one projection(raw_weight, [0.25, 4.0])
```

Adverse-first and weak/timeout classes remain at ordinary weight. This preserves the three-class probability surface while modestly improving opportunity recall.

### Initial D0–D4 screen

| Arm | Description | Rank IC | Log loss | Brier | Top-30 recall | Top-40 recall |
|---|---|---:|---:|---:|---:|---:|
| D0 | Existing weights | 0.30388 | 0.96765 | 0.57520 | 35.16% | 46.08% |
| D1 | Smooth score weight | 0.30462 | 0.97264 | 0.57789 | 35.28% | 46.15% |
| D2 | Positive robust-clear boost | 0.30524 | 0.97677 | 0.57857 | 35.43% | 46.46% |
| D3 | High-score adverse boost | 0.30380 | 0.96958 | 0.57699 | 34.96% | 46.00% |
| D4 | Combined | 0.30532 | 0.98729 | 0.58394 | 35.28% | 46.22% |

D2 advanced because it improved recall. Threshold refinement selected top 20%; boost refinement selected 1.5x. It preserved zero global decile violations and nearly unchanged log loss while improving Brier.

### Base-only 2026 confirmation

| Arm | Rank IC | Log loss | Brier | Top-30 recall | Top-40 recall | Top-5 uplift |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 0.19992 | 1.07800 | 0.65529 | 32.01% | 42.56% | 8.06 pp |
| D2 top-20 ×1.5 | **0.20043** | **1.07189** | **0.65070** | **32.64%** | **43.16%** | **9.69 pp** |

The gain transports, although absolute 2026 base learnability remains weaker than 2025.

## 5. Residual self-distillation

The residual teacher is its own prior OOF/prequential residual rank. The economic classes are:

- positive: policy residual above +100 bps;
- negative: policy residual at or below −150 bps.

The initial 80k-row complete-query screen suggested D3 might help:

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% |
|---|---:|---:|---:|---:|---:|
| D0 | +66.55 | +43.61 | +28.91 | +4.26 | −13.94 |
| D1 | +66.66 | +42.80 | +28.38 | +3.28 | −13.28 |
| D2 | **+74.86** | +46.77 | +26.39 | +3.96 | −13.43 |
| D3 | +67.80 | **+46.95** | **+31.47** | **+5.48** | **−11.02** |
| D4 | +70.99 | +46.48 | +28.92 | +3.51 | −12.03 |

However, the full C3 stack rejected D3:

| Arm | Top-2% net | Portability | Worst month | Portfolio net |
|---|---:|---:|---:|---:|
| **D0 ordinary residual weighting** | **+128.37** | **+141.88** | **+66.66** | **+96.08** |
| D3 adverse-tail weighting | +117.46 | +135.77 | +44.05 | +93.04 |

The residual threshold/boost refinement stopped at this gate. Smooth-score exponent tuning also stopped because D1 did not win. Ordinary residual weighting remains canonical.

## 6. Sequential combined stack

The combined experiment obeys the stacking order:

```text
prior teacher OOF base rank
→ monthly D2 base fit
→ strict prequential D2 base predictions
→ causal policy-net map on prior predictions/outcomes
→ D0 residual refit on those predictions
→ C3 and executable stack
```

The residual learner never sees in-sample predictions from the newly fitted base.

### Matched global tails

| Year | Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% |
|---:|---|---:|---:|---:|---:|---:|
| 2025 | Matched D0 | **+192.06** | +160.46 | +131.61 | +78.43 | +35.25 |
| 2025 | D2 base + D0 residual | +170.21 | **+164.27** | **+137.97** | **+88.92** | **+45.05** |
| 2026 | Matched D0 | **+116.45** | +85.60 | +68.77 | +26.50 | −0.02 |
| 2026 | D2 base + D0 residual | +111.66 | **+91.10** | **+78.11** | **+35.61** | **+2.76** |

D2 sacrifices a small amount at top 0.5% but improves the declared top-1/top-2 secondary gate, top-5, top-10, and stability in both years.

### Top-2 stability

| Year | Arm | Pooled net | Median month | MAD | Worst month | Positive months | Portability |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2025 | Matched D0 | +131.61 | +136.31 | 55.21 | +65.66 | 7/7 | **+108.71** |
| 2025 | D2 base | **+137.97** | **+137.97** | 60.16 | **+68.06** | 7/7 | +107.89 |
| 2026 | Matched D0 | +68.77 | +71.94 | 19.94 | +20.72 | 7/7 | +61.97 |
| 2026 | D2 base | **+78.11** | **+82.42** | **8.08** | **+33.45** | 7/7 | **+78.37** |

### Paired day bootstrap: D2 minus matched D0

| Year | Tail | Mean delta | 95% interval | P(delta > 0) |
|---:|---:|---:|---:|---:|
| 2025 | 1% | +3.49 | [−18.37, +22.77] | 63.7% |
| 2025 | 2% | +6.42 | [−10.72, +21.70] | 78.8% |
| 2025 | 5% | +10.37 | [+0.24, +20.65] | 97.8% |
| 2026 | 1% | +5.82 | [−6.36, +18.53] | 82.0% |
| 2026 | 2% | +9.58 | [+0.32, +20.78] | 97.9% |
| 2026 | 5% | +9.06 | [+1.28, +18.03] | 99.1% |

This is supportive rather than untouched proof. Top-1/top-2 intervals still cross zero.

## 7. Causal admission and constrained portfolio

| Year | Arm | Trades | Trades/day | Net bps/trade | Positive rate | Max drawdown |
|---:|---|---:|---:|---:|---:|---:|
| 2025 | Matched D0 | 2,642 | 12.46 | +81.71 | 60.1% | −95.3% |
| 2025 | D2 base | **3,151** | **14.86** | **+95.42** | **62.4%** | −94.2% |
| 2026 | Matched D0 | 1,356 | 6.40 | +48.95 | 52.0% | −78.7% |
| 2026 | D2 base | **1,471** | **6.94** | **+60.08** | **52.6%** | −81.0% |

The D2 stack improves admitted EV and participation. Drawdown remains unacceptable for production. June–July 2026 have no admitted trades because the recent-EV map has no supported bin at +50 bps; this is causal fail-closed behavior, not missing scores.

The auction results are outcome-covered executable replays: admission scores and thresholds are causal, but rows without valid future policy paths cannot have realized PnL simulated. They must not be interpreted as proof that an unavailable future path could have been skipped live.

## 8. Reliability diagnostics

- Top-20 base feature-importance consecutive-month Jaccard is 0.773 for D0 and 0.778 for D2.
- Calibration curves, score-decile economics, dominant-K9 tail economics, and selected-set overlap are materialized with the report artifacts.
- D2/D0 selected-set Jaccard at top 2% is 0.756 in 2025 and 0.711 in 2026: the curriculum changes ranking materially but does not replace the whole population.
- The targeted correctness suite passes, including explicit tests that score-tail selection precedes future-path coverage and that invalid paths are not converted to economic failures.
- 2026 exact policy rows use exact-minute precedence; stale timestamp-complete coarse paths cannot silently become valid flat trades.

## 9. Canonical decision

Canonical research settings now are:

- long side only;
- six-month downstream training window;
- four-week downstream refit cadence;
- three-month C3 raw-geometry/K9 burn-in;
- D2 top-20 robust-clear base curriculum at 1.5x;
- ordinary residual-head weighting;
- selected pre-2025 SimplePolicyOptimiser exit policy;
- causal 21-day side-local +50-bps expected-net admission;
- portfolio auction and constraints.

The older permanently frozen October–December 2024 K9 bundle remains a schema-v2 historical control. It is no longer the selected C3 downstream representation. Residual D3 and score-power exponent tuning do not advance.

## 10. Implementation and artifacts

| Item | Path |
|---|---|
| C3 window/cadence runner | `scripts/run_strict_r3_c3_window_cadence_ablation.py` |
| Self-distillation weights | `extreme_price_movements/strict_r3_self_distillation.py` |
| Self-distillation runner | `scripts/run_strict_r3_self_distillation.py` |
| Corrected policy replay | `scripts/replay_strict_r3_simple_policy_15m.py` |
| Causal-denominator recomputation | `scripts/recompute_strict_r3_c3_causal_tail_metrics.py` |
| Report generator | `scripts/report_strict_r3_self_distillation_funnel.py` |
| 2025 matched D0 | `data_perp/artifacts/strict_r3_self_distillation_matched_d0_fullstack_long_2025_janjul_20260810_v1` |
| 2025 D2 winner | `data_perp/artifacts/strict_r3_self_distillation_combined_base_d2_residual_d0_fullstack_long_2025_janjul_20260810_v1` |
| 2026 matched D0 | `data_perp/artifacts/strict_r3_self_distillation_matched_d0_fullstack_long_2026_janjul_exact_policy_20260810_v1` |
| 2026 D2 winner | `data_perp/artifacts/strict_r3_self_distillation_combined_base_d2_residual_d0_fullstack_long_2026_janjul_exact_policy_20260810_v1` |
| Corrected funnel audit | `data_perp/artifacts/strict_r3_c3_causal_tail_recompute_funnel_long_2025_janjul_20260810_v3` |
| Consolidated report artifacts | `data_perp/artifacts/strict_r3_c3_self_distillation_long_only_report_20260810_v3` |

The next gate is a later untouched frozen period and a risk-budget repair; neither the development nor 2026 confirmation evidence authorizes production deployment.
