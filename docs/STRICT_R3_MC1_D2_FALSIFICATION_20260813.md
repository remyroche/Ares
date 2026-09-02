# Strict-R3 MC1_d2 frozen-champion falsification

## Status

MC1_d2 remains the frozen research champion. Robust-21 remains the production control. No result below authorizes parameter changes because 2026 has become model-selection/falsification evidence.

Frozen contract SHA-256:

`b1485219617884dfb1cb9bc7b58bf8faf3c8b1dfa87fa1e38786c2384b0ca8bc`

## Causal lineage receipt

The six MC1 inputs are:

1. `final_score`
2. `base_rank42`
3. `conditional_consensus_rank`
4. `upstream`
5. `ordinary_shadow_consensus_rank`
6. `correctness_rank`

Across 2,125,988 rows, each field is bit-identical between the target-free `walkforward_predictions.parquet` producer and the later outcome-joined evaluation ledger. The target-free file contains neither `policy_net_bps` nor `policy_label_available_ts`.

The authoritative source manifest records:

- zero held-window percentile operations;
- no outcomes consumed during scoring;
- outcomes joined after scoring;
- 28-day shared reserve excluded from active upstream/conversion fits;
- identical upstream/conversion cutoff per block;
- frozen Geometry/K9 hash across every block;
- strict-prequential stack identity.

This establishes that `correctness_rank`, `conditional_consensus_rank`, and `ordinary_shadow_consensus_rank` are already-fixed target-free outputs at the candidate decision timestamp. They are not recomputed from held outcomes.

## Admission provenance

| Cohort | Rows | Net bps/trade | Total net bps |
|---|---:|---:|---:|
| Shared MC1 + R21 | 10,242 | +179.62 | +1,839,670 |
| MC1-only | 8,492 | +144.87 | +1,230,269 |
| R21-only | 18,486 | +13.05 | +241,232 |

MC1-only minus R21-only separation is +131.82 bps/trade. MC1-only is positive in every 2026 month; June is +43.70 bps/trade.

## Component decomposition

Candidate-level 2026 admission results, before portfolio constraints:

| Component | Trades | Net bps/trade | Total net bps | Worst month |
|---|---:|---:|---:|---:|
| Score only | 20,860 | +58.66 | +1,223,698 | -14.44 |
| Score + 21d shift | 13,774 | +101.56 | +1,398,861 | +18.52 |
| Agreement only | 20,628 | +149.71 | +3,088,176 | +12.54 |
| Agreement + shift | 18,668 | **+164.68** | **+3,074,154** | +42.56 |
| Correctness only | 18,455 | +148.43 | +2,739,257 | +22.15 |
| Correctness + shift | 15,992 | +166.87 | +2,668,651 | +54.29 |
| Full, no shift | 20,349 | +149.82 | +3,048,615 | +12.54 |
| Frozen full + shift | 18,734 | +163.87 | +3,069,938 | +43.70 |

Interpretation:

- frozen score supplies the initial ordering but is insufficient for admission calibration;
- agreement supplies the largest cross-sectional improvement;
- the global shift improves temporal calibration and worst-month behavior;
- correctness changes the partition/downside profile, but does not improve mean EV over agreement + shift in this sample;
- the full arm remains frozen because these component comparisons were opened on 2026.

## Orthogonality and null controls

After fitting `E[agreement | frozen score]` on training data, orthogonal agreement retains:

- mean within-score-band Spearman: +0.65;
- positive ordering in 9/10 score bands;
- mean low-to-high spread: +39.67 bps.

At identical selected counts across three seeds:

| Control | Mean net bps/trade |
|---|---:|
| Observed candidate agreement | +124.23 |
| Within-day × score-band permutation | +121.80 |
| Previous-day agreement control | +121.34 |

The candidate-specific incremental effect is modest but consistent. The much larger full-stack uplift is therefore an interaction of structural agreement calibration, temporal shift, and admission authority—not agreement in isolation.

## Complexity and seed plateau

| Geometry | 2026 net bps/trade range | Positive months | Zero-trade days |
|---|---:|---:|---:|
| Depth 1, three seeds | +170.86 to +173.73 | 7/7 | 0 |
| Depth 2, three seeds × leaf floors | +162.29 to +166.71 | 7/7 | 0 |

Leaf floors 100/250/500 retain the effect. Depth 1 looking stronger is reassuring about smoothness but cannot be promoted on this evidence.

## Concentration falsification

Frozen MC1 leave-one-month-out 2026 EV ranges from +149.88 to +177.68 bps/trade:

- remove February: +149.88;
- remove June: +177.68;
- remove July: +163.54.

The result survives removal of every month, including the drought-breaking and strongest rebound periods.

June MC1-only predicted-EV deciles have Spearman +0.59 with realized EV. The relationship is directionally useful but not perfectly monotonic, so no threshold or calibration refinement is authorized.

## Forward promotion

Promotion gates are frozen in `config/strict_r3_mc1_d2_forward_promotion_gates_20260813_v1.json`. At least 42 new calendar days, 300 accepted trades, 100 resolved MC1-only rows, and six resolved weeks are required. All economic, downside, drought, calibration and concentration gates must pass. Failures do not authorize tuning on the forward period.
