# Short base target / objective funnel — 2026-08-20

## Decision

**No short base target advances to meta-layer or trading use.**  The strongest
candidate, standard H12-net LambdaRank (C1), improves strict-OOS
within-timestamp net ranking as history grows, but it does not produce a
positive, stable exact-policy top-1% tail.  This remained true after the
predeclared twelve-month retrain using the completed side-local Stage-I feature
selection contract.  The staged capacity/HPO branch was therefore deliberately
not entered.

This is a sequential funnel result, not a failure of the measurement:
all score selections were made before the exact one-minute exit paths were
opened.  No portfolio, admission map, meta/consensus model, or policy HPO was
included; this isolates the base head.

## Fixed substrate

| Item | Contract |
|---|---|
| Side | Short only |
| Candidate population | Complete target-free 170-symbol hourly grid; rejection/entry status is decision-time only |
| Features | 3/6/9m: frozen `base_fields_by_side.short`, 120 fields.  12m: the frozen 15-field side-local Stage-I subset, selected before the Oct–Dec holdout; every retained field passes >=90% target-free, entry-executable training-population coverage. |
| Entry | Exact one-minute decision open at signal close + one hour |
| Primary label horizon | 12 hours, label available only at decision +12h |
| H12 label geometry | TP +6 ATR / SL -4 ATR; adverse same-minute tie; 100 bps cost once |
| Fixed policy diagnostic | Short exact one-minute policy: SL 3 ATR, trailing activation 0.5 ATR, giveback 0.25 ATR, 12-hour timeout, 100 bps cost once |
| OOS rule | Train rows require `label_available_at < held-window start`; invalid/incomplete paths are null/excluded, never zero-valued failures |

The full 2024 short feature panel contains 1,493,280 target-free rows.  The
full exact-label ledger contains all twelve month partitions.  Valid H12 label
coverage rises from 25.2% in January to 51.6% in December; unavailable paths
are retained for coverage audit but excluded from fitting and outcome metrics.

## Round 1 — target / objective screen

Training was January–March 2024; OOS was April–June 2024.  `Policy top-1%` is
the exact one-minute policy net bps per resolved selected trade; it is the
economic selection metric, not AUC.

| Arm | Target / objective | Global net IC | Mean timestamp net IC | Positive timestamp fraction | H12 top-0.1% net bps | Policy top-1% net bps |
|---|---|---:|---:|---:|---:|---:|
| A0 | R3, `P(clear)-0.5P(adverse)` | -0.009 | 0.034 | 56.9% | -45.65 | -86.42 |
| A1 | R3, equal timestamp | -0.014 | 0.033 | 56.8% | -80.73 | -86.44 |
| B1 | clipped H12-net Huber | -0.031 | 0.030 | 56.2% | +35.55 | -134.80 |
| C1 | H12-net LambdaRank, standard economic bins | +0.023 | +0.047 | 61.4% | **+135.03** | **-71.81** |
| C2 | tail-focused H12-net LambdaRank | +0.018 | +0.045 | 60.1% | +93.79 | -82.31 |
| D1 | H12-gross LambdaRank | +0.023 | +0.047 | 61.4% | +135.03 | -71.81 |
| E1 | `P(H12 net > 0)` | +0.024 | **+0.052** | **62.2%** | -25.49 | -77.52 |
| E2 | `P(H12 net > 100)` | +0.017 | +0.044 | 58.8% | +26.07 | -87.70 |
| F1 | ordinal H12 economics | **+0.026** | +0.050 | 62.1% | -74.70 | -77.45 |

`D1` is an exact duplicate of `C1` under the fixed 100-bps cost: gross = net
+ 100 bps and its shifted bins preserve the same ordering.  It is not an
independent candidate.

## Round 2 — weighting screen

Only C1, E1 and F1 were expanded over the predeclared weighting choices.

| Best view by family | Weight | Mean timestamp net IC | Policy top-1% net bps | Result |
|---|---|---:|---:|---|
| C1 | ordinary | +0.047 | **-71.81** | Best C1 policy tail |
| C1 | equal-month × timestamp | +0.048 | -75.37 | More balanced but worse policy tail |
| E1 | equal timestamp | +0.052 | -77.52 | Learns class probability; no economic advance |
| F1 | equal-month × timestamp | +0.052 | -73.59 | Closest ordinal challenger; no economic advance |

Recency/equal-month variants did not improve the policy tail.  C1 ordinary
and C1 equal-month × timestamp were retained for the history test solely
because C1 was the only arm with a positive H12 top-0.1% economic tail.

## Round 3 — training-history test

All rows are evaluated on the *same* October–December 2024 OOS block.  The
3-, 6-, and 9-month fits respectively use Jul–Sep, Apr–Sep, and Jan–Sep as
their decision-time training windows.  Every training row’s H12 label had
resolved before 1 October.

| History | Weight | Global net IC | Mean timestamp net IC | Positive timestamp fraction | H12 top-0.1% net bps | Exact-policy top-0.1% net bps | Exact-policy top-1% net bps |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3m | ordinary | +0.032 | +0.076 | 69.4% | -33.77 | -144.94 | -89.88 |
| 3m | equal-month × timestamp | +0.037 | +0.083 | 71.5% | -119.62 | -85.77 | -100.46 |
| 6m | ordinary | +0.034 | +0.089 | 72.6% | -19.24 | -61.52 | -83.06 |
| 6m | equal-month × timestamp | +0.039 | +0.093 | 74.0% | +68.08 | -16.79 | -88.42 |
| 9m | ordinary | **+0.050** | +0.100 | 75.0% | **+111.85** | **+5.37** | -78.02 |
| 9m | equal-month × timestamp | +0.049 | **+0.104** | **77.3%** | +27.37 | -3.79 | -74.74 |

The expanded October-2023–September-2024 120-field panel exposed five genuine
historical source-coverage gaps (86.8–89.4% coverage), so the full frozen
contract correctly failed closed for the 12m fit.  The 12m rerun instead uses
the already completed long-equivalent Stage-I selector: target-free
coverage/variance, univariate plus bounded ReliefF rescue, target-free
Spearman representatives, two chronological economic-MDA folds, then the
smallest prefix within one standard error.  Its 15 fields are a subset of the
same configured short base family; no label, path, score, or future field was
introduced.

| History | Features | Weight | Global net IC | Mean timestamp net IC | Positive timestamp fraction | H12 top-0.1% net bps | Exact-policy top-0.1% net bps | Exact-policy top-1% net bps |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 12m | 15, Stage-I frozen subset | ordinary | +0.033 | **+0.084** | **71.0%** | +197.06 | -30.79 | -18.80 |
| 12m | 15, Stage-I frozen subset | equal-month × timestamp | **+0.035** | +0.080 | 69.4% | **+323.26** | -33.25 | **+7.99** |

The 9m ordinary C1 result is the closest candidate, but its positive
exact-policy result is restricted to the top 0.1% (204 resolved rows).  It
fails the required broad-tail and month-stability gate.

The predeclared second-best Round-2 target, F1 ordinal economics with
equal-month × timestamp weighting, was also taken through the full history
ladder.  It does not change the decision.

| F1 history | Features | Global net IC | Mean timestamp net IC | Positive timestamp fraction | H12 top-0.1% net bps | Exact-policy top-0.1% net bps | Exact-policy top-1% net bps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3m | 120 | +0.004 | +0.076 | 70.2% | -154.11 | -223.96 | -81.46 |
| 6m | 120 | +0.033 | +0.084 | 69.6% | -223.72 | -113.92 | -133.49 |
| 9m | 120 | +0.075 | **+0.091** | **72.5%** | -48.48 | -49.44 | -86.79 |
| 12m | 15, Stage-I frozen subset | +0.022 | +0.070 | 65.9% | **+966.55** | **+174.14** | -29.84 |

F1's 12m extreme 0.1% result is a small-tail observation only: its top-1/2/5%
exact-policy means are -29.84/-68.18/-85.01 bps.  Its month-level top-1%
means are -66.31, +9.84, and +18.94 bps in October, November, and December.
It fails the same broad-tail advancement gate as C1.

### Month stability — exact-policy top 1%

| Arm | October | November | December | Worst month |
|---|---:|---:|---:|---:|
| C1, ordinary | -70.31 | -117.66 | -54.67 | -117.66 |
| C1, equal-month × timestamp | -70.56 | -99.19 | -69.15 | -99.19 |
| Matched R3 control, ordinary | -92.81 | -21.68 | +67.81 | -92.81 |

| Arm | October | November | December | Worst month |
|---|---:|---:|---:|---:|
| 12m C1, Stage-I 15, ordinary | -47.63 | +9.49 | -15.03 | -47.63 |
| 12m C1, Stage-I 15, equal-month × timestamp | -36.97 | +6.24 | +44.15 | -36.97 |
| 12m R3 control, Stage-I 15 | -101.00 | -56.82 | -182.14 | -182.14 |

The equal-month 12m arm has a barely positive pooled top-1% mean, but fails
the required all-month stability gate and has a -566.03 bps policy CVaR(10%)
in that same selected tail.  It is not an advancement signal.

The matched R3 control is not used to select C1; it is provided only to
distinguish rank learning from policy alignment.

### Matched 9m control comparison

Both models use Jan–Sep training, Oct–Dec OOS, exactly the same candidate
population and the same fixed exact-one-minute exit policy.

| Model | Global net IC | Mean timestamp net IC | Top-0.1% policy net bps | Top-1% policy net bps | Top-2% policy net bps | Top-5% policy net bps |
|---|---:|---:|---:|---:|---:|---:|
| R3 control | **+0.080** | +0.040 | -61.58 | **-50.99** | **-48.96** | **-55.99** |
| C1 H12-net LambdaRank | +0.050 | **+0.100** | **+5.37** | -78.02 | -89.06 | -85.93 |

C1 genuinely improves within-timestamp H12 opportunity ordering and the
extreme top 0.1%, but it worsens the broad executable tail relative to R3.
It is therefore not a replacement base target.

### Matched 12m Stage-I control comparison

Both rows use the October-2023–September-2024 training window, the same
15-field selected causal contract, identical Oct–Dec target-free candidates,
and the same exact one-minute fixed policy diagnostic.

| Model | Global net IC | Mean timestamp net IC | Top-0.1% policy net bps | Top-1% policy net bps | Top-2% policy net bps | Top-5% policy net bps |
|---|---:|---:|---:|---:|---:|---:|
| R3 control | +0.008 | +0.021 | -182.92 | -105.39 | -85.49 | -85.94 |
| C1 ordinary | +0.033 | **+0.084** | -30.79 | -18.80 | -43.94 | -70.76 |
| C1 equal-month × timestamp | **+0.035** | +0.080 | -33.25 | **+7.99** | -26.69 | -71.19 |

The selected features therefore improve the economic-H12 ordering materially
relative to matched R3, but do not make the broad policy tail viable.

## Advancement decision and diagnosis

The predeclared capacity/HPO sweep is **not run**.  It applies only after a
target/weight/history candidate clears a positive and stable exact-policy
tail.  Neither C1 nor F1 did.  Increasing capacity after this result would be
a post-hoc search over a rejected base economics contract.

The evidence points to a conversion problem rather than a zero-information
short feature set:

1. C1’s query IC improves monotonically from 0.076 (3m) to 0.100 (9m), and
   75% of timestamp queries are positive.
2. That improvement does not transport from H12-net ordering into the
   trailing-policy payoff: C1’s policy top-1% remains negative in every OOS
   month.
3. The R3 control is less useful within a timestamp but less poor in the
   broad policy tail.  The target that best recognises H12 opportunity is not
   the target that ranks paths well for this exit geometry.

Next research should therefore keep the base experiment separate from policy
conversion: use the short R3/C1 outputs as OOF base inputs to an explicit
short policy-residual/reliability layer, or redefine the base ranking target
against a policy-aligned, cost-aware event.  Neither action is included in
this funnel or promoted by it.

## Reproducible artifacts

- Runner: `scripts/run_short_base_target_objective_funnel.py`
- Focused tests: `extreme_price_movements/tests/test_short_base_target_objective_funnel.py`
- Target-free features: `data_perp/artifacts/strict_r3_short_features_full2024_20260820_v1`
- Exact label ledger: `data_perp/artifacts/strict_r3_short_target_labels_full2024_20260820_v1`
- Round 1: `data_perp/artifacts/strict_r3_short_base_target_objective_round1_3m_oos_2024_20260820_v1`
- Round 2: `data_perp/artifacts/strict_r3_short_base_target_objective_round2_weighting_3m_oos_2024_20260820_v1`
- History runs: `data_perp/artifacts/strict_r3_short_base_target_objective_history_{3m,6m,9m}_oos_2024q4_20260820_v1`
- Matched control: `data_perp/artifacts/strict_r3_short_base_target_objective_history_9m_r3control_oos_2024q4_20260820_v1`
- F1 history runs: `data_perp/artifacts/strict_r3_short_base_target_objective_f1_history_{3m,6m,9m}_oos_2024q4_20260820_v{1,2,1}`
- Expanded Oct-2023–Dec-2024 target-free panel: `data_perp/artifacts/strict_r3_short_features_oct2023_2024_20260820_v1`
- Expanded exact short ledger: `data_perp/artifacts/strict_r3_short_target_labels_oct2023_2024_20260820_v1`
- Frozen Stage-I selected feature contract: `data_perp/artifacts/strict_r3_short_stagei_style_feature_selection_2024q1_20260820_v7/selected_features.json`
- 12m C1 retrain: `data_perp/artifacts/strict_r3_short_base_target_objective_history_12m_stagei15_oos_2024q4_20260820_v1`
- Matched 12m R3 control: `data_perp/artifacts/strict_r3_short_base_target_objective_history_12m_stagei15_r3control_oos_2024q4_20260820_v1`
- 12m F1 retrain: `data_perp/artifacts/strict_r3_short_base_target_objective_f1_history_12m_stagei15_oos_2024q4_20260820_v1`

The runner’s compact target-audit receipt stores continuous-label distribution
summaries and rank-relevance support, rather than serialising one entry per
unique realised bps value.  Historical immutable manifests are preserved as
created; this only affects subsequent artifacts.
