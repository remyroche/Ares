# Hourly exit-parameter ablation — 2026-08-13

> **Baseline reconciliation — superseded evidence.** The v2 metrics below are
> paired and internally OOF, but they are not representative of the canonical
> 2026 candidate stream. The inherited 12,000-row compute cap selected the
> lexicographic head of `candidate_id` within month, leaving only 28
> alphabetically early symbols. In addition, only rows with locally complete
> 15-minute paths could enter the exit replay. In 2026, the v2 covered subset
> has -12.12 net bps/trade, while the complete canonical score>=0.90 ledger has
> +9.06 and the causal EV-admitted ledger has +86.99. A deterministic
> month-stratified hash sampler now replaces the identity-head sampler. Its
> 12,000-row population covers 169 symbols, but local 15-minute availability
> remains outcome-selective: all sampled 2026 rows average +3.97, locally
> covered rows -19.98, and missing-path rows +79.43 bps/trade. Therefore all
> adaptive-uplift and authority numbers below remain conditional diagnostics
> only and must be rerun after historical 15-minute coverage is completed. They
> must not be compared with canonical admitted or portfolio EV and must not be
> used for promotion.

## Decision

The hourly controller is learnable, but the useful formulation is not the
one-hour-only target. The portable arm forecasts the remaining H12 favorable
and adverse geometry from each completed hourly state, then adjusts the frozen
SimplePolicyOptimiser trailing activation. The stop forecast is useful as a
diagnostic, but allowing the model to move the stop reduced performance.

Nothing from this experiment is promoted automatically. The authoritative run
status is `COMPLETE_NOT_PROMOTED_PENDING_USER_DECISION`.

## Frozen upstream contract

- Long side only.
- Candidate population is selected causally from the upstream score before
  future-path coverage is checked.
- Development selection: strict OOF 2025 only.
- Confirmation: strict OOF 2026, not used for feature/population selection.
- Four expanding chronological folds, with a 12-hour purge.
- State is formed after every four completed 15-minute bars; the chosen action
  starts on the next 15-minute bar. This is the once-per-hour inference clock.
- Frozen policy artifact:
  `data_perp/artifacts/strict_r3_schema_v2_simple_policy_targetfree_long_pre2025_20260809_v3/winner.json`
- Policy SHA-256:
  `2dc9a145766ae383a4ab7c33e8a9f9e358175597e05582300ff0a05732673603`
- Frozen baseline geometry: stop 4.1520 ATR, trailing activation 2.3262 ATR,
  fixed trailing gap 0.10237 ATR, H12 timeout, 100 bps round-trip cost exactly
  once.
- Exact static-baseline parity is demonstrated by the unchanged activation
  level-2 control: 0.000 bps/trade uplift.

This is a candidate-local exit study on the existing top-10% upstream
population. It does not include a fresh EV-admission or portfolio-constrained
replay and therefore cannot by itself authorize live deployment.

## Corrected hourly state

The prior implementation advanced cumulative MFE/MAE only on the bars sampled
by the hourly decision clock. It could therefore omit extrema in the three
intervening 15-minute bars. The implementation now builds vectorized prefix
state over every completed source bar and reads that state at each hourly
decision.

All ATR normalization now uses the exact effective policy ATR, including the
same minimum barrier floor as the frozen policy. This removed pathological ATR
ratios caused by near-zero raw ATR values.

The selected `S24_short_path` contract has 28 fields:

1. `trade_age_hours`
2. `pnl_atr`
3. `mfe_atr`
4. `mae_atr`
5. `drawdown_from_mfe_atr`
6. `fraction_given_back`
7. `time_since_mfe_hours`
8. `time_since_mae_hours`
9. `pnl_velocity_bps_per_hour`
10. `pnl_acceleration_bps_per_hour2`
11. `current_effective_stop_distance_atr`
12. `distance_to_trailing_activation_atr`
13. `trailing_active`
14. `atr_frac`
15. `return_15m_atr`
16. `return_30m_atr`
17. `return_since_prior_decision_atr`
18. `path_range_since_prior_decision_atr`
19. `path_close_location_side`
20. `path_realized_vol_since_prior_decision_atr`
21. `path_positive_bar_fraction`
22. `path_efficiency_since_entry`
23. `mfe_increment_since_prior_decision_atr`
24. `mae_increment_since_prior_decision_atr`
25. `drawup_from_mae_atr`
26. `new_mfe`
27. `new_mae`
28. `fraction_horizon_elapsed`

The larger 37-field entry-context arm was worse. This supports a compact,
short-term, path-since-entry controller rather than reusing broad entry-model
context.

## Feature and population selection

Selection score was `mean target Spearman + 0.25 * worst target Spearman`, on
2025 OOF predictions only.

| Feature contract | Population | States | Mean Spearman | Worst target | Selection score |
|---|---:|---:|---:|---:|---:|
| S24 short path | Top 10% | 32,280 | 0.4840 | 0.3349 | 0.5678 |
| S14 core | Top 10% | 32,280 | 0.4627 | 0.2872 | 0.5345 |
| S33 path + entry | Top 10% | 32,280 | 0.4321 | 0.3113 | 0.5099 |
| S24 short path | Top 5% | 14,395 | 0.4131 | 0.1524 | 0.4512 |
| S14 core | Top 5% | 14,395 | 0.4037 | 0.1586 | 0.4433 |
| S33 path + entry | Top 5% | 14,395 | 0.3909 | 0.1929 | 0.4392 |

## Targets and query construction

Four causal targets are materialized after each completed hourly state:

- next-hour favorable excursion in ATR;
- next-hour adverse excursion in ATR;
- remaining-H12 favorable excursion from entry in ATR;
- remaining-H12 adverse excursion from entry in ATR.

The next-hour target starts at the next 15-minute open and covers the following
four 15-minute bars. Conservative same-bar touch ordering gives the adverse
event priority.

Forecasts use shallow LightGBM quantile regression: q65 for favorable movement
and q80 for adverse movement, depth 4 / 15 leaves, a 700-tree ceiling, and
30-round early stopping. Forecast labels are capped only inside the actionable
model domain: 8 ATR for next-hour and 12 ATR for remaining-H12 geometry; raw
targets remain persisted.

Direct-action alternatives use all 25 stop/activation actions for one causal
state as a LambdaRank query. The tested targets were ordinal delta-Q, Huber
exact delta-Q, and binary `delta-Q > 25 bps`. This state-action query is the
correct grouping for choosing an action; timestamp-only queries would compare
different trades rather than actions available to one trade.

## Target and combination results

All values are paired OOF uplift versus the exact frozen policy, in net
bps/trade over 9,120 trades.

| Arm | Uplift | Positive months | Worst month | Positive weeks | Worst week | Winner retention |
|---|---:|---:|---:|---:|---:|---:|
| Remaining-H12 forecast | **+14.20** | 13/13 | **+0.15** | 45/58 | -35.41 | 96.5% |
| 50% next-hour / 50% remaining | +13.06 | 11/13 | -6.69 | 41/58 | -40.71 | 94.1% |
| 25% action rank / 75% forecast | +12.71 | 12/13 | -7.02 | 42/58 | -40.71 | 94.1% |
| State-action LambdaRank | +6.46 | 11/13 | -7.66 | 36/58 | -32.85 | 95.3% |
| Static activation level 0 | +5.93 | 10/13 | -10.85 | 32/58 | -174.99 | 76.4% |
| Static activation level 1 | +4.67 | 9/13 | -3.87 | 35/58 | -49.13 | 90.3% |
| Binary delta-Q >25 bps | +3.53 | 7/13 | -29.03 | 32/58 | -93.34 | 86.4% |
| Next-hour forecast only | +3.28 | 8/13 | -19.85 | 29/58 | -91.56 | 88.7% |
| Huber exact delta-Q | +2.83 | 7/13 | -23.87 | 38/58 | -93.34 | 90.4% |

The result is clear: the once-hourly controller should update its estimate of
remaining trade geometry, not optimize only the next hour. Combining the two
targets did not improve portability.

Forecast quality by year:

| Target | 2025 Spearman | 2025 MAE ATR | 2026 Spearman | 2026 MAE ATR |
|---|---:|---:|---:|---:|
| Next-hour favorable | 0.335 | 0.453 | 0.564 | 0.362 |
| Next-hour adverse | 0.346 | 0.545 | 0.593 | 0.371 |
| Remaining favorable | 0.640 | 1.552 | 0.191 | 3.254 |
| Remaining adverse | 0.615 | 1.613 | 0.686 | 1.243 |

Remaining favorable distance has the largest 2026 portability gap. This is the
principal model-risk item even though the paired policy uplift remains positive
in every 2026 month.

## Authority ablation

`shrinkage` blends the model-proposed activation back toward the frozen policy.
The `edge` gate requires a minimum score advantage before any intervention.
Only activation is adjusted in the best mode; the policy stop and fixed
giveback remain frozen.

| Authority | Pooled uplift | Adaptive EV | Positive months | Positive weeks | Worst week | Winner retention | Trades ever changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Activation only, shrink .75, edge .20 | +36.21 | +42.83 | 13/13 | 52/58 | -6.85 | 98.6% | 98.1% |
| Activation only, shrink .50, edge .20 | +33.46 | +40.08 | 13/13 | 53/58 | -4.23 | **99.9%** | 94.1% |
| Activation only, shrink .75, edge .30 | +29.83 | +36.45 | 13/13 | **54/58** | **0.00** | 99.6% | 90.8% |
| Activation only, shrink .75, edge .40 | +20.98 | +27.60 | 13/13 | **54/58** | **0.00** | 99.9% | 72.5% |
| Activation only, shrink .50, edge .30 | +20.17 | +26.79 | 13/13 | 52/58 | -8.44 | 99.8% | 79.6% |
| Activation only, shrink .50, edge .40 | +6.99 | +13.61 | 13/13 | 47/58 | approximately 0 | 99.9% | 51.4% |

An edge of 0.50 ATR or more disables the controller on this action domain. The
two credible authority candidates are:

- balanced: shrink 0.75, edge 0.40; and
- aggressive research: shrink 0.75, edge 0.30.

The balanced arm gives up about 8.85 bps/trade of pooled uplift but reduces
trade intervention by 18.2 percentage points. It is the safer candidate for a
future frozen replay. The 0.30 arm is useful as an upper-authority challenger.

Year split for the aggressive research arm:

| Year | Trades | Baseline EV | Adaptive EV | Uplift | Positive months | Worst month | Positive weeks |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 | 4,141 | +29.15 | +69.54 | +40.38 | 6/6 | +24.55 | 27/27 |
| 2026 | 4,979 | -12.12 | +8.93 | +21.05 | 7/7 | +0.86 | 27/31 |

This controller improves the very poor June 2026 baseline only by +0.86
bps/trade; it does not repair entry/admission quality in that month. Exit
adaptation cannot substitute for fixing a bad candidate population.

## Interpretation and next gate

The experiment works as an exit-policy overlay, with the following limits:

1. Use remaining-H12 favorable/adverse forecasts; retain next-hour forecasts as
   monitoring diagnostics, not the primary controller.
2. Give the model authority over trailing activation only. Keep the
   SimplePolicyOptimiser stop and giveback frozen.
3. Use shrinkage and an edge gate. Do not grant the raw forecast full authority.
4. Treat 2025 as development. Although 2026 was not used in this run's
   selection, it has been inspected elsewhere in the research program and is
   not an untouched final test.
5. Before promotion, replay the balanced and aggressive arms through the actual
   causal EV-admitted and portfolio-constrained stream, then freeze one and
   validate on a later untouched period.

## Reproduction

```bash
python3 scripts/run_hourly_exit_parameter_ablation.py \
  --out-dir data_perp/artifacts/hourly_exit_parameter_ablation_long_2025_2026_20260813_v2_effective_atr \
  --max-train-states 25000
```

Use `--resume` only to reproduce model/replay tables from the already
materialized causal state, path, and suffix-action ledgers. The authoritative
artifacts are under:

`data_perp/artifacts/hourly_exit_parameter_ablation_long_2025_2026_20260813_v2_effective_atr`

The earlier `..._v1` directory used raw ATR normalization and is invalidated for
economic interpretation.
