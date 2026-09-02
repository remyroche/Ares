# Path-Based Adaptive Exit Optimisation — corrected v2 handover

## Decision

The diagnostic and learner now use the shared specification's actual path
clock and exact state families.  The research result is **not promoted**.  The
best matched weighting arm improves mean net outcome and left-tail risk, but it
fails the required temporal-portability gates.

The frozen `simple_policy_optimiser` policy therefore remains canonical.

The final full 12,000-row/24-permutation oracle and learnability audit,
including zero-inflation-aware CMI for velocity, acceleration, and change
fields, is documented in `docs/PATH_BASED_EXIT_ORACLE_DIAGNOSTIC_20260813.md`.
It supersedes the earlier 4,000-row/6-permutation diagnostic for learnability
conclusions.  The exact 625-action oracle economics are unchanged.

The earlier v1 state-level results in this document are withdrawn.  They
labelled a decision at a 15-minute bar's start while consuming that bar's high,
low, and close, and they included the incumbent exit bar.  The corrected v2
contract emits a state after a bar is complete, applies an action from the next
bar, and excludes the incumbent exit bar from the decision ledger.

## Placement and frozen baseline

```text
causal long-side admission and entry
  -> frozen global SimplePolicyOptimiser exit
  -> research-only path-state adaptive exit overlay
```

The overlay cannot alter admission, entry, side, size, or cost.  It may tighten
the stop or change a still-live trailing rule.  It cannot widen the stop,
deactivate an active trail, or loosen an earned trail lock.

The frozen baseline is strategy `long_s52_meta_threshold_handoff` from:

`data_perp/artifacts/s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/simple_policy_optimiser/deployment/best_policy_params.json`

- stop: 4.0 ATR;
- trailing activation: 2.294102 ATR, with the frozen decay contract;
- trailing power: 2.454087;
- squash divisor: 4.521387;
- giveback beta: 0.456996;
- 15-minute path, H12 timeout;
- 100 bps round-trip cost, applied once.

## Corrected decision and action contract

Each decision uses a completed 15-minute observation.  The earliest path
decision is therefore 15 minutes after entry.  The chosen action is installed
for the following bar.  A state whose observation bar is the incumbent exit bar
is never emitted.

There are five levels for each of four components, or 625 joint actions:

- stop: incumbent through 50% of incumbent distance; tightening only;
- activation: 0.50, 0.75, 1.00, 1.25, 1.50 times incumbent;
- trailing power: 0.50, 0.75, 1.00, 1.25, 1.50 times incumbent;
- giveback: 0.50, 0.75, 1.00, 1.25, 1.50 times incumbent.

The no-change action is `s0_a2_p2_g2`.  The target is the paired suffix value

```text
Delta-Q(state, action)
  = net outcome after installing action on the next bar
  - incumbent suffix net outcome
```

All 625 actions are replayed on each identical observed suffix.  The baseline
target is exactly zero.

## Implemented feature contract

The completed state ledger exposes 149 causal candidate fields.  CMI may remove
any of them inside each training fold; there are no protected feature tiers.

### Entry-frozen context

- base score/rank/anchor, consensus/correctness/final score, and upstream;
- causal 21-day EV/probability/support plus 42/84-day support;
- rule, path, model, leaf, and K9 support/OOD/drift fields;
- contribution-weighted support and Mahalanobis, PSI, and KS diagnostics;
- timestamp-local K9 support/OOD and covariance/correlation breaks;
- K9 entropy and top-two margin;
- entry-frozen hourly trend, momentum, volatility, volatility-of-volatility,
  VWAP, and volatility-regime state.

### Evolving path state

- trade age;
- PnL, MFE, MAE, and drawdown from MFE in bps and ATR;
- fraction of MFE given back;
- time since and time to MFE, plus new-MFE state;
- wall-clock PnL velocity and acceleration;
- effective stop distance, incumbent stop distance, and trailing-active state.

### Evolving hourly context

- short/medium/long ATR-normalised trend and momentum;
- cross-horizon differentials, acceleration, and agreement;
- short/medium/long realised volatility, ratio, and acceleration;
- volatility-of-volatility level, percentile, state, change, and acceleration;
- short/medium/long VWAP distance, slope, alignment, and change;
- volatility-regime code;
- change since entry for each hourly field;
- latest completed hourly snapshot age and source-completeness state.

Hourly snapshots are built from complete 15-minute bars, published only after
all four bars of an hour complete, then joined backward with
`available_at <= decision_ts`.  Appending future bars cannot alter an earlier
state.

Features that were invented in the superseded draft, rather than declared by
the shared specification, were removed.  The exact materialised list is the
149 columns returned by `_adaptive_feature_columns` and stored in the state
ledger.

## CMI, interaction, and Bayesian contracts

- exactly five weighted bins, fitted on the training fold only;
- 16 one-component action contrasts;
- all six action-component pairs audited for synergy;
- at most two-way state interactions;
- circular 12-hour block-preserving permutations within month;
- environment-specific effect profiles and stability checks;
- sparse graph with bounded degree;
- robust Huber IRLS fit with ridge and effective-support shrinkage;
- diagonal Laplace posterior approximation;
- posterior uncertainty, action-distance, and action-change penalties;
- strict chronological OOF folds with a 12-hour purge;
- all states from one trade remain in one fold.

The six pair audits use a bounded training-only representative level contract,
selected by weighted absolute synergy, so the diagnostic remains computationally
tractable without silently dropping an action pair.

## Data and validity

- long side only;
- 3,000 causally admitted candidates sampled evenly by month before path
  inspection;
- 2,258 complete 15-minute paths (75.27% outcome coverage);
- 70,609 corrected path-decision states;
- 1,435 OOF trades in three supported expanding folds;
- baseline replay maximum absolute parity error: 0.000186 bps;
- outcome completeness never participates in upstream admission.

This remains outcome-covered OOF research evidence, not a complete portfolio
replay or untouched promotion period.

## Diagnostic oracle opportunity

| Ledger | Rows | Mean regret | Median | P90 | P95 | Positive | >=50 bps | 1-component capture | 2-component capture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Entry 625 | 2,258 | 79.12 | 0.00 | 239.32 | 392.32 | 45.84% | 36.23% | 87.75% | 99.00% |
| Path-state 625 | 70,609 | 60.98 | 1.74 | 210.21 | 327.06 | 56.18% | 28.70% | 88.54% | 98.64% |

The state oracle ceilings are:

| Allowed change | Mean oracle improvement |
|---|---:|
| Stop only | +26.42 bps |
| Activation only | +24.02 bps |
| Power only | +5.86 bps |
| Giveback only | +8.46 bps |
| Best one component | +53.99 bps |
| Best two components | +60.15 bps |
| Best three/four components | +60.98 bps |

Two components capture virtually all oracle opportunity.  A maximum normalised
intervention magnitude of 0.50 captures +60.32 of the +60.98 bps ceiling.  The
evidence argues for a smaller downside-focused controller, not a more complex
four-component learner.

Opportunity attribution:

| Opportunity type | States | Mean regret | Share of total regret |
|---|---:|---:|---:|
| Left-tail loss prevention | 16,004 | +132.64 bps | 49.30% |
| Winner giveback prevention | 8,225 | +106.59 bps | 20.36% |
| Winner preservation/extension | 46,380 | +28.16 bps | 30.33% |

Thus 69.66% of the oracle opportunity is loss or giveback prevention.

The opportunity/learnability map contains 13 `MODEL_HIGH_VALUE`, 1,364
`MODEL_WITH_STRONG_SHRINKAGE`, 941 `ABSTAIN`, and 212 `IGNORE` cells.  Large
oracle regret alone is common; high regret, stable CMI, and adequate support is
rare.

## Corrected static A-H ablation

Static entry actions are diagnostic and cannot promote a dynamic policy.

| Arm | Definition | Uplift | Positive months | Worst month | Month MAD | Trade CVaR5 improvement | Winner retention |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Frozen baseline | 0.00 | 0.0% | 0.00 | 0.00 | 0.00 | 100.00% |
| B | Train-best static joint action | +2.99 | 33.3% | -32.46 | 3.64 | +118.72 | 91.60% |
| C | Raw unshrunk surface | +5.07 | 55.6% | -31.46 | 27.14 | +241.79 | 89.16% |
| D | Bayesian action main only | 0.00 | 0.0% | 0.00 | 0.00 | 0.00 | 100.00% |
| E | Bayesian state main | +1.15 | 44.4% | -25.69 | 12.76 | +181.39 | 92.18% |
| F | State interactions | +5.56 | 55.6% | **-21.01** | 13.52 | +199.14 | **92.15%** |
| G | Action-pair main only | 0.00 | 0.0% | 0.00 | 0.00 | 0.00 | 100.00% |
| H | Full sparse CMI/Bayesian | **+6.29** | 55.6% | -31.39 | 19.40 | **+229.24** | 90.62% |

## Exact state-suffix weighting ablation

| Training weight | Uplift | Positive months | Worst month | Month MAD | Positive weeks | Worst week | Trade CVaR5 improvement | Winner retention | Portable-oracle capture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw state | **+7.77** | 55.6% | -25.54 | 17.01 | 45.7% | -99.51 | +225.27 | 90.39% | **12.28%** |
| Trade balanced | +4.67 | 55.6% | -33.39 | 13.37 | 45.7% | -105.53 | **+233.36** | 89.74% | 7.38% |
| Trade + 12h block balanced | +2.40 | 44.4% | **-24.83** | **8.46** | 45.7% | **-90.44** | +219.11 | **90.49%** | 3.79% |

Raw weighting has the best mean and oracle capture; block balancing is the least
unstable.  None passes the portability gate.  Raw weighting also changes 94.1%
of trades, which is too broad for a trustworthy overlay.

Raw-weight monthly uplift:

| Month | Uplift bps/trade |
|---|---:|
| 2025-06 | +8.44 |
| 2025-07 | +25.45 |
| 2026-01 | -15.40 |
| 2026-02 | +55.97 |
| 2026-03 | -13.14 |
| 2026-04 | -25.54 |
| 2026-05 | +14.70 |
| 2026-06 | 0.00 (7 trades) |
| 2026-07 | +12.78 |

The best model improves total summed outcome by 11,152.57 bps and improves
maximum drawdown by 4,019.31 summed bps, Sortino by 0.0304, trade CVaR5 by
225.27 bps, and daily CVaR5 by 2,366.61 bps.  Those aggregate gains do not
override the negative January/March/April and weekly instability.

## Portable signal findings

The most recurrent training-fold state information includes MFE in ATR/bps,
ATR fraction, entry K9 timestamp correlation break, entry medium-horizon VWAP
distance, entry momentum short-minus-long and acceleration, entry trend level
and acceleration, volatility-of-volatility change/acceleration, K9-weighted
OOD/distance, and model PSI drift.

The action-specific pair audit finds stable information chiefly for
stop-by-activation.  Stop plus activation is also consistent with the nested
oracle ceiling.  Pair-level semantics are not promoted automatically.

## Decision and next experiment

Promotion gates require at least +5 bps/trade, at least 60% positive months,
worst month no worse than -10 bps/trade, winner retention at least 85%, and no
trade-CVaR deterioration.  No sequential arm passes all gates.

Decision: `KEEP_FROZEN_BASELINE`.

The next bounded experiment should:

1. restrict the first learned action space to stop and activation; this is the
   strongest fixed pair at 75.33% of full state-oracle headroom, while the
   98.64% best-two-component ceiling requires state-wise pair selection;
2. cap normalised action distance at 0.50;
3. train a loss/giveback-risk gate that abstains outside supported portable
   regions;
4. reduce the intervention rate materially;
5. preserve the exact v2 decision clock and rerun on an untouched period.

No model or configuration from this analysis is canonical without explicit
approval.

## Reusable commands and artifacts

Corrected base/state replay:

```bash
/usr/local/bin/python3 scripts/run_path_based_exit_portability_validation.py \
  --max-trades 3000 \
  --out-dir data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260812_v2_exact_contract \
  --resume
```

Oracle diagnostic:

```bash
/usr/local/bin/python3 scripts/run_path_based_exit_oracle_diagnostic.py \
  --run-dir data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260812_v2_exact_contract \
  --output-dir data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260812_v2_exact_contract/oracle_diagnostic_v10
```

Matched weight arm, reusing the materialised ledger:

```bash
/usr/local/bin/python3 scripts/run_path_based_exit_portability_validation.py \
  --suffix-only \
  --state-weight-mode raw_state \
  --out-dir data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260813_v3_weight_raw
```

Relevant code:

- `extreme_price_movements/path_based_exit_optimisation.py`;
- `scripts/run_path_based_exit_portability_validation.py`;
- `scripts/run_path_based_exit_oracle_diagnostic.py`;
- `tests/test_path_based_exit_optimisation.py`.

Primary corrected artifacts:

- `data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260812_v2_exact_contract/`;
- `data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260812_v2_exact_contract/oracle_diagnostic_v10/`;
- `data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260813_v3_action_cmi/`;
- `data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260813_v3_weight_raw/`;
- `data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260813_v3_weight_trade/`.
