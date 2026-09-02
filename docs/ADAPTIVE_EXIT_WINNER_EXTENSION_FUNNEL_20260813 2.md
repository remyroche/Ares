# Adaptive Exit Winner-Extension Funnel — 2026-08-13

## Decision

Adaptive Exit V1 remains canonical. No winner-extension challenger is promoted.

The activation-only W1 branch finds a real but extremely sparse OOF edge. Its
best development threshold improves candidate-level net EV by only +0.081
bps/trade across the matched exact-15-minute population, changes 25 economically
material exits out of 34,469 trades, and makes no material change from April
through July 2026. The action-value heads are not sufficiently learnable or
portable to justify opening the giveback/power action lattice.

## Frozen contract and causal protocol

- Long side only.
- Frozen Adaptive Exit V1 reconstructed with the canonical 40,000-state cap.
- Entry and exit paths use the canonical complete 15-minute source.
- Decisions are made only after a completed hourly bar and become effective on
  the next 15-minute bar.
- Stop, giveback, power, timeout, costs, and fill rules remain frozen in W1.
- Cost is applied exactly once under the incumbent replay contract.
- Chronological OOF folds use resolved earlier labels with a 12-hour purge.
- Winner eligibility uses only state available at the decision timestamp.
- Activation heads are trained only before trailing activation, because a later
  activation threshold cannot undo an already armed/protected trailing state.
- Feature screening is performed on a training-only 4,000-state subsample.
- Each shallow head uses at most 20,000 equal-month training states and an inner
  chronological early-stopping split.
- 2025 is development; 2026 is confirmation. Neither is claimed untouched.
- Promotion is disabled.

V1 replay parity is exact for practical purposes:

| Check | Result |
|---|---:|
| Paired OOF rows | 26,524 |
| Maximum absolute net difference | 0.00112 bps |
| Exit-bar match | 100.0% |

## V1 winner decomposition

The decomposition confirms substantial winner-side headroom, but also shows
why a safe extension rule is difficult: large continuation and large adverse
movement coexist in a meaningful part of the population.

| Phenotype | Trades | Share | Median capture gap | Median post-exit continuation | P90 continuation | Median post-exit adverse |
|---|---:|---:|---:|---:|---:|---:|
| Good / low-regret exit | 13,576 | 39.39% | 0.00 bps | 0.00 ATR | 0.82 ATR | 0.00 ATR |
| Necessary protection | 4,310 | 12.50% | 9.16 bps | 0.00 ATR | 0.31 ATR | 3.59 ATR |
| Excessive giveback | 10,699 | 31.04% | 262.79 bps | 0.17 ATR | 5.57 ATR | 0.86 ATR |
| Premature smooth continuation | 5,222 | 15.15% | 206.78 bps | 3.11 ATR | 12.87 ATR | 0.00 ATR |
| Winner degraded versus baseline | 662 | 1.92% | 195.70 bps | 2.70 ATR | 6.32 ATR | 0.17 ATR |

Robust medians and P90s are reported because a small number of low-ATR assets
make the arithmetic mean of ATR-normalized post-exit moves unstable.

## Heads and features

The experiment fits:

1. `p_new_mfe_1h`: probability of a new MFE at least 0.25 ATR higher within one hour.
2. `p_reversal`: probability of at least 0.50 ATR adverse movement before that new MFE.
3. `p_loss_veto`: probability that the frozen V1 trade ultimately finishes below -100 bps.
4. Per-action Huber mean and 25% lower-quantile value heads.
5. Per-action sparse useful classifiers for gain above +25 bps.
6. Per-action economically weighted sign classifiers. Their binary target is
   `extension_gain_bps > 0`; sample weights are `clip(abs(extension_gain_bps),
   5, 500)`. This makes costly false extensions matter much more than inert ties.

The fold-local CMI screen retains 30 features. Twenty-seven persist in every
fold:

- trade age and fraction of horizon elapsed;
- new MFE/MAE flags and increments;
- time since MFE and MAE;
- MFE/MAE ordering and MFE slope;
- 15-minute return and path close location;
- positive-bar fraction, signed path efficiency, and one-hour sign-change rate;
- MFE persistence within 0.25/0.50 ATR;
- near-MFE time within 0.10/0.25/0.50 ATR;
- new-MFE counts over 15/30/60 minutes;
- failed MFE retake, lower-high, and recovery-from-recent-low fields.

Across folds, 32 fields appear at least once and 27 appear in all folds. Target,
future-path, proposal-gain, and outcome fields are excluded from inference
features.

## Head learnability

| Head | OOF ROC-AUC | OOF PR-AUC | Prevalence / note |
|---|---:|---:|---:|
| New MFE within 1h | 0.767 | 0.314 | 14.10% |
| Reversal before next MFE | 0.611 | 0.468 | 38.54% |
| Frozen-V1 loss veto | 0.609 | 0.468 | 24.39% |
| Mild action positive | 0.544 | 0.120 | 10.50% |
| Substantial action positive | 0.547 | 0.142 | 11.69% |

Continuation is learnable. Actual action usefulness is not: the top 1% of the
standalone economic action scores has negative realised action value (-9.0 bps
for mild and -50.4 bps for substantial). Positive W1 economics arise only from
the intersection with continuation, reversal, and loss-veto gates.

The state-level activation oracle is sparse but non-trivial: 5.81% of actionable
pre-activation OOF states have more than 0.01 bps of best-action headroom, 4.00%
have more than +25 bps, and the mean best achievable gain is +5.89 bps/state.
This is opportunity, not evidence that it can be selected causally.

## W1 matched results

All numbers below are exact-15-minute candidate-level net bps/trade against the
same reconstructed V1 population. They are not portfolio-auction metrics.

| Arm | Rule | All | 2025 | 2026 | Materially changed trades |
|---|---|---:|---:|---:|---:|
| W0 | Frozen V1 | 135.612 | 132.313 | 146.540 | 0 |
| W1 mean/LCB | continuation + reversal/loss veto + positive mean/LCB | 135.612 | 132.313 | 146.540 | 0 |
| W1 useful p10–p30 | gain-above-25 classifier | 135.612 | 132.313 | 146.540 | 0 |
| W1 economic p20 | weighted gain-sign + loose three-head veto | **135.692** | **132.380** | **146.667** | **25** |
| W1 economic p25 | same, stricter | 135.643 | 132.347 | 146.562 | 12 |
| W1 economic p30 | same, stricter | 135.638 | 132.330 | 146.596 | 4 |
| W1 economic p40/p50 | same, strictest | 135.612 | 132.313 | 146.540 | 0 |

For W1 economic p20:

- uplift: +0.081 bps/trade globally, +0.067 in 2025, +0.128 in 2026;
- materially changed exits: 25 (21 beneficial, 4 harmful);
- mean gain on changed exits: +111.29 bps; median +46.94 bps;
- best/worst changed exit: +787.13 / -610.58 bps;
- candidate bootstrap 95% interval for average uplift: +0.006 to +0.166 bps;
- CVaR05 and CVaR02 are unchanged at reported precision;
- 8 months show more than +0.01 bps/trade uplift, none show less than -0.01,
  and 11 are effectively unchanged;
- 12 weeks are positive, 3 negative, and 65 effectively unchanged;
- no material exit changes occur from April through July 2026.

## Why W2–W6 were not run

The attached funnel says to add giveback or power only after activation
relaxation produces portable MFE improvement. W1 does not meet that economic
standard:

- uplift is less than one tenth of one basis point per trade;
- action-value ranking is weak and reverses in its raw top-score tail;
- the result depends on only 25 changed exits;
- the latest four 2026 months receive no changed exits;
- a single February 2026 loss is -610.58 bps and produces a -6.29 bps/trade
  weekly uplift in that week;
- fixed-gap execution makes the power component behaviorally inactive anyway.

Opening a larger giveback/joint-action lattice at this point would increase
selection risk without evidence that the causal action-value problem is solved.

## Reusable command

```bash
python3 scripts/run_adaptive_exit_winner_extension_round1.py \
  --out-dir data_perp/artifacts/adaptive_exit_winner_extension_round1_20260813_v9 \
  --resume
```

The expensive exact-path labels and reconstructed V1 ledger are checkpointed,
so model/gate revisions can resume without rematerialising the path substrate.

## Artifacts

- `data_perp/artifacts/adaptive_exit_winner_extension_round1_20260813_v9/run_manifest.json`
- `data_perp/artifacts/adaptive_exit_winner_extension_round1_20260813_v9/v1_decision_ledger.parquet`
- `data_perp/artifacts/adaptive_exit_winner_extension_round1_20260813_v9/v1_feature_contracts.parquet`
- `data_perp/artifacts/adaptive_exit_winner_extension_round1_20260813_v9/v1_winner_decomposition.parquet`
- `data_perp/artifacts/adaptive_exit_winner_extension_round1_20260813_v9/v1_winner_decomposition_summary.parquet`
- `data_perp/artifacts/adaptive_exit_winner_extension_round1_20260813_v9/winner_extension_labels.parquet`
- `data_perp/artifacts/adaptive_exit_winner_extension_round1_20260813_v9/winner_extension_feature_cmi.parquet`
- `data_perp/artifacts/adaptive_exit_winner_extension_round1_20260813_v9/winner_extension_fit_audit.parquet`
- `data_perp/artifacts/adaptive_exit_winner_extension_round1_20260813_v9/winner_extension_oof_predictions.parquet`
- `data_perp/artifacts/adaptive_exit_winner_extension_round1_20260813_v9/winner_extension_candidate_replay.parquet`
- `data_perp/artifacts/adaptive_exit_winner_extension_round1_20260813_v9/winner_extension_metrics.parquet`

## Terminal classification

`WINNER_EXTENSION_ACTIVATION_EDGE_REAL_BUT_TOO_SPARSE`

`ACTION_VALUE_LEARNABILITY_FAILURE`

`W2_W6_NOT_JUSTIFIED`

`ADAPTIVE_EXIT_V1_REMAINS_CANONICAL`
