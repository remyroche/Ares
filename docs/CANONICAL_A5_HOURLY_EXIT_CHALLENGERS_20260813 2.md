# Canonical A5 hourly exit challengers — 2026-08-13

## Decision

No challenger is promoted by this experiment.

The current best downstream candidate remains the remaining-H12 favourable
excursion forecast using the compact 28-field causal path contract.  On the
complete canonical A5-admitted population it improves the frozen hourly policy
proxy by **+25.73 bps/trade in 2025 development** and **+12.91 bps/trade in
2026 confirmation**.  All 16 evaluated months have positive uplift.

This is strong research evidence, but not yet an executable replacement for
the frozen SimplePolicyOptimiser policy.  The adaptive outcome is a complete
one-hour conservative replay, and the portfolio comparison holds the
canonical accepted IDs fixed.  Promotion requires an exact 15-minute replay
and a capacity-aware portfolio rerun with adaptive exit timestamps.

## Canonical handoff retained

The upstream stack is unchanged and follows
`docs/TP6_SL4_BASE_CONSENSUS_CANONICAL_20260807.md`:

1. strict-R3 base;
2. prequential policy-net map;
3. ten conditional policy-residual heads;
4. 75/25 base-consensus blend;
5. frozen October-December 2024 Geometry/K9 bundle;
6. prior-28-day calibration and Cell-day trim-15 EV map;
7. A0 expected net at least +50 bps plus timestamp-local top 15%;
8. A5 bounded-10 reranking without changing membership;
9. canonical portfolio auction;
10. frozen SimplePolicyOptimiser exit fallback.

The exit fallback is:

- signal close plus one hour entry;
- stop: 4.152000643 ATR;
- trailing activation: 2.326224920 ATR;
- fixed giveback: 0.102371990 ATR;
- H12 timeout;
- 100 bps cost exactly once.

## Causal experiment contract

- Population: all 53,282 `a5_bounded10_admitted` long candidates.
- Source coverage: 53,282 / 53,282 complete 12x1h paths, 164 symbols.
- States: 411,585 incumbent-live completed-hour states.
- Controller clock: once per completed hour.
- Action timing: a decision observed after hourly bar `j` applies at bar
  `j+1`.
- Train window: up to nine preceding months, capped at 40,000 equally sampled
  state rows.
- Purge: 12 hours before each held block.
- Held blocks: three months.
- Development: April-December 2025. January-March are warm-up.
- Confirmation: January-July 2026.
- Direct action query: the five activation choices for the same causal trade
  state.
- Authority: trailing activation only. Stop and giveback remain frozen.

The exact canonical `policy_atr` is used before falling back to `atr_1h` and a
causal Wilder-14 hourly ATR.  On rows whose canonical stored outcome is itself
the hourly proxy, the rebuilt baseline has numerical parity: MAE approximately
`6e-15` bps, zero bias, Spearman 1.0.  Against existing exact/15-minute
outcomes, the one-hour proxy has 17.03 bps MAE and +7.26 bps bias.  That is a
source-resolution limitation, not a model gain.

## Features tested

### F0 — 14 core fields

Age, PnL/MFE/MAE in ATR, giveback, time since MFE/MAE, velocity,
acceleration, effective stop distance, distance to activation, trailing state,
and ATR fraction.

### F1 — 28 compact causal path fields

F0 plus:

- one-bar and two-bar return in ATR;
- return/range/realized volatility since the prior hourly decision;
- side-normalized close location and positive-bar fraction;
- path efficiency;
- MFE and MAE increments;
- drawup from MAE;
- new-MFE/new-MAE flags;
- fraction of H12 elapsed.

### F2 — 43 rich path fields

F1 plus signed path efficiency, excursion asymmetry, MFE/MAE persistence and
slopes, sign-change frequency, excursion recency interactions, and
age-weighted excursions.

### F3 — 82 path/archetype/entry-context fields

F2 plus stable continuous trade archetypes (underwater, recovery, giveback,
stall, fresh extension, adverse pressure, trailing giveback) and entry-time
base/consensus/A0/A4/A5/support/OOD/Geometry-K9 aggregate context.  No raw K9
posterior coordinate is used.

### F4 — 127 path/context/evolution fields

F3 plus current, entry, delta-1h, acceleration-1h, and change-since-entry
versions of base score/rank, consensus rank, correctness/residual rank, and
upstream score.  Current values are attached by backward as-of joins to the
strict-prequential score ledger.

## Targets and decision rules tested

1. Incumbent remaining-H12 forecasts: q65 favourable and q80 adverse.  Under
   activation-only authority the historical rule effectively uses favourable
   distance; adverse distance is constant across allowed actions.
2. 50/50 next-hour and remaining-H12 geometry.
3. Explicit risk-adjusted remaining target:
   `favourable - lambda * adverse`, lambda 0.25, 0.50, 1.00.
4. State-action LambdaRank on ordinal delta-Q.
5. Huber regression on exact state-action delta-Q.
6. Dual binary action target:
   `P(delta-Q > +25 bps) - lambda * P(delta-Q < -25 bps)`, lambda 0.5, 1, 2.
7. Activation-only shrink 0.75 with rank-edge 0.30 and 0.40.

The two edge settings are identical on the five-action grid because action
ranks move in 0.25 steps: both thresholds require the same next attainable
edge.  A future authority test should use continuous ATR/bps distance, not
action-rank distance.

## Main all-admitted OOF results

| Arm | 2025 EV | 2025 uplift | 2026 EV | 2026 uplift | Positive months | Worst month uplift | Positive weeks | Worst week uplift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 remaining-H12 incumbent | +140.08 | **+25.73** | +172.36 | **+12.91** | **16/16** | +0.77 | 63/68 | -3.26 |
| F0 remaining-H12 | +139.76 | +25.41 | +172.33 | +12.88 | 16/16 | +0.77 | 63/68 | -3.26 |
| F2 rich-path remaining-H12 | +139.59 | +25.24 | +172.65 | +13.20 | 16/16 | +0.77 | 63/68 | -3.26 |
| F4 risk-adjusted lambda .25 | +139.60 | +25.25 | +170.86 | +11.40 | 16/16 | +0.77 | 61/68 | -4.43 |
| Prequential dual classifier lambda 2 | +139.18 | +24.83 | +170.22 | +10.77 | 15/16 | -12.66 | 59/68 | -28.73 |
| F4 remaining-H12 | +138.56 | +24.21 | +172.15 | +12.70 | 16/16 | +0.99 | 63/68 | **-0.28** |
| F1 50/50 next-hour/remaining | +137.70 | +23.35 | +172.31 | +12.85 | 15/16 | -0.30 | 62/68 | -4.43 |

The hourly baseline itself is +114.35 bps/trade in 2025 and +159.46 in 2026
over the same admitted rows.  These are coarse-policy values, not the canonical
mixed exact/15-minute headline.

## Fixed canonical portfolio-ID diagnostic

This keeps the IDs accepted by the canonical auction and changes only their
exit outcome.  It does not yet release/reallocate capacity at the adaptive
exit timestamp.

| Arm | 2025 baseline / adaptive | 2025 uplift | 2026 baseline / adaptive | 2026 uplift | Worst month uplift |
|---|---:|---:|---:|---:|---:|
| F1 remaining-H12 incumbent | 130.40 / 155.05 | **+24.65** | 168.99 / 185.84 | **+16.85** | +3.31 |
| F0 remaining-H12 | 130.40 / 154.86 | +24.46 | 168.99 / 185.90 | +16.90 | +3.51 |
| F2 rich-path remaining-H12 | 130.40 / 154.74 | +24.34 | 168.99 / 185.84 | +16.85 | +3.56 |
| F4 remaining-H12 | 130.40 / 153.34 | +22.94 | 168.99 / 185.87 | +16.87 | +3.17 |
| F1 50/50 | 130.40 / 152.44 | +22.04 | 168.99 / 186.20 | **+17.21** | +0.71 |

## Monthly incumbent results

| Month | Trades | Baseline net | Adaptive net | Uplift |
|---|---:|---:|---:|---:|
| 2025-04 | 2,305 | +225.83 | +241.63 | +15.80 |
| 2025-05 | 5,033 | +66.86 | +106.70 | +39.83 |
| 2025-06 | 1,268 | +14.41 | +47.08 | +32.67 |
| 2025-07 | 4,269 | +107.62 | +135.53 | +27.90 |
| 2025-08 | 4,184 | +97.99 | +126.45 | +28.46 |
| 2025-09 | 3,825 | +78.71 | +92.73 | +14.01 |
| 2025-10 | 3,787 | +191.09 | +207.22 | +16.13 |
| 2025-11 | 1,543 | +144.30 | +190.34 | +46.04 |
| 2025-12 | 2,410 | +116.56 | +130.78 | +14.22 |
| 2026-01 | 1,566 | +232.71 | +244.78 | +12.06 |
| 2026-02 | 1,040 | +306.66 | +336.78 | +30.11 |
| 2026-03 | 3,383 | +111.96 | +125.01 | +13.06 |
| 2026-04 | 2,749 | +160.30 | +176.89 | +16.58 |
| 2026-05 | 3,941 | +127.50 | +134.54 | +7.04 |
| 2026-06 | 180 | +200.55 | +201.33 | +0.77 |
| 2026-07 | 176 | +210.87 | +213.31 | +2.45 |

June and July 2026 have low admitted support and must not dominate selection.

## Feature learnability versus economic utility

| Contract | Fields | 2025 mean forecast IC | 2026 mean forecast IC | 2025 policy uplift | 2026 policy uplift |
|---|---:|---:|---:|---:|---:|
| F0 | 14 | 0.507 | 0.481 | +25.41 | +12.88 |
| F1 | 28 | 0.511 | 0.482 | **+25.73** | +12.91 |
| F2 | 43 | 0.511 | 0.482 | +25.24 | **+13.20** |
| F3 | 82 | 0.502 | 0.479 | +23.92 | +12.48 |
| F4 | 127 | **0.522** | **0.483** | +24.21 | +12.70 |

F4 learns the excursion labels best but does not convert that gain into the
best policy uplift.  Context should therefore be tested as a separate
reliability/veto/shrink mechanism, not assumed useful merely because it raises
target IC.

## Recommended next funnel

1. **Exact-resolution gate.** Rebuild the same states/actions from complete
   15-minute histories on identical IDs.  Require positive paired uplift and
   at least 0.8 rank correlation of per-trade improvement versus the hourly
   proxy.
2. **Capacity-aware gate.** Feed adaptive exit timestamps into the canonical
   auction so freed capacity, concurrency, asset limits, and wallet PnL are
   recomputed rather than holding accepted IDs fixed.
3. **Context as shrinker.** Keep the F1 opportunity forecast.  Train a small
   causal gate on F4-only context to predict whether F1's intervention will
   disappoint.  Test consensus-only intervention, disagreement abstention,
   and continuous shrink.  Do not let this gate alter A5 admission.
4. **Stable archetypes.** Replace hand-built archetype interactions with
   training-only binned causal states over PnL/MFE/MAE/giveback/age/trailing
   status.  Estimate action uplift with hierarchical shrinkage and feed only
   stable role outputs: expected uplift, downside probability, support, and
   uncertainty.  Never expose fold-local cluster IDs.
5. **Continuous authority.** Predict a continuous activation ATR and shrink it
   toward 2.326 ATR using uncertainty/support.  Test asymmetric bounds:
   earlier activation may receive more authority than delayed activation, or
   vice versa, but selection must be 2025-only.
6. **Tail-risk target.** The direct dual classifier did not advance.  A final
   target attempt should use distributional remaining favourable excursion
   with lower-quantile downside or expected shortfall, not another exact
   delta-Q regressor.
7. **Leave stop and giveback frozen.** Nothing in this experiment supports
   granting the hourly controller authority over those components.

## Artifacts and reproduction

Runner:

`scripts/run_canonical_a5_hourly_exit_challengers.py`

Command:

```bash
python3 scripts/run_canonical_a5_hourly_exit_challengers.py \
  --out-dir data_perp/artifacts/canonical_a5_hourly_exit_challengers_20260813_v2 \
  --max-train-states 40000
```

Authoritative output:

`data_perp/artifacts/canonical_a5_hourly_exit_challengers_20260813_v2`

Important files:

- `run_manifest.json`
- `replay_rows.parquet`
- `causal_hourly_states.parquet`
- `activation_action_delta_q.npz`
- `oof_forecast_predictions.parquet`
- `oof_adaptive_replay.parquet`
- `metrics.parquet`
- `feature_arm_audit.parquet`
- `hourly_path_coverage.parquet`

The earlier `canonical_a5_hourly_exit_challengers_20260813_v1` used `atr_1h`
instead of the canonical `policy_atr` and is superseded.

Focused verification: 45 tests passed across frozen-policy labels, causal path
state/action replay, purged folds, and continuation targets.
