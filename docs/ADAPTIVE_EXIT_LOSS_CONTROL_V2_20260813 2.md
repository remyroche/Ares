# Adaptive Exit V1 decomposition and loss-control V2 ablations

Date: 2026-08-13  
Status: completed, no challenger promoted

## Decision

Adaptive Exit V1 remains frozen and canonical. The V2 work found real, causally
learnable failure risk and a small portfolio-level safety/EV opportunity, but no
challenger is yet portable enough to replace V1 without explicit approval.

The most interesting challengers are:

- `action_value_gt10`: best aggregate portfolio result and worst-week result,
  but it intervenes only in April--June 2025 and does nothing in 2026.
- `haz100_t90_sl3p0`: weaker aggregate uplift, but the cleanest portable rule;
  it is approximately EV-neutral/positive in both 2025 and 2026 and retains
  essentially all winners.
- `haz100_t85_sl3p0`: more active and improves 2026 tail loss, but has noisier
  monthly transport and a higher false-defense cost.

## Frozen incumbent

| Metric | Frozen baseline | Adaptive Exit V1 | Delta |
|---|---:|---:|---:|
| Portfolio trades | 8,453 | 8,622 | +169 |
| Trades/day | 14.72 | 15.01 | +0.29 |
| Net EV/trade | +163.09 bps | +189.36 bps | **+26.27 bps** |
| Sortino | 0.465 | 0.548 | +0.083 |
| Worst week | -35.09% | -19.28% | **+15.81 pp** |
| Max drawdown | -76.53% | -76.53% | unchanged |

V1 is an activation-only adaptive layer in this matched replay. Stop, trailing
power, and giveback remain frozen. Its OOF population uses the stored hourly
path proxy; fine-path and hourly outcomes differ by about 21.9 bps on their
overlap, so these results are internally matched to the incumbent but are not a
new fine-path reconciliation.

## Where V1's uplift comes from

### Baseline-outcome buckets

| Baseline bucket | Prevalence | Conditional delta | Unconditional contribution |
|---|---:|---:|---:|
| Catastrophic loser | 5.41% | +126.78 bps | +6.86 bps |
| Severe loser | 10.75% | +78.44 bps | +8.44 bps |
| Ordinary loser | 21.90% | +30.97 bps | +6.78 bps |
| Near-flat | 3.12% | +21.74 bps | +0.68 bps |
| Ordinary winner | 20.07% | +4.29 bps | +0.86 bps |
| Large winner | 25.64% | +0.24 bps | +0.06 bps |
| Extreme winner | 13.11% | -4.38 bps | -0.57 bps |

The paired candidate-level decomposition sums to about +23.10 bps/trade. The
portfolio uplift is larger because changed exit times release capacity and alter
which later candidates the auction accepts.

### Mechanisms

| Mechanism | Unconditional contribution |
|---|---:|
| Baseline loser becomes V1 winner | **+21.49 bps** |
| Both winners, V1 better | +5.32 bps |
| Both losers, V1 less bad | +1.00 bps |
| Both winners, V1 worse | -3.53 bps |
| Baseline winner becomes V1 loser | -1.15 bps |
| Both losers, V1 worse | -0.03 bps |

V1 is therefore already primarily a loss-side/conversion repair, not merely a
take-profit optimizer.

For baseline losers, mean loss improves from -252.33 to -195.51 bps; CVaR5
improves from -1,021.46 to -996.97 bps; severe-loss frequency falls from 40.88%
to 34.25%, and catastrophic-loss frequency from 13.68% to 11.79%.

For baseline winners, mean outcome rises only from +375.79 to +376.84 bps.
Mean MFE capture is essentially unchanged (0.8471 versus 0.8467), and 19.8% of
baseline winners are degraded. That is the principal false-defense constraint.

## Common max-drawdown episode

The shared -76.53% drawdown ran from 2025-01-31 19:00 UTC to 2025-02-03 02:00
UTC and recovered on 2025-02-04 15:00 UTC. It involved 58 active trades across
41 assets, all long, with 7.06 mean and 8 maximum concurrent positions.

No V1 OOF state prediction exists during this warm-up episode. Consequently,
unchanged max drawdown does not show that V1 or V2 failed to control a scored
exit. It is classified as `PORTFOLIO_ENTRY_OR_WARMUP_CONTROLLABLE`. All V2 arms
inherit this same max drawdown by construction.

## V2 design and lineage

The V2 state ledger contains 392,494 counterfactual states and 313,714 held OOF
predictions. There are six chronological folds; training uses nine months with a
12-hour purge, up to 40,000 equal-month sampled states, and no held-period
threshold fitting.

Each fold performs training-only conditional-MI screening and retains 40 fields.
The core action heads also use a fixed 28-field short-horizon path contract.

Models are shallow LightGBM heads with:

- at most 500 trees and early stopping;
- learning rate 0.03;
- depth 4, 15 leaves;
- minimum leaf support max(1% of sampled rows, 100);
- row and feature fractions 0.75;
- L2 = 25;
- binary log-loss, Huber, or 25th-quantile loss by target.

Targets include final loss below -50/-100/-200 bps, next-hour MAE
deterioration, scalar defensive gain, direct usefulness (`gain > 25 bps`), and
mean/25th-quantile incremental gains for tighter stops at 3.5, 3.0, 2.5, and
2.0 ATR versus V1. The original stop is approximately 4.152 ATR.

The market-shock threshold is a shifted prior-28-day timestamp-level quantile.
The completed OOF ledger is never used to define it. The hard OOD percentile
gate was removed because it also depended on the completed ledger; OOD remains
an ordinary fold-fitted model input.

## A--K portfolio comparison

| Arm | Net EV/trade | Delta vs V1 | Sortino | Worst week | Delta worst week |
|---|---:|---:|---:|---:|---:|
| A frozen baseline | +163.09 | -26.27 | 0.465 | -35.09% | -15.81 pp |
| B Adaptive Exit V1 | **+189.36** | -- | **0.548** | -19.28% | -- |
| C direct loss features | +181.64 | -7.71 | 0.490 | -26.77% | -7.49 pp |
| D failure hazard | +187.41 | -1.95 | 0.524 | -16.97% | +2.31 pp |
| E defensive value | +181.52 | -7.83 | 0.495 | -26.61% | -7.33 pp |
| F stop specialist | +187.80 | -1.56 | 0.538 | -22.59% | -3.31 pp |
| G MFE protection | +183.62 | -5.74 | 0.513 | -24.02% | -4.74 pp |
| H never-profitable specialist | +185.12 | -4.23 | 0.531 | -21.79% | -2.51 pp |
| I shock specialist | +189.36 | 0.00 | 0.548 | -19.28% | 0.00 pp |
| J supported specialists | +180.05 | -9.31 | 0.498 | -26.65% | -7.38 pp |
| K constrained downside | +184.11 | -5.25 | 0.515 | -18.21% | +1.07 pp |

The broad direct/value/union arms over-intervene. The shock arm never obtains
enough joint evidence to act. Hazard and action-value heads are the only useful
directions.

## Narrow refinement

| Challenger | Net EV/trade | Delta vs V1 | Sortino | Worst week | Daily CVaR5 | Time underwater |
|---|---:|---:|---:|---:|---:|---:|
| V1 | +189.36 | -- | 0.5480 | -19.28% | -21.06% | 40.74% |
| Action value > +10 bps | **+189.87** | **+0.51** | 0.5479 | **-13.38%** | **-20.25%** | **40.36%** |
| P(loss<-100) >= .85, SL 3.0 | +189.64 | +0.28 | 0.5487 | -15.69% | -20.37% | 40.56% |
| P(loss<-100) >= .90, SL 3.0 | +189.69 | +0.33 | **0.5508** | -16.82% | -20.51% | **40.32%** |
| P(loss<-100) >= .90, SL 3.5 | +189.03 | -0.33 | **0.5523** | -18.35% | -20.60% | 40.89% |

All retain the inherited -76.53% warm-up max drawdown and -50.57% worst day.

### Candidate-level portability

| Arm | 2025 uplift | 2026 uplift | 2025 CVaR5 change | 2026 CVaR5 change | Winner retention 2025 / 2026 |
|---|---:|---:|---:|---:|---:|
| Action value > +10 | +0.97 | 0.00 | +23.81 | 0.00 | 99.70% / 100% |
| Hazard .85, SL 3.0 | -0.16 | +0.36 | +14.20 | +9.75 | 99.75% / 99.97% |
| Hazard .90, SL 3.0 | +0.15 | +0.02 | +10.05 | +0.44 | 99.90% / 100% |
| Hazard .90, SL 3.5 | -0.18 | +0.01 | +2.02 | +0.18 | 99.93% / 100% |

`action_value_gt10` acts in only three 2025 months and has no 2026
interventions. Its portfolio uplift is real within the tested replay but is not
portable confirmation. The .90 hazard/3.0-ATR rule is much more conservative:
it has positive candidate uplift in both years and far lower false-defense
exposure, but its incremental benefit is small.

Across 81 candidate weeks:

- action-value >10: 11 positive-uplift, 68 unchanged, 2 negative weeks;
- hazard .85 / SL3.0: 28 positive, 30 unchanged, 23 negative;
- hazard .90 / SL3.0: 27 positive, 42 unchanged, 12 negative.

The .90 rule is the best stability compromise, but this is research evidence,
not enough to promote it over the incumbent.

## Learnability and oracle ceiling

Loss probability is learnable. The top predicted decile of the -100-bps head
has roughly 0.80 predicted and 0.77 realised loss frequency, and roughly +60
bps mean realised defensive gain. The equivalent -200-bps decile is about 0.66
predicted, 0.63 realised, and +63 bps defensive gain.

The stop-only loss oracle has approximately +13.6 bps/state headroom, but it
changes more than 99% of states. This is an upper bound, not a deployable policy.
The practical bottleneck is action usefulness and false-defense control, not
recognition of high-risk states.

## Limitations and next valid test

1. Reconcile the promising `.90 / SL3.0` rule on complete 15-minute paths;
   current evidence is matched to V1's hourly proxy.
2. Start evaluation after the first OOF fold when judging max drawdown, or build
   a genuinely pre-period model. Do not attribute the January warm-up DD to an
   unavailable exit head.
3. Improve action-value transport. The +10-bps head does not fire in 2026;
   recalibrating predicted action gain using prior-fold residuals is more useful
   than widening the action grid.
4. Retain the loss-probability head as risk evidence, but require action-specific
   mean/LCB agreement. A loss forecast alone should never tighten a stop.
5. Validate on an untouched post-development period before any promotion.

## Reproduction

Primary run:

```bash
PYTHONPYCACHEPREFIX=/tmp/ares-pyc \
NUMBA_CACHE_DIR=/tmp/ares-numba \
MPLCONFIGDIR=/tmp/ares-mpl \
python3 scripts/run_adaptive_exit_loss_control_v2.py \
  --out-dir data_perp/artifacts/adaptive_exit_loss_control_v2_20260813_v3
```

Narrow refinement:

```bash
PYTHONPYCACHEPREFIX=/tmp/ares-pyc \
NUMBA_CACHE_DIR=/tmp/ares-numba \
MPLCONFIGDIR=/tmp/ares-mpl \
python3 scripts/run_adaptive_exit_loss_control_v2_refinement.py \
  --parent data_perp/artifacts/adaptive_exit_loss_control_v2_20260813_v3 \
  --out-dir data_perp/artifacts/adaptive_exit_loss_control_v2_refinement_20260813_v2
```

Both manifests declare `COMPLETED_NOT_PROMOTED` and `promotion: none`.
