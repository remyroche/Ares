# Archetype-conditioned exit and sizing ablation

## Question

Can a pre-existing causal `policy_archetype` change exit or size without
changing whether a candidate is traded?  It must not affect model score,
ranking, entry threshold, or the initial candidate population.

## Matched protocol

- Candidate ledger: 565 July 2026 candidates from the S52 deployment replay.
- Common executable/path-valid population: 426 rows.  This intersection was
  enforced in every arm because the raw simulator's internal sanitisation had
  otherwise produced exit-dependent row counts.
- Paths: 15-minute bars, 96 bars / 24 hours; 100 bps round-trip cost.
- Parent portfolio policy, rank fields and admission inputs were frozen.
- The archetype affects only the declared exit parameter(s).  Different later
  accepted counts are possible because exit duration changes capacity and risk
  locks; there are no new candidates relative to the parent arm.
- Conditional geometry and sizing history are May--June 2026 only; replay is
  July.  The seven conditional values come from the archived selected trial,
  not a fit on the replay rows.

## Exit results

`bps/trade` below is the portfolio replay mean net return × 10,000.

| Arm | Trades | Net bps/trade | Net PnL | Delta net PnL vs parent |
|---|---:|---:|---:|---:|
| Parent exit control | 51 | -189.6 | -1,418.3 | — |
| Archetype SL only | 51 | -178.8 | -1,382.9 | +35.4 |
| Archetype giveback only | 51 | -191.8 | -1,434.9 | -16.6 |
| Archetype trailing-power only | 51 | -189.9 | -1,420.0 | -1.7 |
| Archetype trailing-activation only | 49 | -214.2 | -1,465.8 | -47.5 |
| Archetype capital-protection only | 49 | -193.7 | -1,414.6 | +3.6 |
| Archetype ATR shape only | 51 | -192.8 | -1,452.5 | -34.2 |
| Full archetype exit geometry | 50 | -196.8 | -1,444.8 | -26.6 |
| Full geometry plus archived size-power | 49 | -196.8 | -1,499.1 | -80.9 |

The conditional values actually tested were:

- SL and activation: historical per-archetype selected geometry.
- Giveback beta: 0.30--0.95.
- Trailing power: 1.5--2.0.
- The archived `size_power` is 1.1 for all seven archetypes, so that arm is
  not a meaningful conditional-sizing test and is retained only as a control.

## Causal sizing overlay

To make sizing genuinely archetype-specific, a separate **post-admission**
overlay was assessed.  It uses each archetype's May--June cumulative net-trade
mean, shrunk with a fixed 1,000-row side prior, then maps its within-side
quality z-score to a bounded multiplier.  It cannot add or remove a trade.

| Arm | Trades | Net PnL | Net bps/notional | Mean multiplier |
|---|---:|---:|---:|---:|
| Parent fixed size | 51 | -1,418.3 | -193.3 | 1.000 |
| 10% EB archetype tilt | 51 | -1,373.6 | -190.3 | 1.002 |
| 20% EB archetype tilt | 51 | -1,407.9 | -188.5 | 1.042 |

The 10% tilt mainly reduces short exposure (mean 0.941) and modestly increases
long exposure (1.048).  It improves both aggregate loss and bps/notional, but
the sample is only 51 accepted trades and remains sharply negative.  It is
evidence for a future constrained sizing study, not a live sizing rule.

## Decision

No archetype-conditioned exit or sizing policy advances.  The sole signal worth
retesting on a larger, chronologically rolling population is **bounded
archetype-conditioned SL**.  Giveback, activation, full geometry, and current
size-power do not help this matched replay.  Trailing power is effectively
neutral.

## Artifacts

- `data_perp/artifacts/archetype_exit_sizing_modulation_20260809_v3/`
- `scripts/run_archetype_exit_geometry_component_ablation.py`
- `scripts/materialize_archetype_trailing_controls.py`
- `scripts/evaluate_archetype_sizing_overlay.py`
