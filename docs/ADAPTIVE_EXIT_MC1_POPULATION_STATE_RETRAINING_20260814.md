# Adaptive Exit V1 — MC1 population and visitation retraining

## Decision

Do not replace Adaptive Exit V1 yet.

The matched replay supports applying the existing Adaptive Exit contract to the
complete reproducible MC1 population.  Changing the training population from
A5 to MC1 adds only a small, borderline portfolio uplift after that coverage is
matched.  Restricting training to states visited by an earlier OOF adaptive
controller does not improve on the MC1 parent-state arm.

No arm was promoted.

## Frozen experiment contract

- Long only.
- MC1_d2 admission at expected net >= +50 bps.
- SimplePolicyOptimiser parent: SL 4.1520006433 ATR, activation
  2.3262249198 ATR, fixed giveback 0.1023719900 ATR, H12 timeout, 100-bps
  round-trip cost once.
- Decisions after completed hourly bars; changes apply from the next 15-minute
  bar.
- Target: `remaining_favorable_from_entry_atr`.
- Objective: LightGBM quantile 0.65.
- Frozen F1/F4 contracts: 28 / 127 ordered fields.
- Nine-month rolling train, three-month OOF blocks, 12-hour purge, 40,000
  equal-month training-state cap.
- Activation-only authority: shrink 0.75, bounds 0.50x to 1.25x parent,
  train-only p80 F1/F4 disagreement abstention.

## Arms

| Arm | Definition |
|---|---|
| Parent | Frozen SimplePolicyOptimiser, no adaptive overlay |
| Existing V1 | Historical stored V1 OOF coverage only |
| Matched A5 control | Old A5 training population, refit fold by fold, scored on every identical held MC1 state |
| C1 | MC1 training population, parent-live states |
| C2 | MC1 population, training rows restricted by earlier strict-OOF C1 visitation |
| C3 | 70% C1-visited / 30% parent-state training mixture |

C2 is a strict OOF **visitation-filter** test: it removes states occurring after
the earlier OOF controller exited.  Activation-sensitive state values retain
the frozen parent-state materialization used by the incumbent research
contract.  It is not a recursive fixed-point regeneration of those fields
under the challenger policy.

## Data correctness

- MC1 exact-path candidates found: 41,109.
- Parent-policy parity valid: 41,108.
- Invalid source-vintage mismatch: 1, excluded from supervision and preserved
  in `invalid_parent_policy_parity_rows.parquet`.
- Valid-row parent parity MAE: 0.00000068 bps.
- Valid-row maximum absolute error: 0.001192 bps.
- No invalid row was encoded as an economic failure.
- Focused next-bar/state tests: 4 passed.

## Matched constrained portfolio results

Common comparison: 2025-07-01 through 2026-08-01.

| Arm | Trades/day | Net EV/trade | Net EV/day | Sortino | Max DD | Ulcer |
|---|---:|---:|---:|---:|---:|---:|
| Parent | 18.07 | +167.75 bps | +3,031.38 bps | 0.607 | -44.91% | 7.12 |
| Existing V1 stored support | 18.35 | +176.49 bps | +3,237.81 bps | 0.654 | -38.47% | 6.41 |
| Matched A5 control | 18.46 | +182.47 bps | +3,369.19 bps | 0.672 | -38.47% | 6.14 |
| C1 MC1 parent states | **18.50** | **+184.15 bps** | **+3,406.27 bps** | **0.676** | -38.47% | 6.21 |
| C2 OOF-visited filter | 18.41 | +184.10 bps | +3,390.12 bps | 0.659 | -38.47% | **6.06** |
| C3 70/30 mixture | 18.45 | +182.56 bps | +3,368.66 bps | 0.660 | -38.47% | 6.18 |

The difference between Existing V1 and the matched A5 control is coverage:
the matched control applies the same old training-population contract across
the complete reproducible held MC1 population.  It contributes +5.98
bps/trade over the stored-support replay.

After coverage is matched, C1 adds +1.68 bps/trade over the matched A5 control.
The 57-week block-bootstrap interval is -0.29 to +3.97 bps and the estimated
probability of positive uplift is 94.7%.  This is encouraging but not strong
enough to replace a live candidate without untouched forward confirmation.

C2 adds +1.64 bps/trade over the matched A5 control, with a -0.46 to +3.81 bps
weekly-block interval.  It does not beat C1 on EV, Sortino, or 2026 EV.

## Time portability

All six arms have 57/57 positive portfolio weeks.  Relative to the matched A5
control:

| Arm | Positive / negative uplift weeks | Median weekly uplift | Worst weekly uplift |
|---|---:|---:|---:|
| C1 | 33 / 22 | approximately 0.00 bps | -22.05 bps |
| C2 | 34 / 21 | +0.47 bps | -17.83 bps |
| C3 | 25 / 29 | approximately 0.00 bps | -40.29 bps |

C1 improves 9 of 13 months versus the matched A5 control.  Its worst monthly
delta is -1.65 bps.  C2 improves 8 of 13 months, but loses -4.28 bps in May
2026 and -2.75 bps in July 2026.  C3 has a -23.79 bps July 2026 failure and is
rejected.

| Arm | 2025 net EV/trade | 2026 Jan-Jul net EV/trade |
|---|---:|---:|
| Matched A5 control | +196.94 bps | +170.02 bps |
| C1 | +198.00 bps | **+172.26 bps** |
| C2 | **+198.82 bps** | +171.53 bps |
| C3 | +198.48 bps | +168.93 bps |

## Interpretation

1. The important immediate repair is complete MC1 inference/replay coverage,
   not self-distillation.  Live V1 already has a fail-closed scorer capable of
   acting on complete MC1 states; historical reporting must stop limiting V1
   to the old stored A5 OOF-ID intersection.
2. MC1-aligned retraining is directionally helpful but incremental.  Its
   portfolio uplift is produced through changed exit timestamps and capacity
   reuse; on identical candidate outcomes the mean C1-minus-matched-control
   delta is -0.29 bps, so the causal mechanism is not yet strong enough for
   promotion.
3. Training only on OOF-controller-visited states does not add value.  The
   visited arm's lower 2026 Sortino and weaker May/July behavior argue against
   replacing parent-state training.
4. Keep the current V1 bundle canonical.  Run C1 as a sealed shadow challenger
   on untouched forward trades.  Promotion should require positive matched
   portfolio uplift, no new dry weeks, and acceptable weekly downside over a
   predeclared forward window.

## Reproduction

```bash
python3 scripts/run_adaptive_exit_mc1_population_state_retraining.py
python3 scripts/finalize_adaptive_exit_mc1_population_state_retraining.py
python3 -m pytest -q tests/test_adaptive_exit_v1.py \
  tests/test_path_based_exit_optimisation.py \
  -k 'continuous_activation or decision_states'
```

Primary artifact directory:

`data_perp/artifacts/adaptive_exit_mc1_population_state_retraining_20260814_v1`

No canonical bundle or canonical handover was modified.
