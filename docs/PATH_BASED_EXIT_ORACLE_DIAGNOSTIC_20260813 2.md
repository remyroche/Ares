# Full 625-Action Exit Oracle Diagnostic — Coverage-Complete Final

## Decision

The requested oracle diagnostic is implemented and completed on all 3,000
sampled long trades and 100,242 causal path states.  It is diagnostic research:
oracle outcomes generate hypotheses only and never enter inference features.
No adaptive exit model is promoted.

Immutable primary artifacts:

- `data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260813_v4_complete15m`
- `data_perp/artifacts/path_based_exit_oracle_complete15m_full12k24_20260813_v6_final`
- `data_perp/artifacts/path_based_exit_challenger_complete15m_20260813_v3`
- `data_perp/artifacts/path_based_exit_oracle_capture_full_context_20260813_v3`

## Requirement verification

| Requirement | Result |
|---|---|
| Full 625-action entry replay | Complete: 3,000/3,000 paths |
| Full 625-action causal path-state replay | Complete: 100,242 states |
| Entry and action timing | Completed-bar decision; action applies next bar |
| Baseline parity | Maximum absolute difference 0.000186 bps |
| Granular requested oracle columns | All persisted |
| Trade/path state | Age, PnL, MFE, MAE, giveback, time since MFE/MAE, velocity, acceleration, persistence, slopes, roughness |
| Entry and score context | Entry confidence; base/consensus/residual/upstream levels, 1h changes and accelerations |
| Current versus entry market state | Momentum, trend, realized-volatility, VWAP, 24h range and changes |
| Regime and feature age | Causal volatility regime and hourly snapshot age |
| OOD/support/K9 trust | Backward-available entry fields only |
| Descriptive bins | Explicit numeric boundaries persisted |
| Conditional interactions | Persisted as hypothesis-only |
| CMI | Training-fold only; five bins; 24 circular 12h-block permutations within month |
| Predictability | CMI combined with training-environment action-value stability |
| Dominance | Material argmax contribution plus effective support and monthly portability |
| Nested ceilings | One/two/three/four components and action-magnitude caps |
| Tail decomposition | Loss prevention, giveback prevention, winner preservation |
| Portable oracle | Training-environment-consistent actions only |
| Capture ratio | Best portable intervention available per matched trade |
| Automatic promotion | Disabled |

The machine-readable verification is `requirement_coverage.json`.  All 29
correctness tests pass.

### Extended objective completion audit

| Objective item | Implementation and evidence | Status |
|---|---|---|
| Entry deltas, multi-horizon VWAP/range, score delta/acceleration | Current, entry and change-since-entry trend, momentum, volatility, VWAP; 24h high/low position; strict-prequential base/consensus/residual/upstream changes and accelerations | Complete |
| Ordered five-level actions | Ordered nonlinear basis with adjacent-level smoothness precision 10; no monotonic-benefit constraint | Complete |
| Rich path representation | Signed efficiency, excursion asymmetry, MFE/MAE persistence/slopes, roughness, separate age/recency and requested interactions | Complete |
| Sparse state-graph ablation | Path only; path plus entry; path plus entry/hourly/transitions; context-veto follow-up | Complete |
| Risk, stability and intervention metrics | Chronological max drawdown, PnL/DD improvement, Sortino, trade/daily CVaR5, winner retention, MFE capture, winner/loser attribution, month/week MAD and worst period, posterior bins, tick/trade intervention, action efficiency | Complete |
| Full lattice then train-only dominance | Full 625 replay precedes Pareto and region dominance; fold screens retain 58/69/105/151 configurations | Complete |
| Minute-data check and fallback | No complete canonical 1-minute path history exists for this universe; all 3,000 sampled paths were backfilled/materialised at 15-minute resolution and explicitly labelled a coarse proxy | Complete with declared fallback |
| Sparse stop+activation challenger | Action distance capped at 0.50; loss/giveback focus; PnL/MFE/giveback/effective-stop primary state; absolute intervention cap; explicit train-month negative-capture penalty | Complete |
| Promotion guard | +13.17 bps required; best non-retrospective full-context challenger is +8.32 and nothing is promoted | Complete |

## Data and causality

- Long side only, 15-minute source because no canonical one-minute history was
  available; the 15-minute replay is explicitly a coarse execution proxy.
- 3,000 trades were sampled before future-path inspection and all 3,000 now
  have complete paths after targeted historical backfill.
- The first decision is after a completed 15-minute bar.  A selected action
  takes effect on the following bar.
- Hourly context is joined backward with `available_at <= decision_ts`.
- Score evolution is sourced from the strict prequential ledger; current-row
  outcomes never enter the state features.
- Chronological folds have a 12-hour purge.  CMI bins, nulls, action-value
  stability, Pareto pruning, dominance and portable-action retention use only
  the training part of each fold.
- The held fold receives frozen training definitions.

## Oracle headroom

| Ledger | Rows | Mean | Median | P90 | P95 | P99 | Positive | >=50 bps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Entry 625 | 3,000 | +62.93 | 0.00 | +200.85 | +344.26 | +687.41 | 37.30% | 29.17% |
| Path state 625 | 100,242 | +45.32 | 0.00 | +136.81 | +272.38 | +561.77 | 44.98% | 21.79% |

This is a sparse-opportunity problem.  The median state has no action regret,
but the upper tail is economically large.

Mean path-state baseline utility is +58.90 bps; unrestricted oracle utility is
+104.22 bps.  The difference is the +45.32-bps mean oracle regret above.

## Complexity and intervention magnitude

| Ceiling | Entry regret | State regret | State capture |
|---|---:|---:|---:|
| Best one-component change | +55.39 | +40.22 | 88.73% |
| Best two-component change | +62.26 | +44.73 | 98.70% |
| Best three-component change | +62.93 | +45.32 | 100.00% |
| Best four-component change | +62.93 | +45.32 | 100.00% |

The third component adds only +0.59 bps/state over two components; the fourth
adds zero.  The adaptive layer should not start as a four-control model.

| Maximum normalized action distance | State regret | Capture |
|---|---:|---:|
| 0.25 | +30.83 | 68.01% |
| 0.50 | +44.86 | 98.97% |
| 0.75 / full | +45.32 | 100.00% |

A 0.50 intervention cap loses only 1.03% of oracle headroom.

The best fixed two-component controller is stop plus activation:

| Fixed pair | State regret | Capture of full state oracle |
|---|---:|---:|
| Stop + activation | +34.29 | 75.66% |
| Activation + giveback | +26.33 | 58.10% |
| Stop + giveback | +25.80 | 56.92% |
| Stop + power | +23.89 | 52.72% |
| Activation + power | +22.76 | 50.21% |
| Power + giveback | +8.06 | 17.77% |

The statewise best-two-component ceiling may choose a different pair at each
state; it is not the capacity of a single fixed pair.

## Where the opportunity is

| Opportunity | States | Mean | Median | P90 | P95 | Share of regret |
|---|---:|---:|---:|---:|---:|---:|
| Left-tail loss prevention | 16,864 | +130.68 | +44.63 | +386.78 | +469.54 | 48.51% |
| Winner-giveback prevention | 8,527 | +106.26 | +58.76 | +298.93 | +420.69 | 19.94% |
| Winner preservation/extension | 74,851 | +19.15 | 0.00 | +64.78 | +106.38 | 31.55% |

Loss and giveback prevention account for 68.45% of the total opportunity.  The
first useful adaptive model is therefore a downside/giveback controller, not a
general winner-extension engine.

## Opportunity x learnability

The map contains 4,184 training-defined regions:

| Decision | Regions |
|---|---:|
| Model high-value region | 14 |
| Model with strong shrinkage | 2,120 |
| Abstain: low support/predictability | 1,294 |
| Ignore: low oracle regret | 756 |

Held high-value examples include low short-horizon trend/momentum, high ATR
fraction, trend disagreement, VWAP distance, trend acceleration, time since
MFE, time since MAE, MAE severity and realized-volatility change.  Their held
mean regret ranges from +6.32 to +153.63 bps/state, but only 14 regions pass all
three opportunity, predictability and support gates.  This strongly favors
shrinkage and abstention.

## CMI and portability

The diagnostic assessed 220 causal candidate features against 16
one-component action contrasts and sparse pair interactions.  Ten fields were
selected in all three eligible folds:

- trade age;
- time since MFE;
- entry medium-horizon trend and momentum;
- entry medium-horizon VWAP distance and VWAP slope;
- entry consensus rank and residual rank;
- entry position within the 24-hour range;
- entry base-rank 1-hour acceleration.

Two-fold support also appears for time since MAE, MFE, PnL, volatility-of-
volatility, K9 entropy/top-2 margin/OOD, realized volatility, VWAP change,
current momentum and upstream/base-score accelerations.

OOD, support and K9 correlation-break fields are tested and sometimes selected,
but mostly in one fold.  They are conditional trust context, not portable
standalone triggers.  Velocity and acceleration are now correctly represented
and tested; raw PnL velocity is not portable by itself, while score acceleration
and some path slopes are conditionally useful.

## Train-only action pruning and dominance

The 625-action train-fold Pareto screen retains 58, 69, 105 and 151 actions in
folds 1–4.  Its risk-adjusted expected-value objective explicitly subtracts
25% of mean negative monthly capture, alongside CVaR, positive support,
positive-month coverage and action distance.  The stricter region-dominance portable oracle retains no
non-baseline action in fold 1, six in folds 2 and 3, and four in fold 4.  This
produces the following held portable ceiling:

| Fold | Held states | Retained actions including baseline | Mean portable regret |
|---|---:|---:|---:|
| 1 | 35,704 | 1 | 0.00 |
| 2 | 21,141 | 7 | +58.66 |
| 3 | 21,526 | 7 | +44.32 |
| 4 | 14,087 | 5 | +36.99 |
| All | 92,458 | — | +29.37 |

The zero first-fold ceiling means there was not yet enough prior-environment
evidence to declare any non-baseline action portable.  It must not be read as
zero raw oracle opportunity.

## Matched learner capture

The capture denominator is the maximum portable headroom available along each
matched trade.  The matched set has 2,752 trades and mean headroom +35.87
bps/trade.

The initial sparse result contained an implementation defect: a nominal 25%
intervention cap retained 25% of already-active decisions, rather than capping
interventions at 25% of all decision ticks.  The corrected train-only cap leaves
naturally sparse policies unchanged and applies a quantile only if their active
rate exceeds the absolute cap.  The correction is covered by a regression test.

| Corrected development arm | Uplift | Exact portable capture | Worst month | Positive months | Positive weeks | Tick intervention |
|---|---:|---:|---:|---:|---:|---:|
| Sparse path-only core | +9.52 | 26.54% | -26.18 | 61.54% | 54.72% | 19.40% |
| Sparse full-context routed | +8.32 | 23.19% | **-14.94** | 53.85% | 33.96% | 12.38% |
| Context-active veto of path actions | +7.04 | 19.64% | **-12.85** | 53.85% | 33.96% | 12.08% |
| Path core, fixed score >=150 | **+10.97** | **30.59%** | -17.98 | **69.23%** | **56.60%** | 16.43% |

The fixed score-150 arm is the best development result, but the threshold was
chosen after inspecting this OOF period.  It is therefore a diagnostic
ablation, not untouched validation and not a promotable policy.  Broader
context improves downside portability when used as a strict veto, but it also
removes substantial good path-only actions.  An OR-style authority rule adds
nothing because every context-approved state is already inside the high path-
confidence population.

After enforcing chronological ordering in the drawdown evaluator, the
full-context arm improves maximum drawdown from -27,153.88 to -19,045.49 bps;
the score-150 arm improves it to -18,231.15 bps.  Their respective
DeltaPnL/absolute-DeltaMaxDD ratios are 2.82 and 3.38.  The full-context arm
improves trade CVaR5 by +137.30 bps, daily CVaR5 by +1,492.36 bps and Sortino
by +0.0448, while retaining 97.95% of baseline-winner value.

No arm clears the mandatory +13.17-bps overall uplift gate.  Nothing is
promoted.

## Authoritative A5 reconciliation and integration assessment

The separate authoritative A5 adaptive-exit result is materially stronger:

| A5 comparison | Baseline | Adaptive | Uplift | Notes |
|---|---:|---:|---:|---|
| Fixed canonical trades | +163.09 | +183.63 | **+20.55** | 8,453 IDs; 6,800 source-matched OOF controls |
| Capacity-aware auction | +163.09 | +181.61 | **+18.52** | 8,663 adaptive trades versus 8,453 baseline |

This is not a direct +20.55 versus +8.32 model comparison.  A paired identity
audit finds only 809 overlapping OOF IDs.  On that intersection A5 adds +32.87
bps and the present challenger +7.48 bps, but their baseline outcomes differ by
89.31 bps MAE and -14.89 bps bias.  They use different baseline geometry,
sampling, decision cadence and outcome-source contracts.  The immutable audit
therefore marks direct uplift comparison invalid rather than silently treating
the substrate difference as learning edge.

The architectural evidence is nevertheless decisive enough to redirect the
next experiment:

1. Use A5's source-aligned continuous activation proposal as the primary
   controller.  Keep stop and giveback frozen.
2. Train the CMI/Bayesian layer on the proposal-specific target
   `utility(A5 proposal) - utility(baseline)`, not on the 625-way argmax.
3. Give that layer only shrink/abstain authority:
   `activation = baseline + trust * (A5 - baseline)`, with `trust` in `[0,1]`.
4. Start with the path-only causal fields.  Add entry and hourly transition
   context only as a trust modifier because the central ablation shows path
   state drives mean edge while context mainly improves downside portability.
5. Preserve A5's source alignment: F4-disagreement-gated continuous control on
   reproducible fine paths and the existing F1 path-28 continuous control on
   hourly-proxy rows.
6. Compare baseline, current sparse challenger, A5, and A5 plus proposal-risk
   shrinker on identical canonical IDs, paths, clock, ATR, cost and auction.
7. Select the shrinker on 2025 only and reserve 2026 or a later frozen block for
   confirmation.  Do not let it change A5 admission.

This integration is preferable to combining final scores.  Oracle evidence
supports simple continuous activation authority: one changed component already
captures 88.73% of state oracle headroom, while a second raises it to 98.70%.
The current Bayesian machinery is therefore more valuable as a risk veto over
the strong A5 proposal than as a competing multi-action policy.

## Recommendations

1. Start with a stop/activation controller capped at 0.50 action distance, but
   do not use the present over-sparse router as-is.
2. Train separate loss-prevention and giveback-prevention value heads, with a
   shared abstention gate.  Those two regions contain 68.45% of headroom.
3. Optimize for portable capture, worst-month uplift, and intervention rate
   jointly.  The corrected 20% portable-capture diagnostic is +7.17 bps/trade,
   but retain the objective's stricter +13.17-bps promotion gate and require no
   materially negative held month.
4. Give the router the stable three-fold state fields first.  Add two-fold OOD,
   K9, volatility and score-change fields only through shrinkage.
5. Use region-specific posterior lower bounds.  Abstain when either effective
   support or action-value stability is weak.
6. Preserve the ordered nonlinear action basis and adjacent-level smoothness;
   do not impose a false monotonic benefit assumption across action levels.
7. Test a small pair router among stop+activation, activation+giveback and
   stop+giveback only after the fixed stop+activation challenger is stable.
8. Freeze the resulting contract and validate on an untouched later period.

## Reusable commands

Full 625-action diagnostic:

```bash
/usr/local/bin/python3 scripts/run_path_based_exit_oracle_diagnostic.py \
  --run-dir data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260813_v4_complete15m \
  --output-dir data_perp/artifacts/path_based_exit_oracle_complete15m_full12k24_20260813_v6_final \
  --cmi-max-rows 12000 \
  --cmi-permutations 24
```

Staged learner funnel:

```bash
/usr/local/bin/python3 scripts/run_path_based_exit_challenger_funnel.py \
  --run-dir data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260813_v4_complete15m \
  --out-dir data_perp/artifacts/path_based_exit_challenger_complete15m_20260813_v3 \
  --cmi-max-rows 4000 \
  --cmi-permutations 6
```

Lightweight matched capture audit without recomputing CMI:

```bash
/usr/local/bin/python3 scripts/run_path_based_exit_oracle_capture_audit.py \
  --oracle-dir data_perp/artifacts/path_based_exit_oracle_complete15m_full12k24_20260813_v4_final \
  --model-run-dir data_perp/artifacts/path_based_exit_challenger_complete15m_20260813_v3/C_path_entry_hourly_transition \
  --output-dir data_perp/artifacts/path_based_exit_oracle_capture_full_context_20260813_v3
```

Path-core/context-authority diagnostic:

```bash
/usr/local/bin/python3 scripts/run_path_based_exit_authority_gate_ablation.py \
  --run-dir data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260813_v4_complete15m \
  --path-run-dir data_perp/artifacts/path_based_exit_challenger_path_choices_20260813_v1 \
  --context-run-dir data_perp/artifacts/path_based_exit_challenger_context_choices_20260813_v1 \
  --output-dir data_perp/artifacts/path_based_exit_authority_gate_20260813_v3
```

Matched A5 contract audit:

```bash
/usr/local/bin/python3 scripts/compare_path_exit_with_authoritative_a5.py \
  --challenger-trades data_perp/artifacts/path_based_exit_challenger_negcapture_w025_20260813_v1/D_routed_stop_activation_challenger/adaptive_exit_state_suffix_oof_trades.parquet \
  --a5-oof-replay data_perp/artifacts/canonical_a5_source_aligned_hybrid_adaptive_exit_funnel_20260813_v4/oof_replay.parquet \
  --a5-fixed-trades data_perp/artifacts/authoritative_a5_adaptive_exit_reconciliation_20260813_v4/fixed_canonical_trade_comparison.parquet \
  --output-dir data_perp/artifacts/path_exit_authoritative_a5_reconciliation_20260813_v1
```

Primary code:

- `extreme_price_movements/path_based_exit_optimisation.py`
- `scripts/run_path_based_exit_portability_validation.py`
- `scripts/run_path_based_exit_oracle_diagnostic.py`
- `scripts/run_path_based_exit_oracle_capture_audit.py`
- `scripts/run_path_based_exit_challenger_funnel.py`
- `scripts/run_path_based_exit_authority_gate_ablation.py`
- `scripts/compare_path_exit_with_authoritative_a5.py`
- `tests/test_path_based_exit_optimisation.py`
