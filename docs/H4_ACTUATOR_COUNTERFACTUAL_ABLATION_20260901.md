# H4 actuator-specific counterfactual ablation — 2026-09-01

## Status

Research-only Stage-1 screen.  No entry, MC1, portfolio, live execution, or
canonical exit artifact was changed.  The existing H4 continuation policy
remains the retained control.

> **Supersession notice (2026-09-01):** all stop-distance findings in this
> document are invalid for selection. Its offline adapter emitted
> `stop_multiplier`, while the rich replay consumes `sl_distance_multiplier`,
> making stop actions inert. The repair and replacement temporary-action
> results are recorded in `H4_NEXT15M_PATH_ACTION_VALUE_ABLATION_20260901.md`.
> This did not affect the live parent policy.

The study follows the linked actuator specification: each controller changes
one rich-policy actuator only after a completed 15-minute state, then applies
the action to the next interval.  It is deliberately not a joint G/T/S search.

## Receipt

- Runner: `scripts/run_causal_sr_h4_actuator_counterfactual_ablation.py`
  (`a4524395ae48f2458894aef8a451f3ea76968bcef167a437f6c36925d1a56762`)
- Complete output: `data_perp/artifacts/causal_sr_h4_actuator_counterfactual_2025oof_2026confirm_20260901_v6`
- Run manifest SHA-256:
  `a774c9b2ab00ebfb739bcd44b28f384c0b7992b5db8556950028c20ff36e0acf`
- 2025 summary SHA-256:
  `4be0f693bcde7d575e68b23d3d3251b0ed456a24896f316daea2a90c8c316525`
- 2026 summary SHA-256:
  `1f910f27123dd08a8a604b9f43924bbc39a7210a8b148bae4903b2f06f211017`

## Contract

- Long-only, causal S/R plus paired BCF/current-v5 MC1 source route.
- Exact, resolved one-minute rich-parent paths; no incomplete H12 path is
  represented as a flat trade or as a capacity-reserving pseudo-trade.
- Training labels: dual MC1 at least +40 bps, no portfolio constraint;
  first/middle/last target-free completed states per candidate.  A label can
  enter a fit only after its parent H12 outcome is resolved.
- Assessment: dual MC1 at least +50 bps and the unchanged normal global,
  chronological constrained portfolio auction.
- Selection: June–December 2025, monthly strict-prior OOF.  Each held month
  sees only earlier resolved labels.
- Confirmation: June–August 2026.  All controllers are frozen from 2025
  labels; 2026 did not select a model, feature, threshold, or mapping.
- Features: all pre-existing numeric, target-free H4 state fields.  No
  feature subset selected in 2026 was imported into the 2025 screen.
- Controllers: giveback, trailing activation, and stop distance separately;
  multipliers `{0.65, 0.80, 1.00, 1.25, 1.50}`.  Stop extension is bounded at
  5% of entry price.
- Stage-1 models: actuator-specific shallow LightGBM, depth 3, 7 leaves,
  5% minimum child support, L2 40, learning rate .035, 280 trees.  Mappings
  are dual widen-only, dual tighten-only, asymmetric dead-zone, and a
  five-class ordinal step control.
- No exchange calls were made.

## Learnability before model fitting

| Actuator | Labelled states | Mean tighten advantage | Mean widen advantage | Oracle advantage | Materially adjustable share |
|---|---:|---:|---:|---:|---:|
| Trailing activation | 52,501 | +4.96 bps | −1.68 bps | +8.08 bps | 3.22% |
| Giveback | 52,501 | +0.72 bps | +0.40 bps | +2.52 bps | 1.19% |
| Stop distance | 52,501 | **Invalid: inert adapter key** | **Invalid** | **Invalid** | **Invalid** |

The former assertion that stop distance has no exact counterfactual variation
is superseded by the adapter-key repair described above.

## Constrained results

### Strict-prior 2025 OOF selection (Jun–Dec)

| Arm | Trades | Net bps/trade | Total net bps | Sortino | Max DD | Worst week | CVaR10 | Worst month |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Parent | 4,424 | +114.24 | +505,405 | 0.495 | −0.516 | +0.194 | −430.23 | +75.78 |
| Activation dual tighten / asymmetric | 4,458 | **+116.93** | **+521,270** | **0.537** | **−0.509** | **+0.393** | **−411.56** | +75.51 |
| Activation ordinal step | 4,448 | +116.75 | +519,315 | 0.534 | −0.509 | +0.393 | −412.85 | **+76.51** |
| Giveback ordinal step | 4,429 | +114.68 | +507,916 | 0.498 | −0.516 | +0.276 | −429.55 | +76.06 |

The activation dual-tighten action was applied at 9.10% of observable states
in 2025; widen action was essentially absent.  Giveback and stop models had
little useful authority; their apparent high ordinal action rates did not
translate into a reliable counterfactual improvement.

### Frozen 2026 confirmation (Jun–Aug)

| Arm | Trades | Net bps/trade | Total net bps | Sortino | Max DD | Worst week | CVaR10 | Worst month |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Parent | 558 | +136.97 | +76,430 | 0.686 | −0.098 | +0.026 | −267.46 | +102.94 |
| Activation dual tighten / asymmetric | 558 | 136.82 | 76,346 | 0.685 | −0.098 | +0.026 | −267.46 | +102.94 |
| Activation ordinal step | 558 | 136.97 | 76,430 | 0.686 | −0.098 | +0.026 | −267.46 | +102.94 |
| Giveback ordinal step | 558 | **+137.35** | **+76,639** | **0.688** | −0.098 | **+0.030** | −267.46 | **+104.03** |

The 2025 winner, activation tightening, does not carry its gain into the
frozen 2026 confirmation (−84.6 total bps).  The giveback ordinal controller
is modestly positive in 2026 but weak in selection (+2,511 total bps) and has
no sufficient evidence to replace the retained H4 policy.

## Decision and next research step

No actuator controller is promoted.  Keep the current H4 continuation policy
unchanged.  If revisited, restrict follow-up to trailing-activation labels:
its oracle headroom and sparse, mostly-tightening action pattern make it the
only actuator with a credible learning signal.  Any HPO must select target,
mapping, capacity, and authority on 2025 only, then repeat the unchanged
frozen 2026 confirmation.
