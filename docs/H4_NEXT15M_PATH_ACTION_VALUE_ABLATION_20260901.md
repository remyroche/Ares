# H4 temporary path-action exit study — 2026-09-01

## Decision

**No new path-action exit controller is promoted.** Every challenger is an
offline research artifact only. The rich parent policy remains unchanged in
the canonical and live stacks.

The promotion rule was fixed before the tuning pass: an added mechanism must
improve frozen OOS net EV by at least **+10 bps/trade (0.10%)**, not merely
improve an internal action label or a risk statistic.

## What changed in the question

The previous permanent-actuator question was too blunt. This study instead
labels a state by the exact counterfactual value of changing *one* parent
actuator for the next completed 15-minute interval, then restoring the rich
parent setting if the trade remains open:

`advantage = exact H12 policy net bps(temporary action) - exact H12 parent net bps`

For every tested state the controller can choose 0.65x, 0.80x, 1.00x, 1.25x,
or 1.50x of one actuator. Tight-only, wide-only, and asymmetric controllers
were evaluated separately. The actions are scheduled causally from target-free
state fields, start after the completed state bar, and are reset at the next
completed state.

This design allows both short-lived tightening and short-lived widening; it
does not assume that protection is always the right action.

## Contract and safeguards

- Labels: paired BCF/current MC1 >=40 bps, no portfolio auction, complete
  exact paths only; first/middle/last target-free state per trade.
- Selection: June–December 2025, strict-prior monthly OOF.
- Evaluation: normal paired BCF/current MC1 >=50 bps route, exact rich-parent
  replay and the normal global chronological portfolio constraints.
- Confirmation: one frozen June–August 2026 evaluation trained only on
  resolved 2025 labels.
- Features: unchanged 91-field target-free H4 state contract. It already
  contains trade state, MFE/MAE, policy distances, returns/momentum, force,
  support/resistance, volatility/volume/OI, VWAP, and expectation-deviation
  fields.
- Scope: no exchange calls or mutations of geometry/K9, admission, MC1, C1
  S/R, portfolio rules, or live policy.

## Results

All values below are deltas to the rich-parent constrained portfolio.

| Mechanism / best 2025-only arm | 2025 OOF EV/trade | 2025 total bps | 2025 max-DD delta | Frozen Jun–Aug 2026 EV/trade | Frozen 2026 total bps | Decision |
|---|---:|---:|---:|---:|---:|---|
| Activation, baseline model, 35-bps gate | +4.53 | +25,750 | +3.48 pp | -0.38 | -210 | Reject |
| Activation, concentrated 50-bps gate | +4.34 | +24,082 | +3.48 pp | -0.10 | -54 | Reject |
| Giveback, tight-only 1-bps gate | +0.67 | +3,867 | +0.08 pp | +0.58 | +325 | Reject: below +10 bps gate |
| Stop distance, corrected key, 15-bps gate | +1.19 | +13,218 | +6.76 pp | -1.99 | -841 | Reject |
| Stop distance, concentrated 50-bps gate | +0.79 | +6,719 | unchanged | -1.31 | -459 | Reject |
| Stop distance, sparse-event L1 HPO, 25-bps gate | +0.95 | +11,807 | +6.36 pp | -1.48 | -557 | Reject |
| Activation, extended 0.40–1.75x envelope with continuous authority | +4.96 | +30,048 | +3.98 pp | -0.38 | -210 | Reject |
| Stop distance, extended 0.40–2.00x envelope with continuous authority | +1.72 | +23,246 | +14.24 pp | -0.31 | +101 | Reject: below +10 bps gate |

Positive max-DD deltas mean a less-negative drawdown. The risk improvements
from stop tightening are real, but are insufficient without the required
per-trade EV improvement.

## What the ablation establishes

Path state is **not useless**:

- Temporary activation tightening has positive local action value in 2025.
- Temporary giveback tightening is positive in both selection and frozen 2026,
  albeit only +0.58 bps/trade in 2026.
- Stop action values have clear OOF ordering. The sparse-event L1 model reached
  a 0.264 strict-prior label-level Spearman correlation and +67.1 bps mean
  temporary value in its top 2% label states.

But this label-level predictive signal does not translate into enough
portfolio-level realised EV. Most temporary actions never alter an exit;
actions that do alter an exit can change availability and portfolio capacity;
and the rich parent already captures much of the readily available path state.
The remaining incremental effect is therefore too small and insufficiently
portable for a new live actuator.

Widening did not earn authority in any run: it was neutral or negative and the
asymmetric controller reduced to its tight-only counterpart. No wide action is
promoted.

## Are the changes large enough, and do they scale with confidence?

**The replay engine allows materially larger moves than the first grid used.**
Activation and giveback are bounded to `0.20x..1.80x`; stop distance is bounded
to `0.20x..3.00x`, with a 5% absolute stop-distance cap. The extended study
therefore explicitly tested 0.40x, 0.55x, 0.65x, 0.80x, 1.00x, 1.25x, 1.50x,
and 1.75x (plus 2.00x for stop distance). In practical terms, 0.40x makes a
trailing activation 60% earlier or a hard-stop distance 60% tighter. This is
large enough to change real exits, while staying inside the parent policy's
hard limits.

The original controller was **not** gradual: it chose a discrete multiplier
only after a predicted-action-value gate. The research runner now also tests a
monotone continuous tightening curve:

`confidence = clip(predicted advantage of the strongest tight action / S, 0, 1)`

`multiplier = 1 + (strongest_tight_multiplier - 1) × confidence`

with predeclared confidence scales `S = 20, 50, 100` bps. It remains causal,
takes effect only after a completed 15-minute state, resets on the following
state, and cannot relax a hard stop or armed smooth protection.

The results do **not** support promotion:

- Activation selected the most conservative gradual scale (`S=100`) in 2025
  OOF: +4.96 bps/trade, but frozen 2026 was -0.38 bps/trade.
- Stop distance selected `S=50`: it improved frozen 2026 Sortino by +0.050 and
  reduced max drawdown by 2.54 pp, but reduced EV/trade by 0.31 bps.
- The gradual curves did not beat the best fixed high-confidence action. They
  act in lower-confidence states and dilute a very sparse local signal.
- Wider actions still receive no authority: their predicted and realised local
  action value is neutral to negative.

So the answer is: **the allowed adjustment range is sufficient; the present
confidence signal is not sufficiently precise to justify using that range in a
live controller.** A future retry should use a calibrated lower confidence
bound (for example, an ensemble/quantile lower bound), not the raw predicted
advantage, and must again clear +10 bps/trade on frozen OOS.

## Corrected stop-key defect

The initial temporary-stop label receipt
`causal_sr_h4_next15m_stop_labels_2025_20260901_v1` is superseded and must
not be used. The offline adapter wrote `stop_multiplier`, while the rich
policy correctly consumes `sl_distance_multiplier`; the old receipt therefore
had all-zero stop effects. The research adapters now use the correct key.
This repair did not alter live exit code or the parent policy.

Earlier permanent-stop results produced by the same adapter are also not valid
for model selection until regenerated with the corrected key.

## Tuning completed

The strict-prior screen compared L2, L1, Huber, depth/leaf/support/
regularisation variants, and sparse-positive weighting. The best new contender
was the sealed sparse-event L1 stop model:

- objective `regression_l1`; depth 3; 7 leaves;
  minimum child fraction 5%; L2 80; positive-label weight 4.
- It improved action-label ranking but not the required downstream economics.

No further HPO on this 91-field contract is justified under the +10-bps rule.

## Next research only if the information set changes

The appropriate next test is not further parameter search. Add new strictly
causal path information, then repeat this exact action-value protocol:

1. impulse-strength/recovery/efficiency decay and adverse-direction efficiency;
2. worsening pullback, failed favourable break, and nearby-level hold failure;
3. participation-direction divergence using volume/OI;
4. a 15-minute micro-regime-flip representation.

Each candidate must be target-free at the completed 15-minute state, tested
first as a label-level incremental feature, then through this constrained
temporary-action replay, and finally meet +10 bps/trade on frozen OOS before
promotion.

## Relevant scripts and receipts

- `scripts/run_causal_sr_h4_next15m_actuator_ablation.py` — exact temporary
  action labels and constrained portfolio confirmation.
- `scripts/screen_causal_sr_h4_next15m_action_models.py` — strict-prior
  sparse-label model screen.
- `data_perp/artifacts/causal_sr_h4_next15m_activation_2025oof_2026confirm_20260901_v1`
- `data_perp/artifacts/causal_sr_h4_next15m_activation_highgate_2025oof_2026confirm_20260901_v2`
- `data_perp/artifacts/causal_sr_h4_next15m_giveback_lowgate_2025oof_2026confirm_20260901_v1`
- `data_perp/artifacts/causal_sr_h4_next15m_stop_2025oof_2026confirm_20260901_v2_corrected_key`
- `data_perp/artifacts/causal_sr_h4_next15m_stop_l1_sparse_hpo_2025oof_2026confirm_20260901_v1`
- `data_perp/artifacts/causal_sr_h4_next15m_activation_extended_envelope_labels_2025_20260901_v1`
- `data_perp/artifacts/causal_sr_h4_next15m_stop_extended_envelope_labels_2025_20260901_v1`
- `data_perp/artifacts/causal_sr_h4_next15m_activation_extended_gradual_2025oof_2026confirm_20260901_v1`
- `data_perp/artifacts/causal_sr_h4_next15m_stop_extended_gradual_2025oof_2026confirm_20260901_v1`

## Causal information-set extension — 2026-09-01

### Decision

**No information-set challenger is promoted.** The rich parent exit policy,
the retained H4 continuation head, and the live stack remain unchanged.

The study deliberately started with an action-sensitivity decomposition rather
than another broad predictor.  For each temporary actuator action it records:

`P(exit path changes | causal state, action) × E[net-bps advantage | exit path changes, causal state, action]`

An exact exit-path change means a difference in the counterfactual exit
minute, reason, or price relative to the neutral rich-parent trace.  The
neutral action is asserted to retain the exact parent exit trace.  Labels only
become trainable after the complete H12 path resolves.

The decomposition does identify action relevance, but it does not improve the
economically relevant action tail versus the direct advantage model.  It is
therefore a diagnostic only, not an authority mechanism.

| Temporary action | Direct 91-field top-1% advantage | Decomposed top-1% advantage | What it establishes |
|---|---:|---:|---|
| Activation 0.40x | +123.18 bps | +85.82 bps | Exit-change probability is identifiable, but direct action value remains better for economics. |
| Giveback 0.65x | +3.64 bps | +2.19 bps | A small, well-ranked local effect; too small to justify a controller. |
| Stop 0.40x | +173.83 bps | +105.92 bps | Sparse stop-path changes are predictable, but the direct target remains the stronger selector. |

### New target-free feature blocks

Every row is built solely from the candidate's exact post-fill one-minute
`high/low/close` path through the already-completed 15-minute decision state.
All price distances are normalized by the immutable entry-time signal ATR.
No block uses an exit, an outcome, a future bar, an action label, or a
cross-candidate refit.  This preserves the frozen Geometry/K9 contract: no K9
membership, centre, or temperature is recomputed or used as an unstable state
label.

- **Path deterioration/recovery (17 fields):** favourable/adverse contiguous
  impulse strengths and decay, directional/adverse efficiency, recovery
  strength/speed/decay, pullback depth/duration/speed/severity trend, failed
  favourable break, and impulse counts.
- **Directional-versus-chop volatility (15 fields):** post-fill 15/30/60m
  realised volatility, acceleration/decay, directional expansion, chop,
  favourable/adverse semivolatility and range shares, and volatility of
  volatility.
- **Micro-regime flip (11 fields):** aligned 15/30/60m trend returns,
  15m-versus-1h flip, adverse acceleration/streak/efficiency, local structure
  break, failed reclaim, range position, and a fixed target-free flip score.
- **Dynamic geometry (12 fields):** prior local one-hour range and position,
  favourable/adverse extension, acceptance/rejection at *prior* boundaries,
  boundary-cross density, balance density, range compression, and breakout
  efficiency.

The Stage-1 screen compared each block separately against the unchanged
91-field H4 control using strict-prior June–December 2025 monthly OOF folds.
It records OOF residual Spearman, discretised conditional mutual information
given the control-score decile, sparse 1/2/5% action-tail diagnostics, and
decile calibration.  The new fields genuinely contain residual information;
for example, adverse semivolatility/chop, pullback-severity trend, the
micro-regime-flip score, and boundary-cross/acceptance geometry each show
non-zero conditional information.  But none produces a stable sufficient
improvement in direct action value across sparse tails.

| Added block and temporary action | Δ top-1% advantage | Δ top-2% | Δ top-5% | Stage-1 disposition |
|---|---:|---:|---:|---|
| Path deterioration/recovery — activation | +7.22 bps | -16.67 bps | +0.66 bps | Inconsistent; no replay |
| Path deterioration/recovery — giveback | -0.93 bps | -0.27 bps | -0.02 bps | Reject |
| Path deterioration/recovery — stop | -6.83 bps | +1.25 bps | +0.27 bps | Reject |
| Directional/chop volatility — activation | -11.35 bps | -10.77 bps | +1.84 bps | Reject |
| Directional/chop volatility — giveback | -0.80 bps | -0.06 bps | -0.13 bps | Reject |
| Directional/chop volatility — stop | +1.00 bps | +0.49 bps | +0.35 bps | Reject |
| Micro-regime flip — activation | -7.32 bps | -21.65 bps | +2.21 bps | Reject |
| Micro-regime flip — giveback | +0.72 bps | +0.26 bps | -0.02 bps | Reject |
| Micro-regime flip — stop | +3.69 bps | -1.19 bps | -0.20 bps | Reject |
| Dynamic geometry — activation | -10.25 bps | -4.72 bps | -1.73 bps | Reject |
| Dynamic geometry — giveback | -0.69 bps | +0.92 bps | -0.24 bps | Reject |
| **Dynamic geometry — stop** | **+7.85 bps** | **+2.38 bps** | **-0.33 bps** | Only near-candidate; constrained replay required |

Positive tail deltas are changes in mean exact temporary-action advantage,
not portfolio PnL.  They are a screening result, not a live claim.

### Constrained replay of the only near-candidate

Only the dynamic-geometry 0.40x temporary-stop candidate advanced.  It used
the exact paired BCF/current MC1 >=50-bps normal route, unchanged chronological
global portfolio constraints, and the rich parent policy.  Model selection was
strict-prior June–December 2025 OOF; the winner was frozen before the
June–August 2026 confirmation.  A direct sparse L1 model was used: depth 3,
7 leaves, 5% minimum-child support, L2 80, positive-label weight 4.  The grid
included tighten, widen and asymmetric authority at 20/40/60-bps gates.

| Period / arm | Net EV/trade | Δ EV/trade | Total net bps | Sortino | Max DD | Worst week |
|---|---:|---:|---:|---:|---:|---:|
| 2025 OOF parent | +114.24 | — | +505,405 | 0.495 | -51.63% | +19.38 bps |
| 2025 OOF best: 0.40x stop, 60-bps tight gate | +116.29 | +2.05 | +529,721 | 0.594 | -37.55% | +44.19 bps |
| Frozen Jun–Aug 2026 parent | +136.97 | — | +76,430 | 0.686 | -9.79% | +2.57 bps |
| Frozen 2026 winner | +136.95 | **-0.02** | +76,691 | 0.745 | -7.71% | +1.91 bps |

Widening scheduled zero states in every 2025 arm.  The asymmetric arm reduced
to the corresponding tightening arm.  The candidate has a worthwhile
risk-shape change, but it fails the predeclared required gain of +10 bps/trade
on frozen OOS by a wide margin.  It is **not** an accepted risk-for-profit
substitution because the stated gate requires both positive economic gain and
no unacceptable risk degradation.

The runner also now supports a quantile objective for a lower-confidence-bound
authority test.  It was intentionally not selected after the direct dynamic
geometry replay failed the preceding economic gate; running LCB/HPO to rescue
a +2-bps effect would be post-hoc search, not falsification.

### New scripts and immutable receipts

- `scripts/run_causal_sr_h4_next15m_actuator_ablation.py` — now accepts an
  exact-keyed target-free feature panel and an optional quantile/LCB objective;
  still offline only.
- `scripts/screen_causal_sr_h4_exit_sensitivity.py` — strict-prior direct,
  exit-change, and conditional-benefit decomposition; supports control versus
  control-plus-extra comparisons.
- `scripts/audit_causal_sr_h4_incremental_information.py` — residual Spearman,
  conditional mutual information, and sparse-tail diagnostics.
- `scripts/build_causal_sr_h4_path_deterioration_features.py`
- `scripts/build_causal_sr_h4_directional_chop_volatility_features.py`
- `scripts/build_causal_sr_h4_micro_regime_flip_features.py`
- `scripts/build_causal_sr_h4_dynamic_geometry_features.py`
- `data_perp/artifacts/causal_sr_h4_*_sensitivity_2025oof_20260901_v1` —
  block-by-actuator strict-prior receipts.
- `data_perp/artifacts/causal_sr_h4_dynamic_geometry_stop_constrained_2025oof_2026frozen_20260901_v1`
  — exact constrained replay and frozen confirmation.

All new manifests state `no_exchange_calls: true`.  No canonical, live, MC1,
admission, portfolio, Geometry/K9, or rich-parent exit artifact was modified.

## Remaining-trade regime ablation — no promotion

The next experiment implemented the coherent remaining-trade regime proposal.
It replaces a one-interval actuator perturbation with one action selected at a
completed 15-minute state and retained through the remaining H12 path:

- **Parent:** unchanged rich parent policy.
- **Protect:** 0.65x future trailing activation and 0.75x future giveback;
  hard stop and smooth-lock floors are unchanged.
- **TrendRide:** 1.25x future activation and giveback; it cannot relax an
  already-ratcheted trailing or smooth floor.

For a state `t` and regime `r`, the label is the exact, causal remainder value
`U(r from t onward) − U(parent from t onward)`.  Labels become available only
after the complete H12 outcome; action selection is therefore target-free at
the decision time.  A chosen action persists for the remainder of the trade,
which is stronger than the proposed 30/60-minute commitment and prevents
flip-flopping by construction.

The 2025 OOF selection population is June–December.  It uses the unchanged
91-field target-free H4 state contract, an unchanged paired BCF/current MC1
>=50-bps route, exact one-minute rich-parent paths, and the normal global
portfolio auction.  The June–August 2026 confirmation uses the single
2025-selected feasible arm without refitting.  No Geometry/K9 state,
admission, portfolio rule, MC1 map, or live artifact was changed.

### Action-space ceiling

| Regime label, sampled target-free states | Labels | Mean exact advantage | Positive share | Exact exit-path change share |
|---|---:|---:|---:|---:|
| Protect | 52,501 | +3.82 bps | 19.29% | 22.95% |
| TrendRide | 52,501 | -2.26 bps | 1.74% | 19.82% |

The non-deployable outcome-aware ceiling reinforces this result.  Oracle R1
(Parent versus Protect) improves the 2025 constrained replay by +7.05
bps/trade; oracle R2 (Parent versus Protect versus TrendRide) improves it by
+8.69 bps/trade.  TrendRide is chosen in only 248 of the R2 oracle's 3,234
changed paths.  Both ceilings are below the predeclared +10-bps-per-trade
minimum for adding an exit mechanism and far below the +50–100-bps threshold
that would justify a broad data-driven R3 regime search.  **R3 is therefore
rejected rather than searched post hoc.**

### Strict-prior, constrained selection

| 2025 Jun–Dec OOF arm | Trades | Net EV/trade | Δ vs parent | Total net bps | CVaR10 | Worst month | Sortino | Max DD | Scheduled actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parent | 4,424 | +114.24 | — | +505,405 | -430.23 | +75.78 | 0.495 | -51.63% | 0 |
| Oracle R1 Protect (not feasible) | 4,461 | +121.29 | +7.05 | +541,078 | -398.90 | +77.80 | 0.575 | -48.05% | 3,204 |
| Oracle R2 Protect/TrendRide (not feasible) | 4,455 | +122.93 | +8.69 | +547,655 | -399.75 | +78.59 | 0.580 | -48.05% | 3,234 |
| Coarse Protect, 20-bps LCB | 4,442 | +115.76 | +1.52 | +514,225 | -423.06 | +77.09 | 0.509 | -51.63% | 336 |
| Conditional ML, mean or 20/30-bps LCB, R1/R2 | 4,424 | +114.24 | +0.00 | +505,405 | -430.23 | +75.78 | 0.495 | -51.63% | 0 |

The coarse, train-only state lookup was the 2025 feasible winner, but its
+1.52-bps gain is materially below the required +10 bps/trade.  The
regularised conditional L1 and quantile-LCB selectors selected no action,
rather than forcing a low-support regime change.

As in the surrounding slot-based replay research, continuous wallet
compounding across thousands of overlapping trades makes compounded account
growth and final-wallet fields non-interpretable.  They are not used for this
decision.  The table reports the stable exact-trade and fixed-capacity
portfolio comparisons: net bps/trade, additive total bps, tail loss, monthly
stability, Sortino, and drawdown.

### Frozen confirmation

| 2026 Jun–Aug frozen arm | Trades | Net EV/trade | Δ vs parent | Total net bps | CVaR10 | Worst month | Sortino | Max DD | Scheduled actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parent | 558 | +136.97 | — | +76,430 | -267.46 | +102.94 | 0.686 | -9.79% | 0 |
| 2025 winner: coarse Protect, 20-bps LCB | 558 | +136.97 | +0.00 | +76,430 | -267.46 | +102.94 | 0.686 | -9.79% | 0 |

The frozen lookup found no state with sufficient 2025-supported lower-bound
advantage in 2026, so it fell back to Parent everywhere.  This is the desired
conservative authority behaviour.  The regime controller is **not promoted**;
the rich parent exit remains canonical.

### Reproducibility

- `scripts/run_causal_sr_h4_remaining_regime_ablation.py` — exact persistent
  Parent/Protect/TrendRide counterfactuals, sampled-state oracle, coarse
  lookup, conditional L1/quantile selectors, and frozen confirmation.
- `data_perp/artifacts/causal_sr_h4_remaining_regime_r1r2_2025oof_2026frozen_20260901_v3`
  — final receipt, target-free feature contract, labels, OOF/frozen
  predictions, schedules, exact outcomes, portfolio ledgers, and manifest.

The final manifest states `no_exchange_calls: true` and records the fixed
parent/state/policy hashes.  It is an offline falsification receipt only.
