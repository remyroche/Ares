# Target Audit — 2026-07-30

## Decision

The native 24-hour alpha target is not an adequate proxy for the realised
12-hour, post-cost execution policy outcome.  Exact H12 **net** is the correct
primary economic outcome for evaluating and training the execution/residual
layer.  It is not yet sufficient as a monolithic base label: the observed
failure is principally conversion from opportunity into post-cost policy value
at the selected global tail, not absence of all predictive rank information.

Do not promote any target from the current historical raw-base study.  Its
best result is still negative, its population is candidate-conditioned, and
its nominally global weighted residual targets contain a material weighting
bug described below.

## Evidence inspected

- Native-alpha versus exact-policy identical-cohort waterfall in
  `ACTIONABLE_PIPELINE_ROADMAP_20260724.md` (especially the February--April
  rows at lines 4305--4319 and 8719--8729).
- Current legacy base/residual target construction in
  `extreme_price_movements/base_residual_label_ablation.py`.
- Current exact-H12 net and competing-risk label contracts in
  `scripts/materialize_execution_ev_cost_aware_competing_risk_labels.py` and
  `scripts/materialize_execution_action_target_pack.py`.
- Auxiliary path targets in `extreme_price_movements/path_auxiliary_targets.py`.
- The completed raw-feature candidate-conditioned experiment
  `long_raw_base_residual_h12_ablation_20260730_v1`.

## What the evidence says

### 1. The base-label/execution mismatch is real

On the identical February--April 2025 rows, long-side native 24-hour alpha
rank IC rises from 0.155 to 0.162 to 0.226.  Against exact H12 net it is only
0.090, 0.093 and 0.143.  The corresponding *pooled-global* top-10 book has
gross/cost/net of +49.38/+100.25/-50.87 bps, +17.05/+100.09/-83.03 bps and
+41.86/+100.21/-58.35 bps.

This is not a claim that the alpha score has no useful order.  Its order is
positive, and even exact-H12 net IC improves in April.  It is a claim that the
native target does not identify a post-cost profitable extreme tail under the
actual exit policy.  Base rank must no longer be judged by native-target IC
alone.

### 2. Exact H12 net is correctly constructed, but its scope is bounded

The exact label contract has explicit decision, H12 endpoint and availability
times, and asserts `gross - row_cost = net` exactly once.  Its action target
pack also has complete one-minute paths and fixed 1/2/3/4/8/12-hour,
post-cost outcomes.  This is the right target family for execution economics.

The available 2023--2024 raw panel is nevertheless only an old
selected-top30/monitor candidate population, uses current-spread
counterfactuals, and lacks historical L2 and bit-exact pre-2025 policy
geometry.  A successful result there would still not establish a full-universe
or portfolio-valid target.  A negative result is useful as a warning, but not
a complete refutation of a target on the actual candidate universe.

The gap is material rather than semantic.  The archived raw source has 285,510
candidate rows in April--December 2023 and 483,012 in January--November 2024,
but the exact stage covers only 42.35% and 46.62% of those raw rows on staged
paths.  Unstaged candidate windows have missing or incomplete one-minute data.
The stage also declares `execution_parity_claim=false`, has no historical
spread/L2, and reconstructs rather than recovers the deployed ATR geometry.
This is why a broader counterfactual backfill can be useful research but cannot
turn the present target study into promotion evidence.

### 3. The latest raw base/residual target study is informative but provisional

The study uses 12 months of base fit, eight frozen base-OOS months, then four
months of residual fit and four untouched months.  It selects final books
globally after a causal 21-day map.

| Global book | Top 1% | Top 5% | Top 10% | Top 20% |
|---|---:|---:|---:|---:|
| Frozen base-score control | -121.30 | -168.54 | -175.10 | -179.80 |
| Best provisional arm: net-hurdle base + clean-tail residual | -46.25 | -122.95 | -129.54 | -146.17 |

All values are bps/trade and all are negative.  The best arm improves the
bounded control by 45.56 bps at top-10, but its selected long subset is
-34.31 bps and its short subset is -155.26 bps.  It is therefore neither a
usable policy nor evidence that a global target has been solved.

The three base labels are also very close to monotone transforms of the same
final net result on this panel: their Spearman correlations with exact net are
1.0000 (net-hurdle), 0.9968 (risk-penalised) and 0.9911 (timely-clean).  The
experiment primarily changes tail emphasis and calibration; it does **not**
yet test genuinely distinct economic mechanisms.

### 4. Correctness issue: global-tail target weighting is currently side-local

`run_long_raw_base_residual_h12_ablation.py` correctly implements a pooled
tail helper, but calls it only *after* filtering the residual training frame to
one side.  Thus `global_tail_weighted_residual` and
`clean_tail_weighted_residual` weight the top 10% of each side, not one pooled
global top 10% across both sides.  Final evaluation is pooled global, but the
training target is not.  This affects the headline best arm.

The previous artifact must consequently be described as a
**candidate-conditioned exact-H12 target diagnostic with side-local tail
weighting**, not as evidence for a pooled-global policy-tail objective.
Correct and regenerate it before comparing weighted target arms.

### 5. Current soft-label composition still mixes distinct decisions

The legacy label-HPO implementation combines a native 24-hour soft execution
target with 12-hour MFE, MAE, timing, early-path and slope components.  The
components are valuable supporting labels, but this mixture has three
problems:

1. it spans native 24-hour alpha and 12-hour execution/path horizons;
2. MFE, MAE and timing are not themselves post-cost outcomes;
3. a slow but profitable immediate entry is penalised even though the entry
   ranker does not execute a wait/reprice action.

Those terms should not be multiplied into an entry target unless their effect
is evaluated through an explicit action counterfactual.  Entry ranking and
wait/target/exit decisions are separate layers.

### 6. The five path targets should be conditionally modelled

The current labels are side-normalised, bounded/log-transformed and have useful
support columns.  That is good infrastructure.  Their distributions remain
zero-inflated, censored or clipped:

| Head | Current issue | Better role |
|---|---|---|
| Peak MFE | Zero mass includes no meaningful opportunity; magnitude is currently mixed with reachability. | Model `P(reach meaningful MFE)` first, then conditional peak magnitude/quantiles. |
| Time to meaningful MFE | Non-reach is right-censored at 12h, not an ordinary regression time. | Discrete survival/hazard or CDF heads for 1/2/4/8/12h, conditioned on a causal reach model. |
| MAE before MFE | Its semantics differ for hit versus non-hit paths; clipping hides tail risk. | Separate adverse-first probability and conditional MAE quantiles; use in action risk, not direct entry score. |
| Bars to adverse stabilisation | It is a noisy realised turning-point coordinate and not a direct economic payoff. | Predict early adverse/recovery hazards; reserve it for exit/tighten decisions. |
| Future slope | Strong conditional persistence/rank signal, but zero/capped mass changes the unconditional meaning. | Conditional slope/persistence quantiles after meaningful reach; use as payoff-duration context. |

This follows the observed learnability: conditional peak magnitude and future
slope are meaningfully rankable, while the event-probability conversion into
unconditional trading value is the bottleneck.

## Recommended target architecture

### Base layer — opportunity, not an implicit action policy

Train side-local base challengers on a cost-aware, exact-H12 **clean
opportunity** target, using all actual base candidates when the full-universe
substrate exists:

`P(clean economic opportunity before adverse/timeout)`

The upper barrier must be row-cost plus a predeclared buffer and the adverse
barrier must be tied to the executable policy geometry.  Retain native alpha
as a frozen control, but make this a genuine direct challenger—not a mixture
of a 24-hour alpha target and 12-hour realised-path descriptors.

### Residual / execution layer — decompose expected policy net

Replace the single transformed-net label with calibrated components on the
same exact policy outcome:

`E[net] = p_clean × μ_clean + p_adverse × μ_adverse + p_timeout × μ_timeout`

where the probabilities form a cost-aware clean/adverse/timeout simplex and
each `μ` is the conditional realised **net** policy payoff.  A simpler
two-state ablation is acceptable first:

`P(net > hurdle) × E[max(net-hurdle, 0) | clear] - P(net ≤ hurdle) × E[max(hurdle-net, 0) | fail]`.

Compare this against direct exact-net residual regression on identical rows,
features, map and global-book evaluator.  The existing cost-aware
competing-risk materializer can supply the event labels; it must not use
realised action/path fields as inference features.

### Timing and exits — separate action layer

Train wait/reprice and hold/partial/exit/tighten policies only against the
existing action target pack and explicit counterfactual replays.  They may use
OOF auxiliary predictions but must not alter the entry score merely because a
realised path was fast, slow, or had a better hindsight exit.

## Required corrections and ablations, in order

1. **Repair and rerun the target study.** Compute one immutable pooled global
   top-10 tail mask over the complete April--July residual-training frame
   before side splitting.  Attach its weights by candidate ID; use the same
   precomputed weights in each residual fold.  Add a test proving the selected
   count, membership and side share equal a single pooled global book.  Mark
   the existing weighted-arm result superseded, not promoted.
2. **Run a target-purity matrix on identical rows.** Compare: direct exact net;
   net residual; cost-aware clean/adverse/timeout expected net; and the
   two-state clear/upside/downside formulation.  Keep the base feature and
   model process fixed for this comparison.
3. **Tune only predeclared target parameters.** On training-only data choose
   hurdle (0/25/50 bps), soft-label temperature, residual shrinkage and
   tail-loss weight using pooled-global top-1/5/10 economic value, worst-month
   and worst-side gates—not rank IC alone.  Freeze before final OOS.
4. **Use a strict prequential companion for score-to-net calibration.** The
   current chronological blocked OOF trains each held-out block using other
   blocks, including later data.  It is acceptable for the explicitly
   non-walk-forward model diagnostic, but not as strict causal mapping
   evidence.  Keep it separate and add an expanding/prequential OOF map
   companion.
5. **Measure candidate and portfolio scope separately.** First prove positive,
   stable candidate-level global-book exact net; only then replay concurrency,
   exposure and asset limits.  Add a portfolio shadow-cost sensitivity only
   after a candidate-level arm is positive.
6. **Build the full-universe factual historical substrate before promotion.**
   It needs exact-H12 labels for every raw candidate, the deployed geometry,
   historical economics and point-in-time feature lineage.  The current
   candidate-conditioned counterfactual panel is insufficient for target
   selection or deployment.

## Acceptance gates

A future target arm is eligible to proceed only when it has all of the
following:

- exact decision-to-H12 policy target and one-time row-cost accounting;
- side-local feature/model fitting with OOF upstream inputs;
- target/HPO selection based on one pooled global book, never timestamp-local
  ranks or side quotas;
- positive pooled top-10 net with acceptable top-1/top-5 behaviour, no
  failing latest month or side, and paired day-bootstrap uncertainty;
- a causal, resolved-label-only map and no score/map overlap;
- a valid full-universe factual panel and a subsequent constrained portfolio
  replay before any promotion.
