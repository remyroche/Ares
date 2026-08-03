# Full-universe Round B/C: strict causal implementation plan

## Scope and decision

This document specifies the next meta-head experiment only.  It does **not**
promote a model and it does not alter the base or any shared runner.

The role of the meta layer is a localized *trust/failure correction* for a
frozen opportunity-ranking base.  It must not be trained as a second global
opportunity model.  The evaluated book remains one globally pooled long/short
ranking in common net-bps units; no timestamp selection, side quota, or
post-evaluation rank threshold is permitted.

The current source panel is
`data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3`.  It supplies the
exact TP3/SL2 H12 contract:

- decision features at signal close `s`;
- entry at the next one-minute open at `s + 1h`;
- long/short signed first-touch exit at +3 ATR / -2 ATR, adverse-first on a
  same-minute conflict, otherwise the H12 timeout;
- `net_bps = gross_bps - 100` and label availability at entry + 12 hours.

The current frozen base prediction artifact contains predictions from 2024-04-01
to 2024-11-30 and its manifest proves the base was fitted through 2024-04-01.
It is therefore admissible for the 2024-04 onward meta experiment.  It is *not*
evidence that arbitrary earlier rows are stack-OOF.

## Non-negotiable causal contracts

For an outer evaluation block beginning at `T`:

1. Every outcome used to fit a meta target or an event-payoff map must satisfy
   `__label_available_at__ < T`.
2. Every base prediction used to train the meta must be base-OOF: its base-fit
   end time is strictly earlier than that candidate's decision time.  Persist
   `base_fit_end`, base model ID, base feature contract hash, and prediction
   timestamp with every row.  Fail closed if this cannot be verified.
3. Feature selection, all quantile cutoffs, conditional-payoff tables,
   shrinkage weights, score-to-bps calibrators, and combination-rule constants
   are fitted only on the outer training block.  They are frozen before scoring
   the outer evaluation block.
4. A high-base membership flag must be formed from a score threshold known at
   the time of the row.  It may never be a percentile calculated over the
   complete training month, complete evaluation month, or realised outcomes.
5. All fitted prediction maps use common net bps before the single global
   ranking.  Separate side models are allowed only if their outputs are mapped
   through the same fold-fitted common-unit contract.

The current static base is fitted before all April--November predictions.  For
the first implementation that is sufficient.  A later rolling-base experiment
must materialise base OOF predictions block by block; it may not reuse
in-sample base predictions merely to lengthen meta training.

## Prerequisite: causal expected-net base representation

Round B must consume a frozen `base_expected_net_bps`, not the legacy raw
`score_bps` residual.  It is constructed, for each candidate `i`, as:

`E_i[gross] = p_upper,i * mu_upper + p_lower,i * mu_lower + p_timeout,i * mu_timeout`

`E_i[net] = E_i[gross] - expected_cost_i`

For the current panel, `expected_cost_i = 100 bps`.  The event is the exact
first-touch class (`upper=0`, `lower=1`, `timeout=2`), and each `mu` is an
actual conditional *gross* payoff in bps under the TP3/SL2 execution contract.

Implement B1/B2 before Round B:

- **B1:** one pooled, prior-resolved conditional-payoff vector.
- **B2:** side-local vectors, each shrunk to B1:
  `mu_side,event = (n_side,event * mean_side,event + kappa * mu_global,event) / (n_side,event + kappa)`.
  Pre-register `kappa = 2,000` resolved rows per event, or select it on an
  earlier development block only.

The simplest B arm that improves the development global gross and net tails is
frozen.  B3 context-conditioned payoffs are not a prerequisite; if run, the
context bucket must be a decision-time feature with a minimum cell count and
the same side-to-global shrinkage.  Do not use a realised-event conditional
payoff for the row being scored.

`base_expected_net_bps` must be recomputed inside each chronological fold from
only resolved outcomes preceding the fold.  Store the three component means,
counts, shrinkage weights and source cutoff in the fold manifest.  The old
`score_bps` is not a substitute: it is a soft-label-weighted net mean and does
not prove a prequential economic calibration.

## Fold-local high-base population

The meta is trained only where it can change the final book.  The global
selection objective does not imply a per-timestamp threshold.

Use chronological calibration blocks inside the outer meta-training history:

1. At the start of block `b`, calculate one pooled common-bps threshold
   `q_p,b` from **prior-resolved, base-OOF** `base_expected_net_bps` values.
2. Mark each row scored in block `b` as eligible if
   `base_expected_net_bps >= q_p,b`.
3. Its target becomes available only after `__label_available_at__`; retain the
   row for fitting only once resolved.  The next block may use it in its
   reference distribution.
4. At outer evaluation start `T`, freeze `q_p,T` from the same prior-resolved
   distribution and apply it unchanged to every evaluation candidate.

Use blocks of seven calendar days initially; this is only to make the threshold
prequential, not a validation cadence.  The Round C candidate populations are:

- P0: all candidates;
- P1: `p = 0.50`;
- P2: `p = 0.70` (top 30%);
- P3: `p = 0.80` (top 20%).

Warm-up requirement: at least 20,000 prior resolved scores and at least 2,000
eligible resolved rows before a block contributes to fitting.  If unmet, use a
frozen global prior threshold only after logging its source and do not silently
backfill with future rows.  At inference/evaluation, non-eligible rows receive
exactly zero meta correction and retain their `base_expected_net_bps` score.

This fixes a material limitation of the already-run cost-clear model: it was
fit on all rows and therefore spent most of its capacity on candidates that can
never reach the global tail.

## Round B targets

All target labels use `t4_tp3_sl2_net_bps` and exact TP3/SL2 barrier exits.
All classifiers must report prevalence, AUC, Brier score, calibration slope,
and calibration intercept on the eligible evaluation population.

| Arm | Target definition for eligible resolved row `i` | Purpose |
|---|---|---|
| M0 | no meta | frozen B1/B2 base control |
| M1 | `I(net_i > 0)` | cost-clear reliability; retain as the reference classifier |
| M2-50 | `I(net_i < base_expected_net_i - 50)` | material base overestimation risk |
| M2-100 | `I(net_i < base_expected_net_i - 100)` | severe base overestimation risk |
| M3a | `I(net_i <= 0)` | failure probability for the hurdle model |
| M3b | `max(-net_i, 0)` fitted only where `net_i <= 0` | expected failure downside magnitude, in bps |
| M4 | `net_i - base_expected_net_i` | correctly scaled residual; Huber regression |
| M5 (diagnostic) | `max(base_expected_net_i - net_i, 0)` | positive-regret severity; test only after M2/M3 show discrimination |

M2 is deliberately not "base argmax correctness."  A class prediction can be
correct but economically weak, and can be class-incorrect without a large
economic loss.  M2 directly asks whether the economically calibrated base has
overstated the candidate by a predeclared material margin.

M3 uses two independent models, trained on the same eligible population:

`failure_penalty_i = P(net <= 0 | x_i, eligible) * E[-net | net <= 0, x_i, eligible]`.

The severity head must be trained only on failed training rows; it predicts a
non-negative bps magnitude (Gamma/Tweedie or log1p-Huber, inverted after
prediction).  Clip only at the fold-fitted 99th percentile of training failure
severity, not using evaluation outcomes.  Do not regress unconditional downside
and multiply it by a failure probability: that double-counts the zero mass.

M4 is meaningful only after B1/B2.  Its training residual must have near-zero
mean on held-out chronological calibration data; otherwise diagnose the base
economic map rather than interpreting a low-dispersion meta prediction as
alpha.

## Meta input contract

Use a strict subset selected within the existing meta-only config pools, not
the base pool and not all 477 candidates.  Preserve the existing family-capped
chronological selection process, but select it only on the eligible training
population and require at least 30 features only if the eligible sample
supports it.  For P3, cap at 20--30 features and increase minimum leaf size;
do not force 36 fields into a small tail sample.

The base and meta raw feature overlap remains forbidden.  Add the following
stopped-gradient, decision-time base-state fields to the meta design matrix;
they do not enter the base retrain:

1. `base_expected_gross_bps`, `base_expected_net_bps`, known cost (100), and
   `base_cost_margin_bps` (= expected gross - cost).
2. Raw simplex diagnostics from `[p_upper, p_lower, p_timeout]`: entropy,
   `p_upper-p_lower`, `p_upper-max(p_lower,p_timeout)`, largest probability,
   top-two margin, Herfindahl concentration, and probability width.
3. Event-payoff uncertainty from the frozen B1/B2 payoff vector:
   `sqrt(sum(p_e*(mu_e-E[gross])^2))`, plus upper-event payoff margin over
   cost.  This is an economic uncertainty measure, not a future label.
4. `base_score_prior_resolved_percentile`, high-base cutoff, and
   `base_margin_to_high_base_cutoff`; each uses the prequential reference CDF
   described above.  No current-day, period-wide, or realised-net percentile.
5. Feature-health diagnostics: selected-base missing-feature count/fraction and
   selected-meta missing-feature count/fraction before imputation.  The current
   panel also has `AE_reconstruction_error`, `mahalanobis_distance`, and stable
   cluster geometry.  They may be candidate meta fields under the user-approved
   historical AE/GMM assumption, but GMM posterior dimensions should be
   excluded initially; test the compact risk summary separately.

Do not add teacher disagreement or seed/geometry stability until they have
been materialised as true base-OOF decision-time fields.  A future target or a
model trained on the row must never be used to derive such a diagnostic.

Forbidden inputs: realised barrier class, realised path MFE/MAE, exit minute,
net/gross PnL, target certainty, post-entry information, future-period ranks,
and any side-calibrated score whose mapping used evaluation outcomes.

## Round C combination rules

Keep the base contribution monotone.  Tune exactly one small parameter grid on
the earlier development split, freeze the winner, then score the later outer
test once.  Compare against M0 on identical candidate IDs and day blocks.

1. **M1 multiplicative trust:** for positive base expectation only,
   `score = base_expected_net * calibrated_P(clear)`.  Else leave base score
   unchanged.  Calibrate probability with a chronological training-only
   isotonic or Platt map, with the calibrator nested before the model's scoring
   block.  Never refit it on the outer evaluation period.
2. **M2 overestimation penalty:**
   `score = base_expected_net - gamma * margin * P(overestimate_margin)`;
   test `gamma in {0.5, 1.0}` and each predeclared 50/100-bps margin.
3. **M3 downside penalty:**
   `score = base_expected_net - gamma * failure_penalty_bps`, with
   `gamma in {0.5, 1.0}`.  Clamp the penalty to the fold-fitted P99 and report
   the share clamped.
4. **Veto:** within the high-base population only, reject the lowest 10%, 20%,
   or 30% of calibrated M1 reliability (or highest M3 failure risk), then rank
   survivors strictly by `base_expected_net_bps`.  A veto changes coverage; it
   must report replacement candidates and must not cherry-pick a smaller
   top-k.  For a fixed global top-k, fill vacancies with next-best non-vetoed
   candidates; if there are insufficient survivors, report the shortfall.
5. **M4 additive residual:** `score = base_expected_net + predicted_residual`.
   This is the only residual combination permitted.

A rank blend remains a diagnostic reference only.  It must not be selected
from the same August--November period used to claim its result.

## Sequential execution and reporting

1. Run B1 then B2 on a predeclared development interval.  Freeze the smallest
   improving representation.
2. With that frozen representation, run Round B target comparison on P0 only:
   M0/M1/M2-50/M2-100/M3/M4.  Require classifier calibration and paired daily
   tail attribution before selecting one target family.
3. Run Round C only for the best Round B target: P0/P1/P2/P3.  Select the
   population on development, not the later test.
4. Run the five combination forms for the single chosen target/population;
   select on development, apply frozen to test.
5. Only then compare shared vs side-local meta fitting.  Side-local outputs
   must be converted using one pooled chronological OOF common-bps calibrator
   before global ranking; reject non-positive slope or tied score artifacts.

For every arm, save a manifest with: candidate count and eligibility count by
side/month, cutoff reference count and value for every block, base fit lineage,
feature list and missingness, selected rows, gross/net bps at 1/5/10/20%,
week-by-week net and counts by side, calibration metrics, common-bps mapping,
and paired day-block delta versus M0.  The comparison must use the exact same
evaluation candidate population and first-touch barrier PnL.

## Promotion gates and likely failure modes

No arm advances on a positive top 1% alone.  It must materially improve gross
top 10, strongly improve or reach non-negative net top 10, avoid a latest-month
or side collapse, and retain a credible paired day-block improvement.  A
result below the 100-bps cost floor remains a diagnostic, not a deployable
policy.

Most likely implementation errors are: using period-wide ranks for tail
membership; base predictions fitted on their own row; using the old raw score
as expected net; thresholding separately per side; fitting side-specific maps
without a pooled bps calibration; selection leakage during feature selection;
and treating the downside regression as unconditional.  Each should be an
assertion, not merely a manifest note.
