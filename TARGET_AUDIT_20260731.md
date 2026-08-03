# Target audit — corrected exact-H12 entry selection

**Status: research-only, no promotion.**  This audit supersedes the target conclusions from `exact_h12_target_purity_ablation_20260731_v1`.  That output accidentally admitted realised row cost and exit-time spread as model inputs, which made the target nearly directly observable.  It is invalidated and must not be used for model, target, or policy selection.

This document assesses only candidate-level entry ranking under one frozen H12 execution policy.  It does not assess sizing, concurrency, exposure, allocation, exits, timing actions, or portfolio replay.

## 1. What is now verified

The corrected experiment is at `data_perp/artifacts/exact_h12_target_purity_ablation_20260731_v2/`.

- **Rows and IDs:** 75,196 ordered evaluation candidates for each of the seven scores (frozen base control plus six target arms).  Candidate identity and finite mapped-score coverage are asserted identical before any comparison.
- **Outcome:** `exact_h12_net_current_frozen_spread_counterfactual_v1`: exact H12 gross minus the frozen policy’s cost, exactly once.
- **Execution policy:** `historical_current_frozen_spread_counterfactual_h12_v1` is unchanged across every arm.
- **Input timing:** each raw feature is in `raw_380_8ace0586b1fb2b40`; candidate time is no later than decision time; entry is decision time; label availability is decision + 12 h.  The sidecar makes each of these assertions.
- **Cost safety:** `row_cost_bps` and `exit_half_spread_bps` are excluded by construction.  The model may use only decision-known estimated entry spread, entry half-spread, fixed barrier geometry, and entry price as execution-context fields.
- **Training sequence:** base fit Apr-2023--Mar-2024; its score-to-net map is OOS Apr--Nov.  Target/meta training is Apr--Jul-2024 with expanding, prior-month-only OOF history; final target scores are Aug--Nov.  All target score maps use the preceding 21 resolved days only.
- **Evaluation:** all top-k diagnostics are pooled-global across candidates, never per timestamp.  Causal daily threshold selection and gross/cost/net reporting are included separately.

The panel is still candidate-conditioned and current-spread-counterfactual.  It has no historical L2 reconstruction, full-universe coverage, or bit-exact pre-2025 execution geometry.  It cannot establish deployability or a promoted winner.

## 2. Targets tested on exactly the same rows/features/folds

| Arm | Definition |
|---|---|
| Control | Frozen OOS base opportunity score, causally mapped to H12 net; not re-trained in this comparison |
| E0 | Direct regression of exact H12 net |
| E1 | Exact-H12 net residual around the strict prequential base score-to-net map |
| E2 | `P(clean/adverse/timeout)` times event-conditional exact-H12 net; `clean` is favorable first, `adverse` is adverse/conflict first, and timeout is a third state |
| E3-0/25/50 | Two-state hurdle expected net at 0, 25, or 50 bps: probability of clearing the hurdle plus conditional upside/downside |

All fits are unweighted.  This was deliberate: the question here is target purity, not a tail-weighted policy optimisation.

## 3. Corrected result

### Pooled-global top-k exact net (bps per selected candidate)

| Arm | Top 1% | Top 5% | Top 10% | Top 20% |
|---|---:|---:|---:|---:|
| Frozen base control | **-8.8** | **-73.4** | **-104.0** | **-124.4** |
| E0 direct net | -47.7 | -115.5 | -123.3 | -136.5 |
| E1 residual net | -69.5 | -107.1 | -125.0 | -132.4 |
| E2 three-state | -35.2 | -94.4 | -117.1 | -135.1 |
| E3 hurdle 0 | -96.6 | -99.9 | **-113.7** | -125.5 |
| E3 hurdle 25 | -90.3 | -106.2 | -119.8 | -139.4 |
| E3 hurdle 50 | -107.9 | -110.1 | -122.8 | -139.1 |

None of the target arms beats the frozen base at any reported global top-k.  The least-bad new target is E3-0 at top 10%, but it remains 9.5 bps below the control.  The paired 122-day bootstrap confirms the direction: E3-0 is -11.8 bps on average versus the control, and only improves in 28.5% of resamples.  E2 is next at -13.7 bps and 25.0%; the rest are materially weaker.

The control is not profitable enough either: its top 1% is -8.8 bps.  That is a useful near-miss, not a viable entry rule.

### What is driving the ranking failure

- The new scores shift global top-10 selection from **65.7% long** for the control to only 28--33% long.  Their short selections are substantially worse than their long selections.  At top 1%, E2’s long component is +55.8 bps, but its short component is -117.0 bps; its pooled result is -35.2 bps.  This is a score-comparability/side-selection failure, not evidence for a side quota.
- The frozen base retains the strongest overall rank association with realised exact net (Spearman 0.183).  The best new arm, E3-25, reaches only 0.172; E3-0 is 0.170 and E2 0.166.
- The daily causal threshold rule is negative for every arm, including the control (-106.8 bps).  The problem is ranking/calibration at the candidate tail, not merely a bad static threshold.
- Absolute high prediction buckets are too small and unstable to serve as a rule.  For example, the control’s `>200 bps` bucket has 51 rows and +70.9 realised bps; E2 has 186 rows and +80.0 bps.  Those observations justify a properly pre-specified tail-learning test, not an operational threshold.

## 4. Target diagnosis

### The direct exact-net target is valid but not currently helpful

E0 is the cleanest answer to “will this frozen-policy entry make money after cost at H12?”  It loses 19.2 bps to the base at top 10% and degrades the side mix.  E1 confirms that a net residual around the base map does not repair this.  Therefore the immediate issue is not simply that the old base objective lacked an H12 net residual.

### The event decomposition is directionally more useful, but not sufficient

E2 is the best top-1% new target (-35.2 bps) and the clean event is economically distinct in meta training:

| Side | Clean rows / mean net | Adverse rows / mean net | Timeout rows / mean net |
|---|---:|---:|---:|
| Long | 7,773 / +64.2 bps | 17,760 / -320.2 bps | 2,897 / -157.6 bps |
| Short | 8,719 / +83.2 bps | 15,864 / -302.6 bps | 3,847 / -157.2 bps |

So the first-event decomposition is learnable and economically meaningful.  Its conversion into an unconditional, globally comparable expected-net score is the weak link.  It needs an event definition closer to the actual post-cost admission decision and a calibrated cross-side score bridge.

### The hurdle idea is not wrong; its current implementation is too blunt

H=0 is better than H=25/50 at top 10%, meaning the available data does not support demanding extra profit before classifying a candidate as attractive.  But E3-0 still trails the control.  The likely limitation is not the two-part expectation identity; it is that a single pooled representation has to learn an abrupt tail decision, conditional upside/downside, and cross-side comparability from a candidate-conditioned panel.

### Cost is not the main source of label noise in this particular panel

In the evaluation period, row cost averages 99.5--99.6 bps monthly with 1.4--2.0 bps standard deviation.  The tiny row-specific variation comes from realised/exiting information and is invalid as an input.  Thus it should not be learned directly.  Use a frozen cost floor/proxy for training and retain exact realised net only for evaluation.  This removes a small causal mismatch without pretending that outcome-known exit costs are knowable at entry.

## 5. Recommended target changes, in order

1. **Use a causal gross-minus-fixed-cost training target.**  Train on `exact_h12_gross_bps - C_policy`, where `C_policy` is a fixed, versioned 100-bps cost floor (and test a decision-known estimated-spread proxy as a separate arm).  Evaluate both against untouched exact net.  This preserves the H12 policy objective while removing the non-causal realised-cost remainder from the training label.  Do not feed realised cost or exit spread back in.

2. **Replace generic clean/adverse with post-cost barrier states.**  For each hurdle `h in {0, 25, 50}`, create a fixed-policy, ATR-normalised/price-space three-state label over 12 h:
   - `clear_h_first`: reaches `C_policy + h` gross before an adverse barrier;
   - `adverse_first`: reaches adverse barrier first;
   - `timeout_or_late`: neither event first by H12.

   Then estimate both the event simplex and conditional H12 gross.  This makes the classifier answer “will it earn enough to cover policy cost safely?” rather than “did it touch a generic favorable barrier?”

3. **Calibrate side scores before global top-k, without enforcing a side quota.**  Fit strictly prequential, side-specific reliability maps from raw score to the same causal gross-minus-fixed-cost target; map both sides into common expected-bps units; only then pool and rank globally.  The acceptance test is whether short scores stop displacing better long candidates at the global tail.  A side quota is not an acceptable substitute.

4. **Make the residual target policy-relevant but conservative.**  Keep the base opportunity score as an input.  Train the residual only on a clipped excess-return target such as `clip(gross - C_policy - base_map, -300, +300)` and use a low-capacity model/shrinkage.  Require paired improvement in top 1%, 5%, and 10% plus latest-month coverage before retaining it.  E1 shows an unrestricted exact-net residual is not enough.

5. **Test tail-oriented, proper losses—not historical tail membership weights.**  Use soft, fixed-label transforms of the post-cost hurdle outcome (for example a logistic temperature sweep around 0 bps and 25 bps) and class-balanced/focal *classification* loss for rare `clear_h_first` states.  Pre-specify the temperatures and judge them out of sample.  Do not train by weighting samples according to realised global top-k membership; the prior v1 pooled-tail error is precisely why that is unsafe.

6. **Add target reliability diagnostics as gatekeepers.**  For every arm and side, retain fixed-bin and quantile-bin calibration, Brier/log loss for each event, event-class prevalence, gross/cost/net by global top 1/5/10/20%, and 400 paired day-block bootstrap draws versus the frozen control.  Reject a target if it improves an aggregate mean but loses the latest month or fails to improve both pooled ranking and calibration.

7. **Only then test richer supporting labels.**  Add time-to-clear-cost, maximum adverse excursion before clear, and H12 path slope as *separate prediction heads*.  They must not be blended into entry EV until each provides incremental, OOF-calibrated improvement over the post-cost event head.  This keeps timing/wait/target-price actions in the separate action layer.

## 6. Next ablation matrix

The next target batch should keep the same candidate panel, feature set, folds, evaluation, and global top-k accounting.  It should contain these pre-registered arms:

| Arm | Training outcome | Purpose |
|---|---|---|
| T0 | Existing frozen base control | Immutable comparator |
| T1 | H12 gross - fixed 100 bps | Remove outcome-known cost residue |
| T2 | H12 gross - decision-known cost proxy | Test whether causal cost variation is incremental |
| T3 | Post-cost 0-bps three-state expected gross/net | Align event semantics to break-even |
| T4 | Post-cost 25-bps three-state expected gross/net | Test an economically meaningful buffer |
| T5 | T3 with side-specific prequential calibration bridge | Directly test the observed global side-mix failure |
| T6 | Clipped residual around base map, T3 label | Test whether residual adds policy information without dominating base |
| T7 | Soft post-cost hurdle labels, pre-specified temperature grid | Test learnability of the tail boundary |

For an arm to proceed, it must beat T0 in paired top-10 bootstrap mean with a majority of improving resamples, not worsen top-1 or latest-month net, preserve or improve calibration, and avoid a mechanically harmful short displacement.  Passing that screen would still be research evidence only until the full-universe factual execution dataset exists.

## 7. Code and artifacts

- Corrected runner: `scripts/run_exact_h12_target_purity_ablation.py`
- Alignment materialiser: `scripts/materialize_historical_exact_h12_alignment_sidecar.py`
- Safeguard tests: `tests/test_run_exact_h12_target_purity_ablation.py`, `tests/test_materialize_historical_exact_h12_alignment_sidecar.py`, `tests/test_long_raw_base_residual_h12_ablation.py` (11 passed)
- Corrected outputs: `data_perp/artifacts/exact_h12_target_purity_ablation_20260731_v2/`
- Alignment contract: `data_perp/artifacts/historical_exact_h12_alignment_sidecar_research_only_20260731_v1/`

## Addendum — roadmap Stage-B expansion

The initial E0--E3 run was extended in `exact_h12_target_purity_ablation_20260731_v4` without changing candidate IDs, folds, policy, costs, features, or evaluation rows.  The additional arms were deliberately targeted diagnostics, not a second optimisation sweep:

| Arm | What it isolates | Top-10 net | Paired top-10 delta vs base | Causal-threshold net |
|---|---|---:|---:|---:|
| E4 | Direct H12 gross minus a fixed, versioned 100-bps policy cost | -130.8 | -26.2 | -139.6 |
| E5 | Direct H12 gross minus a training-only, decision-time spread cost proxy | -123.3 | -18.7 | **-102.2** |
| E6-0 | Post-cost (0-bps) clean/adverse/timeout decomposition | -112.4 | **-8.3** | -134.8 |
| E6-25 | Post-cost (25-bps) competing-risk decomposition | -115.8 | -13.8 | -130.1 |
| E7 | E6-0 with a strictly prequential, side-specific score-to-net bridge | -111.2 | -9.4 | -103.7 |

None clears the Stage-B economics gate.  E6-0 is the strongest *ranking* diagnostic among new target formulations, and E7 shows that a side-specific mapping reduces the failure somewhat.  Neither has positive causal-threshold or pooled-top-10 net, so neither is an execution-model feature or inference architecture choice.

This narrows the diagnosis:

- **Not primarily realised-cost residue:** replacing exact net with fixed-cost or causal-cost-proxy labels does not improve the global tail.
- **Partly path semantics and score comparability:** post-cost competing risk materially improves over direct net, while the side bridge makes the newest month and causal threshold less negative.  It remains far below the evidence required for adoption.
- **Base opportunity remains the stronger available ordering:** its pooled net Spearman is 0.183 versus 0.171 for E6-0 and 0.173 for E7.  It is still net-negative and therefore a reference/control, not a deployable model.

Accordingly, the roadmap stops after Stage B.  Base-target selection, OOF supportive-head stacking, a target/support interaction check, and threshold/gate optimisation are intentionally deferred.  Running them now would optimise downstream layers against an execution formulation that has failed its own causal decision gate.

### Base-output feature diagnostic

The final Stage-B check used the strongest target diagnostic (E6-0) in three feature configurations:

| Configuration | Top-10 net | Paired top-10 delta vs base | Whole-panel rank IC | Interpretation |
|---|---:|---:|---:|---|
| Raw context only (E6-0) | -112.4 | -8.3 | 0.171 | Best challenger tail, still fails economics |
| Strict OOS base output only (E8) | -120.2 | -15.5 | **0.222** | Better broad ordering but catastrophic executable-tail selection |
| Raw context + strict OOS base output (E9) | -123.6 | -21.0 | 0.175 | Raw conversion features destroy rather than preserve base value |
| E8 with side-specific bridge (E10) | -122.5 | -17.8 | 0.177 | Mapping does not repair the base-only tail |

This makes the disposition sharper: do not use the E8/E9 execution-head score as an inference input merely because its IC is high.  It is a diagnostic that the post-cost head learns broad label structure but fails to identify the economically usable global tail.  The next Stage-B work needs exact barrier-time labels and tail-calibration diagnostics, not a base-target or auxiliary-head search.

### Exact one-minute barrier-time result

The next Stage-B iteration materialised `exact_1m_h12_postcost_barrier_first_fixed100bps_v1` directly from all 272,686 aligned 720-minute high/low paths.  It uses fixed 100-bps gross cost floors, the frozen 2% adverse barrier, and treats same-minute dual hits as adverse/conflict.  This replaces the final-gross proxy, not the policy outcome used for evaluation.

| Arm | Definition | Top-10 net | Paired delta vs base | Causal threshold |
|---|---|---:|---:|---:|
| E11-0 | Exact H0 clear-cost-before-adverse event simplex | -120.7 | -15.5 | -143.6 |
| E11-25 | Exact H25 event simplex | -117.4 | -11.2 | -135.7 |
| E12 | E11-0 with side-specific prequential mapping | -111.4 | -6.9 | -103.4 |

Exact reachability does **not** repair the execution target.  It does, however, reveal the missing mechanism: among H0 clear-first paths, 41.1% of long and 33.6% of short examples end with non-positive exact H12 net.  Conditional means are only +22.6 bps long and +33.2 bps short; clear-then-retained paths average roughly +195/+150 bps, while clear-then-giveback paths average −224/−197 bps.

Therefore:

- `P(clear cost before adverse)` is a valid **reachability diagnostic/head target**, not a sufficient entry-EV target.
- `P(retain positive H12 net | clear cost first)` and its complement `P(giveback | clear cost first)` are the economically coherent next **conditional persistence head** targets.
- Neither may be added to an inference graph yet.  It must first show incremental strict-OOF value relative to a viable execution target; otherwise it remains diagnostic/action-layer support.

The complete materialised target packs are `historical_exact_h12_postcost_events_20260731_v1` and `historical_exact_h12_postcost_persistence_labels_20260731_v1`.
