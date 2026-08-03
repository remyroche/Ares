# Roadmap ablation status — candidate-level exact-H12 alignment

## Decision

`STAGE_B_NO_EXECUTION_TARGET_ADVANCES`

The roadmap was applied as a diagnostic decision tree, not as a search for a single high tail metric.  The correct outcome at this point is a halted production-graph selection: no candidate execution target has earned the right to feed base-target, supportive-head, calibrator, or threshold selection.

## A. Correctness and contract validation — passed

- Same 75,196 candidate IDs, in the same chronological order, for the frozen base and all 11 target arms.
- Same April--November 2024 evaluation rows and April--July 2024 meta training.
- Same frozen H12 policy, exact gross/cost/net accounting, and 380 causal raw fields.
- Upstream base prediction is OOS; target score maps are prior-resolved 21-day prequential maps.
- Global tails are computed before any side split.  The earlier side-local weighted result is marked superseded.
- Realised row cost and exit-time spread are forbidden execution inputs.

## B. Execution-target formulation — completed, no arm advances

The full result is in `data_perp/artifacts/exact_h12_target_purity_ablation_20260731_v4/`.

| Formulation | What it tests | Diagnostic conclusion |
|---|---|---|
| E0 direct exact net | Whether one post-cost regression is enough | Rejected: weaker ranking, tail, threshold, month and side economics |
| E1 net residual | Whether base-to-net conversion is the missing piece | Rejected: residual worsens every operational criterion |
| E2 generic three-state | Whether first-event mechanism explains conversion | Rejected, but less weak than direct net at top 1/5% |
| E3 hurdle | Whether economic clearing is the central distinction | Rejected; 0-bps is strongest hurdle but still negative |
| E4 fixed-cost direct | Whether realised cost residue causes the failure | Rejected; worse than E0 |
| E5 causal cost proxy | Whether decision-time cost modelling repairs it | Rejected; best causal threshold among challengers but still -102.2 bps |
| E6 post-cost competing risk | Whether states should be defined at post-cost economics | Best new ranking diagnostic: -8.3 bps paired top-10 delta, still rejected |
| E7 side bridge | Whether cross-side score comparability is responsible | Helpful but insufficient: -9.4 bps paired delta, threshold -103.7 bps |

All candidates have negative causal-threshold exact net, negative pooled-global top-10 exact net, a negative latest-month contribution, and a failing side contribution.  The frozen base control also fails economics, although it has the strongest rank association and is nearest to breakeven at top 1%.

## Which layer is misaligned

The evidence points to **execution conversion and cross-side global score comparability**:

1. The base opportunity score preserves more H12-net ordering than the re-trained execution scores, so simply retraining a direct conversion model discards useful structure.
2. Post-cost competing-risk semantics improve the challenger materially over direct net, so the path/event definition is informative.
3. A side-specific prequential bridge improves the same post-cost formulation, showing that common bps calibration remains incomplete.
4. Neither change crosses the candidate economics gate.  The actionable issue is therefore not a threshold, not a side quota, and not a supporting-head blend.

The strict-OOS base-input check reinforces this.  A base-only post-cost execution head has the best whole-panel rank IC (0.222), but its global top-10 net is -120.2 bps and its paired improvement probability is only 9.0%.  Adding raw execution context reduces IC to 0.175 and top-10 to -123.6 bps.  A side-specific mapping does not recover the tail.  This is broad label learnability without decision-useful tail ranking, so it must remain diagnostic.

## Output disposition

| Output | Disposition | Reason |
|---|---|---|
| Frozen OOS base opportunity score | Reference/control only | Best available ordering, but no positive executable economics |
| E6 post-cost event simplex | Diagnostic-only candidate for next Stage-B iteration | Economically meaningful classes and best challenger ranking, but not enough final value |
| E7 side mapping bridge | Diagnostic-only | Shows an identifiable calibration issue; fails operational gate |
| E8/E9/E10 base-output post-cost heads | Diagnostic-only, explicitly not an inference input | Higher broad IC does not translate into global-tail or causal-threshold economics |
| Direct and residual net targets | Rejected | No incremental exact-H12 candidate value |
| Fixed/causal cost variants | Rejected as primary remedy | Do not repair ranking; cost residue is not the central failure |
| Reachability/adverse/magnitude/persistence/recovery heads | Deferred, diagnostic-only | Roadmap forbids stacking them before an execution target passes Stage B |
| Threshold or adverse gate | Not selected | No positive calibrated expected-net output to threshold or gate |

## Downstream stages intentionally not run

Stages C--F are not incomplete implementation work; they are explicitly **not valid yet** under the roadmap’s sequential contract:

- C: base-target selection must compare final economics through a frozen viable execution model.
- D: supportive-head cumulative and leave-one-group-out tests require a viable execution target to avoid attributing target failure to auxiliary heads.
- E: a final calibrator can be assessed, but no calibrator turns all-negative economics into a deployed expected-net rule.
- F: an absolute entry threshold/gate cannot be selected while all causal threshold results are negative.

## Next valid work

Remain in Stage B and materialise the missing causal target support before any downstream stack:

1. exact intrahorizon post-cost barrier timestamps (clear cost+h before adverse versus late/timeout), rather than the conservative final-gross proxy used by E6/E7;
2. a strictly prequential, side-aware calibration bridge that is evaluated for both tail membership and daily threshold decisions;
3. a feature-diagnostic ablation around E6 (base output, raw context, base plus context) to determine whether the conversion model is losing base structure or lacks causal conversion features;
4. only if an arm clears the Stage-B gate, proceed to base targets and strict-OOF supportive-head groups in the prescribed cumulative order.

## Exact-path repair outcome

The requested exact intrahorizon post-cost labels are now materialised from 272,686 candidate-aligned 720-minute paths.  This is stronger target data than the previous final-gross proxy, but the resulting exact competing-risk arms still fail Stage B:

- H0 exact path event: top-10 −120.7 bps, paired delta −15.5 bps.
- H25 exact path event: top-10 −117.4 bps, paired delta −11.2 bps.
- H0 side-mapped: top-10 −111.4 bps, paired delta −6.9 bps.

The label analysis identifies a concrete missing layer: **persistence after reachability**.  After reaching fixed post-cost value before adverse, 22,867 long and 17,674 short H0 candidates give back to non-positive exact H12 net.  The new persistence target pack separates `clear_then_retained` from `clear_then_giveback`, with labels available only at the exact H12 endpoint.

Disposition:

- Reachability: diagnostic/conditional head target; do not use as standalone execution EV.
- Persistence/giveback: first supporting-head candidate once a Stage-B execution formulation passes; currently materialised but not stacked.
- Base target, raw-plus-base execution head, threshold and adverse gate: remain unselected.

This is candidate-level research only.  It does not establish portfolio profitability, factual historical execution, or deployment readiness.

## Exact persistence update — v8

The exact-path persistence decomposition has now been tested in `exact_h12_target_purity_ablation_20260731_v8`.  It splits each H0/H25 path into `clear_then_retained`, `clear_then_giveback`, `adverse_first_or_conflict`, and `timeout`, with all labels resolving only at the H12 endpoint.

It validates the mechanism but does not change the Stage-B decision:

| Arm | Global top-10 exact net | Causal threshold net | Paired top-10 improvement probability |
|---|---:|---:|---:|
| Four-state persistence H0 | -114.8 bps | -134.5 bps | 29.8% |
| Four-state persistence H25 | -118.8 bps | -136.0 bps | 22.8% |
| Four-state persistence H0 + side bridge | -112.3 bps | -100.6 bps | 35.0% |

The H0 side bridge improves score balance and is the least weak persistence arm, but is still 8.0 bps below the frozen control at global top 10%.  The relevant next target test is therefore a **hierarchical** path model—`P(clear)` and `P(retain | clear)` calibrated separately, with state-conditional magnitude—not another flat multiclass blend or an entry-feature stack.  The detailed target diagnosis and proposed test matrix are in `TARGET_AUDIT_20260731_EXACT_PERSISTENCE.md`.

## Hierarchical persistence update — v9

The hierarchy has now been run.  It has a valid probability simplex and explicitly separates reachability, retention conditional on reachability, and adverse risk conditional on non-reachability.  It still does not advance:

| Arm | Global top-10 exact net | Causal threshold | Bootstrap improvement probability |
|---|---:|---:|---:|
| Hierarchical H0 | -125.3 bps | -125.1 bps | 20.0% |
| Hierarchical H25 | -116.2 bps | -127.5 bps | 27.0% |
| Hierarchical H0 + side bridge | -112.1 bps | -112.3 bps | 31.5% |

The diagnostic is nevertheless decisive: reachability has OOF Spearman 0.290 and adverse-given-non-reach 0.482, whereas retention-given-reach is only 0.106.  The raw decision-time features do not currently identify durable versus transient opportunity well enough to use persistence as entry EV.  Retention/giveback should remain diagnostic/reserved for an action layer; do not pass it into the execution model as a supportive feature.  If Stage B continues, test soft/ordinal cost-aware retention labels and component-wise calibration before changing the base target or adding auxiliary features.

## Soft-terminal target update — v10

Pre-specified direct soft terminal labels at 50, 100 and 150-bps temperatures all fail.  Their global top-10 nets are -119.6, -119.0 and -112.8 bps, with causal thresholds -119.7, -113.5 and -112.9 bps and bootstrap improvement probabilities of 19.3%, 17.8% and 28.8%.  The 150-bps arm is only the least weak new formulation; it also has a severe top-1/top-5 reversal and a negative latest month.

This completes the current target-formulation branch.  No more terminal-label transformation should be selected or added to the inference graph from this substrate.  The next valid Stage-B work is to diagnose causal score mapping and the missing decision-time features for retained-versus-giveback paths.  The target, base, supportive-head, threshold and portfolio dispositions remain unchanged: **no execution target advances**.

## Mapping and retention-feature closure

Two final Stage-B checks are complete:

1. Deterministic raw-score resolution inside exactly equal mapped-EV plateaus changes global top-10 results by at most 0.25 bps and has bootstrap intervals centred at zero.  Isotonic tie membership is a reproducibility detail, not the failure mechanism.
2. Side-local `retain | clear` feature selection fitted only before April 2024 worsens the hierarchy: −137.8 bps at global top-10 (−134.2 with the side bridge), versus −125.3/−112.1 for the inherited-feature hierarchy.  Its retention-head OOF Spearman falls from 0.106 to 0.102.

The complete Stage-B diagnosis is now:

- **Target formulation:** rejected. Direct, residual, hurdle, competing-risk, exact barrier, flat persistence, hierarchical persistence, and soft terminal labels do not identify positive final-policy H12 entries.
- **Mapping plateaus:** diagnostic only; fix deterministic tie handling but do not expect economics to change.
- **Reachability/adverse heads:** learnable diagnostic path outputs, but not admitted to entry EV because their composition does not improve final candidate selection.
- **Persistence/giveback:** economically real, but not reliably identifiable from the available entry snapshot; reserve for future observed-path action/hold research.
- **Base target, auxiliary stacking, threshold, entry gates and portfolio replay:** explicitly unselected.  The roadmap prohibits selecting them while Stage B fails.

The historical substrate may still reject weak ideas and validate contracts, but does not contain the historical L2/flow information required to test a credible entry-time persistence repair.  The next valid work needs a new timestamped factual/causal feature substrate—not more score or target optimisation on this panel.

## Existing historical-data extension

The final sentence above needs one qualification: the workspace does contain usable 2022--2024 historical exact-H12 research data. It has now been incorporated without mixing product contracts:

- 309,132 compatible frozen linear-PF OOF rows cover 2022-08 through 2024-12; their residual expected-EV score has broad net rank IC 0.097 but a negative pooled global top-10 (-98.5 bps).
- 50,880 inverse-PI rows cover 2022H1 and remain a separate contract; they are not pooled into linear-PF economics.
- 118,734 2022H2--2023 linear-PF rows carry 212 decision-known transition fields. Neither the full context nor six compact mechanism groups improves the Stage-B entry gate; every arm has negative global top-10 and zero-bps threshold economics.
- Transition **state** itself is learnable on the frozen hourly spine (active-state ROC-AUC 0.951), but candidate economics conditional on active/onset states remain negative. It is retained as a diagnostic/controller workstream, not as an entry-EV feature.

So the correct revised conclusion is not “older data is unavailable.” It is: **older data rejects the present target and bulk transition-feature repair, while supporting a separate regime-controller diagnostic layer.** Historical L2/flow remains absent only for the narrower claim that a factual persistence repair using such inputs has been tested.

Artifacts: `reconstructed_stack_all_eras_audit_20260731_v1`, `historical_transition_target_learnability_20260731_v1/v2`, `historical_transition_identifiability_20260731_v1`, and `historical_residual_transition_reliability_20260731_v1`.
