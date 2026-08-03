# Target audit — exact H12 post-cost persistence

**Research status: no promotion.  Decision: `STAGE_B_NO_EXECUTION_TARGET_ADVANCES`.**

This is the target-focused update to `TARGET_AUDIT_20260731.md`.  It incorporates the completed exact-path persistence experiment in `data_perp/artifacts/exact_h12_target_purity_ablation_20260731_v8/`.  It is deliberately limited to candidate-level entry selection under the frozen H12 execution policy.  It does not establish a deployable policy, a portfolio result, or an exit/timing rule.

## Executive diagnosis

The execution target is not failing because the post-cost path is economically meaningless.  It is failing because the current conversion turns several meaningful but different path questions into one globally ranked expected-value score without sufficiently reliable probability and cross-side calibration at the selected tail.

The new exact labels prove that the missing distinction is real:

- A candidate can clear the fixed cost before the adverse barrier and still finish the frozen H12 trade negative.
- That *giveback* state is large, economically severe, and separable from both adverse-first and timeout paths.
- A four-state target containing that distinction did **not** improve selected entry economics.  Therefore persistence is a valid path/support target, but it is not yet evidence for blending it into entry EV or the base/residual graph.

This is a useful negative result.  It rules out the hypothesis that a richer terminal event partition, by itself, repairs entry ranking.

## 1. Contract and causal status

All candidate arms use the same ordered 75,196 August--November 2024 evaluation candidates, frozen H12 policy, current-spread counterfactual cost accounting, raw decision-known feature contract, base OOS output, and causal 21-day prior-resolved mapping.  Tails are pooled **globally** after mapped expected value; no per-timestamp selection or side quota is used.

The exact label packs were materialised from 272,686 candidate-aligned 720-minute high/low paths.  They use:

- the fixed 100-bps gross cost floor and the frozen 2% adverse barrier;
- conservative `adverse_first_or_conflict` treatment when both barriers hit within the same minute;
- label availability at decision + 12 hours;
- no realised row cost or exit-time spread in model inputs.

For each hurdle, the new four-state target is:

1. `clear_then_retained`: clears cost before adverse, then ends H12 above the hurdle;
2. `clear_then_giveback`: clears cost before adverse, then ends H12 at or below the hurdle;
3. `adverse_first_or_conflict`;
4. `timeout`.

The current implementation estimates a four-class event simplex and state-conditional H12 net, then derives expected net.  This is causal in training/evaluation timing; it is not an inference feature yet.

## 2. What the label reveals

The four states are both populous and economically distinct in the training period.

| Side | Retained: rows / mean H12 net | Giveback: rows / mean H12 net | Adverse: rows / mean H12 net | Timeout: rows / mean H12 net |
|---|---:|---:|---:|---:|
| Long | 6,440 / +187.2 bps | 5,039 / -228.7 bps | 11,698 / -399.5 bps | 5,253 / -195.0 bps |
| Short | 8,329 / +145.8 bps | 4,038 / -205.8 bps | 9,655 / -393.5 bps | 6,408 / -197.0 bps |

So the label is economically sound.  In particular, `giveback` is not a marginal annotation: it is about 44% of long clear-first cases and 33% of short clear-first cases in the training support.  A reachability-only classifier systematically overstates entry value because it merges retained moves with these large negative givebacks.

## 3. Result: four states do not yet make a viable entry target

All figures are exact H12 net bps per globally selected candidate.  The control is the frozen OOS base opportunity score, causally mapped; it is a control only and is itself not economically viable.

| Arm | Top 1% | Top 5% | Top 10% | Causal threshold | Latest-month top-10 |
|---|---:|---:|---:|---:|---:|
| Frozen base control | -8.8 | -73.4 | **-104.0** | -106.8 | -78.7 |
| Exact reachability + side bridge (E12) | -34.3 | -75.1 | -111.4 | -103.4 | -77.2 |
| Four-state persistence, H0 (E13) | -55.7 | -104.0 | -114.8 | -134.5 | -172.4 |
| Four-state persistence, H25 (E13) | -39.1 | -102.5 | -118.8 | -136.0 | -163.7 |
| Four-state persistence, H0 + side bridge (E14) | -35.3 | -80.1 | -112.3 | **-100.6** | -83.8 |

E14 is the best persistence formulation, but it remains 8.0 bps below the control at global top 10%.  Its 400 paired day-block bootstrap improvement probability is only 35.0% (mean delta -7.0 bps; 5th--95th percentile -39.0 to +27.6 bps).  E13-H0 and E13-H25 improve in only 29.8% and 22.8% of resamples respectively.

The side bridge improves score balance (E14 global top 10% is 46.3% long versus E13-H0's 28.9%), but the economics remain worse, especially for its selected shorts:

| Arm, global top 10% membership | Long net | Short net |
|---|---:|---:|
| Frozen base control | -97.1 bps | -117.4 bps |
| Exact reachability + side bridge (E12) | -81.9 bps | -138.2 bps |
| Persistence H0 (E13) | -68.8 bps | -133.5 bps |
| Persistence H0 + side bridge (E14) | -72.3 bps | -146.9 bps |

This is not an argument for a side quota.  It is evidence that scores are still not comparable enough in common expected-bps units: mapping more short observations into the tail increases harmful short displacement.  The test remains global after mapping, exactly as the intended policy requires.

Every arm fails the same pre-declared gate: non-positive causal threshold, non-positive global top 10%, negative latest-month contribution/coverage, and a failing side contribution.  No target advances and no downstream base-target, auxiliary-stack, threshold, or portfolio selection should be inferred from it.

## 4. What is now known about the target problem

1. **A terminal direct net target is too noisy for the selected tail.**  Direct H12 net, residual net, fixed-cost and decision-known-cost variants all lost to the frozen base.
2. **Path semantics matter.**  Post-cost competing risk is consistently less weak than direct net.  Exact barrier timing did not add enough on its own.
3. **Reachability is insufficient.**  The exact label shows why: clear-first includes a large, negative giveback population.
4. **Persistence is meaningful but a flat four-class target is not enough.**  E13/E14 capture the correct states and still fail final selected economics.  The issue has moved from target definition to learnability, conditional calibration, and global score conversion.
5. **Cross-side comparability remains a real but incomplete repair.**  The side bridge improved the four-state causal threshold from -134.5 to -100.6 bps, yet worsened the short component.  A single raw-score-to-net side map is too blunt.
6. **Do not mistake broad association for entry quality.**  Earlier base-only execution diagnostics reached higher whole-panel IC while selecting a worse global tail.  Tail economics, latest-month support, and causal threshold remain the decision criteria.

## 5. Recommended target work, ordered by value of information

### A. Replace the flat four-state prediction with a hierarchical transition target

Test the same exact labels, but model the two causal transitions separately:

- `p_clear = P(clear cost before adverse)` on all candidates;
- `p_retain_given_clear = P(H12 net > hurdle | clear first)` only on clear-first candidates;
- separate conditional net models for retained, giveback, adverse, and timeout states.

Combine them algebraically into entry EV, for example `p_clear × [p_retain_given_clear × E(net | retained) + (1-p_retain_given_clear) × E(net | giveback)] + (1-p_clear) × E(net | adverse/timeout)`.  The expected value identity is unchanged, but each learner gets a simpler, denser question.  This is the most direct test of whether the current multiclass competition is the learnability bottleneck.

Acceptance: it must improve the frozen control in paired global top-10 bootstrap, not worsen top-1 or the latest month, and improve calibrated probability diagnostics.  It must be side-local in fitting but globally ranked only after a common-bps map.

### B. Make the retained-transition label soft and cost-aware

Hard `H12 net > 0` assigns nearly identical labels to -1 and -500 bps, and separates +1 from -1 bps even though neither is robust after execution uncertainty.  Test pre-specified soft labels such as:

`y_tau,h = sigmoid((exact_H12_net_bps - h) / tau)`, with `h ∈ {0, 25}` and `tau ∈ {50, 100, 150}` bps.

Use a proper binary loss and report Brier score, log loss, reliability by decile, and final global-tail economics.  The label remains H12-policy aligned; softness reduces boundary noise without using realised tail membership as a training weight.

### C. Test ordinal, not just regression or binary, terminal-value targets

Use cost-aware ordered bins such as `≤-200`, `(-200,0]`, `(0,100]`, `(100,250]`, `>250` bps (with fixed thresholds chosen before fitting).  Train an ordinal/cumulative probability model and reconstruct expected net from out-of-fold bin means.  This retains the severe downside distinction that plain binary retained labels discard, while reducing terminal-regression variance.

### D. Calibrate components, then assemble EV

Do not rely only on an isotonic map from a final raw EV score to realised net.  Calibrate `p_clear`, `p_retain_given_clear`, and adverse/timeout probabilities independently with strictly prior-resolved, side-local calibrators (temperature/beta calibration and isotonic as pre-specified alternatives).  Then map the assembled score into common expected-bps units using a regularised side intercept/slope, evaluated on a later block.  Report calibration separately by side, month, regime-transition bucket, and score decile.

This is more diagnostic than another raw-score bridge: it can identify whether the failure comes from reachability, persistence, conditional magnitude, or the final cross-side conversion.

### E. Measure label learnability before adding another target family

For every component above, emit strict-OOF side/month/regime metrics:

- class prevalence, ROC-AUC and PR-AUC for `clear` and `retain | clear`;
- Brier/log loss and calibration slope/intercept;
- rank IC for conditional magnitude on the applicable state only;
- stability of the highest predicted decile across folds/months;
- contribution to final globally pooled top 1/5/10% exact net when added sequentially.

If `retain | clear` has weak OOF discrimination, the appropriate conclusion is that it belongs in a later hold/exit action layer, not that it should be forced harder into entry selection.

### F. Test mechanism sensitivity without changing the final policy outcome

The 2% adverse barrier is a *diagnostic path partition*, not the final H12 outcome.  Pre-register a small geometry sweep (e.g. 1%, 1.5%, 2%, 3% adverse barriers and cost hurdles 0/25 bps) while always evaluating the same frozen exact H12 net.  Select no geometry from the evaluation period; use a validation block, then freeze it.  This determines whether the failure is partly caused by a path-state boundary that is not aligned with the realised fixed-policy exit.

### G. Use regime-transition fields only as causal conditioning, not a hidden target change

Fit/calibrate the hierarchical target with the existing decision-known regime-transition features, then report the component reliability by regime bucket.  If one component fails only in a distinguishable transition regime, test an interaction or a regime-specific calibrator with enough prior support.  Do not train separate entry models merely because one month is poor; require stable, causally identifiable regimes and out-of-time support.

## 6. What should not be done now

- Do not promote reachability, persistence, timing, MAE, target-price, or wait predictions into the entry graph from their standalone IC or label separation.
- Do not choose a threshold, side quota, or portfolio constraint to compensate for all-negative candidate-level expected net.
- Do not optimise against realised global top-k membership; global top-k is the **evaluation** rule after causal mapping, not a label available at decision time.
- Do not resume base/residual target selection or auxiliary-head stacking until at least one Stage-B execution target passes the complete economic and calibration gate.

## 7. Conditional follow-on once an execution target passes

Only after a target passes, test whether base/residual objectives should become more H12-cost aligned.  The right comparison is a frozen architecture with (a) the existing opportunity target, (b) a soft H12 post-cost target, and (c) a conservative residual around the causal mapped base value.  Evaluate all three through the selected execution target and the same globally pooled top-k rule.  Rising 24-hour base IC alone is not sufficient evidence: it can describe native alpha while H12 execution EV deteriorates.

## Artifact references

- Exact persistence experiment: `data_perp/artifacts/exact_h12_target_purity_ablation_20260731_v8/`
- Exact barrier event labels: `data_perp/artifacts/historical_exact_h12_postcost_events_20260731_v1/`
- Exact persistence labels: `data_perp/artifacts/historical_exact_h12_postcost_persistence_labels_20260731_v1/`
- Prior target audit: `TARGET_AUDIT_20260731.md`
- Roadmap status: `ROADMAP_ABLATION_STATUS_20260731.md`

## Addendum — hierarchical transition test (v9)

The recommended hierarchy was implemented and evaluated on the same 75,196 rows, folds, causal maps, costs and global evaluator as the flat four-state test.  It predicts `P(clear cost before adverse)`, then `P(retain | clear)`, and separately `P(adverse | not clear)` before recomposing a finite four-state simplex and state-conditional H12 net.  The probability-simplex error is effectively zero (mean `1.6e-17`), so this is not a reconstruction defect.

| Arm | Global top-1 | Global top-5 | Global top-10 | Causal threshold | Bootstrap improvement vs control |
|---|---:|---:|---:|---:|---:|
| Hierarchical H0 (E15) | -48.7 bps | -97.0 bps | -125.3 bps | -125.1 bps | 20.0% |
| Hierarchical H25 (E15) | -68.1 bps | -100.9 bps | -116.2 bps | -127.5 bps | 27.0% |
| Hierarchical H0 + side bridge (E16) | -22.9 bps | -93.1 bps | -112.1 bps | -112.3 bps | 31.5% |

E16 is close to the flat bridge at global top-10 (-112.1 versus -112.3 bps) but is not an improvement over the frozen base control (-104.0 bps), has no positive causal rule, and fails the latest-month and side gates.  It remains rejected.

The component diagnostics explain why a richer entry formulation does not help:

| Exact transition, H0 | OOF rows | Brier | Log loss | Prediction/actual Spearman |
|---|---:|---:|---:|---:|
| Clear cost before adverse | 75,196 | 0.223 | 0.634 | **0.290** |
| Retain H12 net conditional on clear | 31,268 | 0.235 | 0.664 | **0.106** |
| Adverse conditional on not clearing | 43,928 | 0.158 | 0.476 | **0.482** |

Thus the current causal raw feature set can identify adverse risk and, to a lesser extent, cost reachability.  It cannot reliably distinguish transient clear-first moves from retained moves at entry.  This makes the retention/giveback outcome a **diagnostic/action-layer candidate**, not an entry-EV component under the current features and model class.  The next target-only test, if Stage B is extended, should test soft/ordinal retention labels and component-wise calibration—not another event partition or a downstream auxiliary-feature blend.

## Addendum — pre-specified soft terminal-value labels (v10)

The last target-only test applied the pre-specified soft cost-aware label
`sigmoid((exact_H12_net_bps - 0) / tau)` directly to the execution learner at three temperatures (50, 100, and 150 bps).  It is not a top-k label, does not use future score distributions, and is converted back to common expected-net bps only through the same prior-resolved causal map as every other arm.

| Soft target | Global top-1 | Global top-5 | Global top-10 | Causal threshold | Bootstrap improvement vs control |
|---|---:|---:|---:|---:|---:|
| H0, tau=50 bps | -102.3 | -110.9 | -119.6 | -119.7 | 19.3% |
| H0, tau=100 bps | -148.7 | -106.1 | -119.0 | -113.5 | 17.8% |
| H0, tau=150 bps | -145.4 | -125.0 | **-112.8** | -112.9 | 28.8% |

None comes close to the required positive causal threshold or global top-10 economics.  Each has a severe top-1/top-5 reversal and a negative latest month.  The temperature effect does not form a credible improvement pattern: smoothing reduces some mid-tail loss at 150 bps but destroys the extreme tail and shifts 70--76% of global top-10 membership into shorts, which are all negative.

### Updated Stage-B conclusion

The following distinct, causally valid formulations have now failed on identical rows:

- direct and residual H12 net;
- fixed-cost and causal-cost proxy net;
- generic, proxy post-cost, and exact-minute competing risk;
- hurdle decomposition;
- flat exact persistence/giveback states;
- hierarchical reachability → retention/giveback → adverse states;
- soft direct cost-aware terminal-value labels.

The remaining issue should **not** be treated as an invitation to search more terminal target transforms.  The evidence instead points to two separable diagnostics:

1. the decision-time features contain meaningful adverse-risk and modest reachability information, but little reliable durable-move/persistence information; and
2. the assembled scores are not sufficiently stable or comparable at the global selected tail, especially when short observations displace long ones.

Accordingly, no current output is admitted to the next execution model.  `P(clear)` and `P(adverse | no clear)` are useful **diagnostic heads**; persistence/giveback is reserved for a future observed-path action layer; all final entry-EV targets remain rejected.  Base-target, supportive-feature stacking, entry thresholds, and portfolio replay stay blocked by the roadmap’s sequencing rule.

The next highest-value work is a causal **mapping/feature diagnosis**, not another target sweep: isolate score-map plateau/tie effects and cross-day/cross-side transport on frozen raw scores; measure transition-head learnability by regime/asset/liquidity slice; and identify decision-known features that discriminate retained from giveback paths.  Any resulting feature must first improve a frozen Stage-B execution formulation under the same globally pooled, causal evaluation.

## Addendum — map-plateau tie audit

The sealed v10 output was audited without refitting any target, score map, or policy.  The only counterfactual was a deterministic secondary ordering by raw decision-time score (then candidate ID) **within exactly equal causal mapped-EV values**.  This is a pooled-global diagnostic; the causal threshold is unchanged.

At the global top-10 boundary, the control has 69 tied candidates for 12 remaining slots; E16 has 31 for 11; E17-tau150 has 49 for 39.  All tied candidates have distinct raw scores, so the test is able to resolve every plateau deterministically.  It has no material economic effect:

| Arm | Stable mapped top-10 | Raw tie-break top-10 | Full-sample delta | Bootstrap 5th--95th |
|---|---:|---:|---:|---:|
| Frozen base control | -104.05 bps | -104.29 bps | -0.25 bps | -0.67 to +0.90 |
| Exact reachability + bridge | -111.45 bps | -111.42 bps | +0.03 bps | -0.52 to +0.79 |
| Hierarchical persistence + bridge | -112.05 bps | -112.03 bps | +0.02 bps | -0.73 to +0.46 |
| Soft terminal, tau=150 | -112.80 bps | -112.85 bps | -0.05 bps | -0.51 to +1.00 |

Therefore map plateaus are a reproducibility detail worth resolving deterministically, but they do **not** explain the 8--20+ bps target failures.  The next feature diagnosis must focus on missing decision-time information and calibration transport, not candidate-ID or isotonic tie artifacts.  Artifact: `data_perp/artifacts/exact_h12_score_map_tie_audit_20260731_v1/`.

## Addendum — retained-versus-giveback feature transport

A side-local read-only transport check was run on the existing 380 decision-known raw fields, restricted to exact H0 clear-first candidates.  Features were ranked only when their April--July 2024 and August--November 2024 Spearman signs agreed; evaluation AUC is descriptive, not a feature-selection result.

There is no strong entry-time retained-versus-giveback discriminator in the current contract.  The strongest transported single fields are all modest:

| Side | Stable field family | Train / evaluation IC | Evaluation AUC |
|---|---|---:|---:|
| Long | realised/prior/peer-residual volatility | +0.086 to +0.108 / +0.074 to +0.098 | 0.544--0.558 |
| Long | market and return dispersion | +0.044 / +0.080--0.099 | 0.547--0.559 |
| Long | funding-tail concentration / market spread | +0.046--0.058 / +0.074--0.085 | 0.544--0.550 |
| Short | peer-residual volatility | +0.090 / +0.075 | 0.546 |
| Short | short-covering, OI-breadth, market return state | about +/-0.055--0.064 / +/-0.043--0.059 | 0.527--0.537 |

This supports the hierarchical result rather than contradicting it: there is a weak volatility/liquidity/regime footprint, but nothing in the available snapshot that reliably tells a durable cleared move from a transient one.  The correct disposition remains:

- volatility, dispersion, funding, OI breadth, and liquidity features: candidates for a **separate, side-local retention research model** after a causal feature-design pass;
- clear/adverse predictions: diagnostic only until a final execution target passes;
- retained/giveback: reserve for action/hold policy work; do not blend into entry EV now.

The next feature work should materialise genuinely new decision-time information rather than reweighting the same fields: short-horizon order-book imbalance/depth change, spread/depth resilience, aggressor-flow or liquidation impulse, distance-to-liquidity/cluster structure, and explicit continuation-versus-exhaustion composites of market trend, volatility expansion, OI and breadth.  Each must be timestamped and strict-OOF evaluated first on `retain | clear`, then only tested in entry EV if it produces stable incremental candidate economics.

## Addendum — target-specific retention feature selection (v11)

The hierarchy was rerun with a side-local `retain | clear` feature list selected only from the pre-April 2024 base-training window.  This is the correct causal test of whether the previous hierarchy was limited by inheriting the clean-opportunity feature list.  It was not:

| Arm | Top-1 | Top-5 | Top-10 | Causal threshold | Bootstrap improvement probability |
|---|---:|---:|---:|---:|---:|
| Inherited hierarchy (E15) | -48.7 | -97.0 | -125.3 | -125.1 | 20.0% |
| Retention-selected hierarchy (E18) | -13.4 | -113.4 | -137.8 | -138.4 | 8.3% |
| Retention-selected + bridge (E19) | -45.0 | -88.8 | -134.2 | -121.5 | 9.8% |

The conditional retention head also weakens slightly (OOF Spearman 0.102 versus 0.106; Brier 0.237 versus 0.235).  Therefore the missing signal is not an omitted subset of the existing raw feature contract, and retention-specific selection must be **rejected** as an entry-model change.  This leaves the disposition unchanged: persistence is diagnostic/action-layer research only; no execution target advances.

## Addendum — existing multi-year historical substrate (2022--2024)

The earlier statement that historical information was unavailable was too broad. The workspace already contains three usable, non-promotional historical cohorts:

| Cohort | Rows | Period | Contract / permitted use |
|---|---:|---|---|
| Inverse PI | 50,880 | 2022-01 through 2022-07 | Separate inverse-perpetual product grid; never mix with linear-PF economics. |
| Linear PF reconstructed stack | 309,132 | 2022-08 through 2024-12 | Exact H12 frozen-policy/current-spread backcast with reconstructed base/residual OOF scores; target and transfer diagnosis. |
| Linear PF transition context | 118,734 | 2022-08 through 2023-12 | Same exact H12 labels plus 212 numeric decision-known transition/context fields; feature-information diagnosis. |

This corrects the data disposition: existing history is sufficient to reject or diagnose target hypotheses. It remains research-only because the base is a frozen backcast and historical L2 spread/flow is not factual live-parity execution data.

### All-existing OOF score audit

The compatible linear-PF 2022H2--2024 cohort was evaluated without mixing it with inverse PI. Residual alpha preserves modest broad execution ordering, but its economic tail remains negative:

| Score | Alpha rank IC | Exact H12 net rank IC | Pooled global top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| Base alpha | -0.009 | +0.007 | -164.4 bps | -176.4 bps | -172.7 bps |
| Residual alpha | **+0.174** | **+0.096** | +44.8 bps | -51.5 bps | **-95.3 bps** |
| Residual expected EV | +0.171 | +0.097 | +33.8 bps | -56.8 bps | -98.5 bps |

The 2022H2--2023 and 2024 subperiods show the same shape: the residual score identifies a very small positive extreme tail (+26 / +60 bps at top 1%), but neither has a positive global top-5 or top-10. This rejects the hypothesis that the original 2024 failure was simply too little data or one bad month. It is not an operative period-wide top-k policy.

### Existing transition-context learnability test

A new five-fold **symmetric calendar-block OOF** diagnostic was run on all 118,734 2022H2--2023 candidates. It deliberately does not require walk-forward transport: held-out calendar blocks may be predicted by models trained on later blocks. Its narrower question is whether existing decision-time context contains enough information to repair exact H12 net selection at all.

Models are side-local LightGBM models, but ranked together only after both emit the same exact-H12-net unit. Tails are one pooled global selection across sides and timestamps. `score_only` contains four reconstructed OOF base/residual outputs; `score_plus_transition` adds all 212 numeric decision-known context fields. No action, realised-path, timing, MAE, target-price, or wait fields are inputs.

| Features / target | Exact-net IC | Top-1 | Top-5 | Top-10 | `score > 0` net |
|---|---:|---:|---:|---:|---:|
| Score only / direct exact net | 0.120 | +4.4 bps | -51.2 bps | -80.0 bps | +4.8 bps |
| Score only / positive-net hurdle + conditional payoffs | 0.121 | +9.2 bps | -55.1 bps | -78.7 bps | +8.1 bps |
| Score + transition / direct exact net | **0.123** | -84.6 bps | -77.1 bps | -95.6 bps | -72.9 bps |
| Score + transition / hurdle + conditional payoffs | **0.123** | -35.8 bps | -71.6 bps | -89.9 bps | -34.6 bps |

The net-positive classifier is only modest (`ROC-AUC` 0.603 and `PR-AUC` 0.329 with score-only; 0.591 / 0.327 after adding transition fields). Thus, even with permitted non-walk-forward information sharing, the current transition panel does not show a latent, easily recoverable target repair. Its tiny IC improvement is economically harmful at the selected tail; transition fields as one unregularised bulk group are rejected for entry selection.

This does not make transition fields useless. It says their representation needs targeted regime-transition objectives and interactions, not an indiscriminate add-to-EV feature block. The revised priority is:

1. retain the multi-year panels as required diagnostic benchmarks;
2. preserve their separate product and frozen-backcast lineages;
3. test compact, mechanism-specific transition feature groups against the score-only control, with side-local selection and strict tail gates;
4. reserve factual L2/flow persistence claims for data that actually contains timestamped inputs.

Artifacts: `data_perp/artifacts/reconstructed_stack_all_eras_audit_20260731_v1/` and `data_perp/artifacts/historical_transition_target_learnability_20260731_v1/`.

### Compact transition-mechanism entry ablation

The preceding all-context result could have hidden a narrow useful mechanism. The same non-walk-forward five-calendar-block OOF protocol was therefore rerun as six predeclared causal groups, always against the same score-only control, exact H12 target, candidate IDs, side-local fits and pooled-global evaluator. This is a mechanism test, not a target-based feature search.

| Added mechanism / direct target | Net rank IC | Global top-1 | Top-5 | Top-10 | Zero-bps threshold |
|---|---:|---:|---:|---:|---:|
| Score-only control | 0.120 | **+4.4** | **-51.2** | **-80.0** | **+4.8** |
| State geometry | 0.098 | -82.2 | -80.1 | -93.8 | -62.3 |
| Regime-change dynamics | 0.100 | -17.2 | -67.5 | -95.2 | -18.1 |
| Breadth/correlation transitions | 0.089 | -68.0 | -82.5 | -98.8 | -64.1 |
| Flow/recovery transitions | 0.095 | -52.0 | -84.9 | -106.3 | -54.9 |
| Static transition state | **0.129** | -55.1 | -71.6 | -88.9 | -69.1 |
| All transition fields | 0.123 | -84.6 | -77.1 | -95.6 | -72.9 |

The hurdle version does not change this result: every compact group has negative top-10 and negative threshold economics. Selected long and short components are also negative for every arm. Static transition state raises broad IC but is economically harmful, an explicit example of a useful-looking model metric that must not enter the execution layer. All six groups are therefore **rejected as direct entry-EV additions**.

### Transition-identification result

The rejection above does not mean regime transitions are unlearnable. The existing frozen 2022--2023 spine contains 11,736 hourly causal rows and future-confirmed transition labels. A separate non-walk-forward, week-group OOF classifier study tested transition labels only; target fields and future geometry were excluded from inputs.

| Label / strongest causal group | Model | ROC-AUC | PR-AUC | Top-decile lift | Event recall | Disposition |
|---|---|---:|---:|---:|---:|---|
| Transition active | state geometry | LightGBM | **0.951** | 0.234 | **7.98×** | 95.7% | Diagnostic state monitor |
| Onset within 3h | static mechanisms | Logistic | 0.778 | **0.397** | **5.78×** | 68.8% | Research early-warning candidate |
| Onset within 6h | state geometry | LightGBM | 0.775 | 0.129 | 4.22× | 54.4% | Diagnostic only |
| Onset within 12h | state geometry | Logistic | 0.777 | 0.179 | 3.75× | 51.6% | Diagnostic only |

This proves an important architectural distinction:

- `transition_active_probability` can be a **separate regime/controller diagnostic**, because state geometry identifies already-active transition conditions reliably;
- onset probabilities are promising research signals but not automated alerts yet: the fixed top-decile rule produces roughly 51--64 false alerts per 30 days, so it needs a calibrated alert budget and a causal forward test;
- neither output is admitted to the current exact-H12 execution/residual model. The compact EV ablation proves that improving transition-label discrimination does not automatically improve globally selected entry economics.

The appropriate next use is to measure residual-score reliability and error conditional on these diagnostic states, then test a regime-specific calibrator only if its **causal** economics improve. Do not use it as a trade-side quota, a top-k allocation rule, or a substitute for the Stage-B execution gate.

Artifacts: `data_perp/artifacts/historical_transition_target_learnability_20260731_v2/` and `data_perp/artifacts/historical_transition_identifiability_20260731_v1/`.

### Residual-score reliability conditional on transition state

The final necessary guard is whether a strongly identifiable transition state can repair the residual score through a state-specific calibration. A read-only join of the 118,734 exact candidate rows to the frozen hourly transition spine shows that it cannot yet justify that experiment as an entry-policy change.

| Condition | Rows | Residual score net IC | Global-top-10 rows | Selected exact H12 net |
|---|---:|---:|---:|---:|
| State 0 | 101,740 | 0.114 | 9,940 | -97.9 bps |
| State 1 | 2,442 | 0.048 | 273 | -150.7 bps |
| State 2 | 7,320 | 0.139 | 843 | -59.2 bps |
| State 3 | 3,662 | **0.167** | 461 | **-43.5 bps** |
| State 4 | 3,570 | 0.129 | 357 | -89.0 bps |
| Confirmed transition-active | 3,374 | 0.136 | 404 | -82.3 bps |
| Onset within 3h | 2,562 | 0.107 | 333 | -119.6 bps |

States 2 and 3 are materially less poor than the background, but remain negative at the common selected tail; onset rows are worse, not better. This has two direct implications:

1. state is a plausible **reliability annotation** for investigation, not a license to trade a state selectively; and
2. a state-specific calibrator should not be fitted merely to create a positive mapping. It first needs a predeclared causal test showing a positive expected-net rule and an incremental improvement versus the common map.

The correct next regime experiment, if Stage B is extended, is therefore narrow: fit a regularised `score × {state 2, state 3}` calibration interaction only on prior-resolved labels, then evaluate it on a later matched block. It must preserve a pooled global ranking and beat the common map in threshold, top-10, latest-period and both-side economics. Until then, state labels remain controller diagnostics and do not enter the frozen entry graph.

Artifact: `data_perp/artifacts/historical_residual_transition_reliability_20260731_v1/`.

### Causal state-calibration ablation — rejected

The predeclared causal test was run rather than inferring a state gate from the diagnostic table. Each 2023 month was mapped using only the prior 90 days of H12-resolved labels. The control used the OOF residual score plus side; the challenger added intercept, score and side interactions only for states 2 and 3. Both arms then emitted common expected-net units, used the same candidate rows and were ranked once globally.

| Causal map | Net IC | Global top-10 | Threshold rows / net | Long top-10 | Short top-10 |
|---|---:|---:|---:|---:|---:|
| Common residual-score map | **0.106** | **-99.1 bps** | 258 / **+53.6 bps** | -79.3 bps | -113.5 bps |
| State 2/3 interaction map | 0.106 | -103.1 bps | 511 / -30.8 bps | -86.7 bps | -113.8 bps |

The challenger fails every relevant comparison: weaker calibration slope (0.64 versus 0.80), lower tail economics, negative threshold economics, a worse latest month (-106.5 versus -102.4 bps), and a worse long contribution. The control threshold result is very low support and unstable by month, so it does not qualify as an operational rule either.

**Decision:** reject state-specific score calibration as an entry-model repair. Preserve `transition_active` / onset classifiers only for regime monitoring and later controller research. Do not add state interactions, gates, quotas or state-specific thresholds to the current candidate entry graph.

Artifact: `data_perp/artifacts/historical_causal_state_calibration_ablation_20260731_v1/`.
