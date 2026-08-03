# Stage-E falsification and overlay acceptance ledger

Authoritative specification: `/Users/remyroche/.codex/attachments/763d7f1b-654b-4f81-ad7a-9376836b6711/pasted-text-1.txt`.

Status: `[x]` proven complete; `[ ]` incomplete or unproven. Conditional stages may be checked only when their required `NOT_RUN` disposition is directly established.

## A. Frozen scope and canonical binding

- [x] Preserve the Stage-B and Stage-C decisions and audit, rather than mutate, the frozen Stage-D decision.
- [x] Bind v9 as canonical and v10 only as its byte-identical companion.
- [x] Freeze model, feature lists, preprocessing, side calibrators, zero-bps margin, fill convention, costs, clear definition and continuation policy.
- [x] Change no entry selection/ranking, sizing, portfolio, stop geometry or continuation policy.
- [x] Prove whether a serialized frozen v9 model/tree artifact exists before E5.

## B. E1 causal sufficiency and target proximity

- [x] Inventory all 61 admitted A0 fields and all side/fold selections with required categories and lineage fields.
- [x] Audit decision-time exit-value bias, MAE, quantiles, side/month and latency sensitivity.
- [x] Falsify the claim that every selected field is decision-time available: `known_row_cost_bps` is outcome-derived.
- [x] Independently recompute the causal A0 subset from entry row and completed prefix, and fail closed on the unavailable selected cost field.
- [x] Produce the three E1 deliverables and pass the four E1 named tests.

## C. E2 minimal-information ablations

- [x] Run M0 through M8 on identical development rows/folds/seeds with training-only transforms.
- [x] Reproduce canonical M0 and report every required diagnostic/economic slice.
- [x] Report the fraction of M0 uplift retained by each arm.
- [x] Pass row/fold identity tests.

## D. E3 deletion and conditional permutation

- [x] Delete each declared A0 family from M0 on identical rows.
- [x] Permute conditionally within UTC day and side, preferably side x time-to-clear bucket.
- [x] Report delta MAE/IC/policy uplift/giveback capture/false-exit cost.
- [x] Identify whether one feature/family or interactions dominate.

## E. E4 execution sensitivity

- [x] Replay frozen M0 decisions at latency 0/1/2/5/10 minutes.
- [x] Apply added exit slippage 0/10/25/50 bps exactly once.
- [x] Stress exit-value estimate at -25/0/+25 bps without refitting.
- [x] Report jump, close-geometry and next-fill divergence ambiguity slices.
- [x] Report the maximum latency/slippage combination retaining positive uplift.

## F. E5 second sealed chronological validation

- [x] Select and seal the earliest later compatible unused period before results.
- [x] Freeze/hash every predeclared model/input/evaluator/gate component.
- [x] Apply the exact frozen model without retraining.
- [x] Report all required metrics and gates, or explicit `NOT_RUN` with direct blocking evidence.

## G. E6 complete-candidate frozen-policy overlay

- [x] Keep P0/P1 entry population identical.
- [x] Leave every non-clear candidate on the frozen policy.
- [x] Change only the first-clear binary action on eligible clear rows.
- [x] Report all candidate-level economics/slices/bootstrap and the required waterfall.
- [x] Add no portfolio constraints or sizing logic.

## H. Required tests and deliverables

- [x] All 16 exact named correctness tests exist and pass.
- [x] All 13 declared Stage-E data/report artifacts exist, with blocked stages explicitly `NOT_RUN`.
- [x] `STAGE_E_FINAL_REPORT.md` uses exactly one allowed Stage-E terminal decision.
- [x] Evidence-driven correctness report and sealed run manifest cover inputs/code/outputs/gates/limitations.

Completion requires direct requirement-by-requirement evidence and a fresh independent audit.
