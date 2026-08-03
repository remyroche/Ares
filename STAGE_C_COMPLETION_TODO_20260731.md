# Stage-C completion todo and acceptance ledger

Authoritative specification: `/Users/remyroche/.codex/attachments/ffff7d6d-a401-40dc-b3f9-f7e38fdc9772/pasted-text.txt`.

Status values: `[x]` proven complete, `[ ]` incomplete or unproven. A generated file is not completion evidence unless its contents prove the corresponding contract.

## A. Frozen scope and identity

- [x] Preserve candidate ID, side, decision timestamp, feature cutoff, H12 endpoint, label-availability timestamp, exact policy and cost IDs.
- [x] Keep inverse PI 2022H1 outside the linear-PF Stage-C population.
- [x] Define `retain_h0_given_clear` only on exact H0 clear-first support; persist validity, condition, side and month fields.
- [x] Persist optional H25 and continuous-net sensitivity labels as diagnostics only.
- [x] Freeze the immutable Stage-C compatible candidate-ID ledger/hash, with exclusions by group, month, side, symbol and reason; arm-level equality remains rechecked in Stage 1/Stage B.
- [x] Add purge/embargo evidence for every forward-label fold.

## B. Stage 0: sources, feature reuse and materialisation

- [x] Build a feature reuse map for every admitted/rejected F1-F7 candidate: existing field, new field, redundant/different-definition field, or rejected field, with formula and lineage.
- [x] F0: reproduce the exact E15 inherited retention-head control and persist the actual per-side feature list and hash.
- [x] F1: complete compact price-path continuation/exhaustion set, including direction changes, high/low recency, symmetric failed breakouts/rejection and side-normalised fields.
- [x] F2: complete volume confirmation/proxy set, explicitly named `*_proxy`, including persistence, shock decay/age, concentration and churn.
- [x] F3: complete volatility state/transition set, including side-adverse semivolatility, shock age/decay, ATR slope/acceleration and climax persistence.
- [x] F4: formally reject archived OI values because native observed/available timestamps and bounded staleness cannot be proven; admit no invented or indefinitely-filled value.
- [x] F5: formally reject archived funding values because native observed/available timestamps cannot be proven; use no future settlement/payment or indefinitely-filled value.
- [x] F6: complete timestamp-eligible universe ranks, breadth, dispersion, relative strength, confirmation and isolated-move fields; persist universe size and membership digest.
- [x] F7: reject raw/learned regime fields until candidate-level strict OOF/prequential provenance is materialised; runner C7 is blocked.
- [x] F8: define and document a small fixed set of transparent composites; no fitted final-period weights.
- [x] Register F1-F8 in retention-only config keys. Do not alter production base/residual/general-meta keys.
- [x] Ensure all generators are vectorised/batched, deterministic and memory-bounded for the full materialisation.
- [x] Record for every admitted/rejected feature: name, source, frequency, lookback, minimum observations, formula, units/range, side normalisation, available timestamp, publication delay, missingness, staleness, point-in-time safety, live parity and proxy/factual status.
- [x] Report historical coverage/start, product coverage, symbol/month/side missingness and all excluded rows.

## C. Stage 1: strict conditional-retention OOF

- [x] Use side-local v11 model class, fixed hyperparameters, arm-invariant paired seeds, folds, purge and embargo.
- [x] Fit only clear-first rows; never map non-clear rows to negative retention.
- [x] Run C0-C8 separately; C4/C5/C7 are source-blocked and no broad `C0 + all raw features` arm exists.
- [x] Perform train-fold-only availability filtering, clipping, correlation reduction and importance/stability selection; cap each mechanism at 32 incremental fields.
- [x] Freeze feature decisions before final OOS; prove no final-period labels or importance entered selection.
- [x] Reproduce/reference E15 conditional diagnostics and state the exact common-row contract difference.
- [x] Report rows/prevalence by side/fold/month plus ROC-AUC, PR-AUC, Brier, log loss, logistic calibration slope/intercept, decile reliability, top-decile retention/lift, Spearman with exact H12 net and net by decile.
- [x] Report long, short, monthly, symbol/asset concentration, feature-importance concentration and missingness sensitivity.
- [x] Add paired C0 comparisons with fold/month deltas and 200-replicate UTC-day bootstrap uncertainty; classify by stability, side transport and symbol breadth.
- [x] Record observational transition-monitor status without gates, quotas, sizing or trade suppression; F7 is source-blocked.

## D. Stage 2: compact combination

- [x] Select survivors using development OOF evidence only and record the predeclared gate: zero groups survive.
- [x] Record `C_best_compact` as correctly NOT_RUN because the admission set is empty.
- [x] Record leave-one-mechanism-group-out as correctly NOT_RUN because no compact arm exists.
- [x] Classify every group as control/diagnostic-only/rejected with conditional, calibration, month, side and bootstrap evidence.

## E. Frozen Stage-B hierarchy test

- [x] Refactor hierarchy API so an alternative matrix can be supplied only to `P(retain | clear)`.
- [x] Record H_control/H_new component-identity proof as correctly NOT_RUN because no new retention head passed admission; existing frozen API isolation remains unchanged.
- [x] Record H_new_bridge mapping-only proof as correctly NOT_RUN because the bridge was not admissible under Outcome A.
- [x] Record prequential split-retention mapping as correctly NOT_RUN because Stage 1 admission failed.
- [x] Enforce the gate: H_control, H_new and H_new_bridge are NOT_RUN because Stage 1/2 admission did not pass.
- [x] Record hierarchy/base comparison as correctly NOT_RUN under Outcome A.
- [x] Record global top-k/threshold/side/calibration outputs as correctly NOT_RUN under Outcome A.
- [x] Record hierarchy paired bootstrap as correctly NOT_RUN under Outcome A.
- [x] Preserve the pooled-global-after-common-bps contract; no alternative ranking or quota was produced.
- [x] Apply Outcome A exactly and promote no threshold, policy, portfolio or production change.

## F. Named correctness tests

- [x] `test_all_features_available_by_decision_timestamp`
- [x] `test_rolling_features_use_trailing_data_only`
- [x] `test_cross_sectional_features_use_timestamp_eligible_universe`
- [x] `test_oi_values_respect_source_timestamp_and_staleness` (formal rejection path)
- [x] `test_funding_values_respect_observation_timestamp` (formal rejection path)
- [x] `test_no_future_funding_payment_used`
- [x] `test_no_inverse_pi_rows_mixed_with_linear_pf_rows`
- [x] `test_clear_first_population_matches_frozen_label_manifest`
- [x] `test_retention_labels_exist_only_on_clear_first_support`
- [x] `test_comparison_arms_use_identical_candidate_ids` (frozen common cohort; model-arm equality is rechecked in Stage 1)
- [x] `test_upstream_transition_predictions_are_oof` (F7 rejected until provenance exists)
- [x] `test_feature_selection_uses_training_data_only`
- [x] `test_scalers_and_clippers_fit_on_training_data_only`
- [x] `test_no_final_oos_feature_selection`
- [x] `test_stage_b_test_changes_only_retention_head_features` (Stage-B API isolation exists; execution correctly gated off)
- [x] `test_cost_and_execution_policy_ids_remain_frozen` (frozen IDs persisted; no Stage-B run or policy mutation)
- [x] `test_global_ranking_occurs_after_common_bps_mapping` (contract preserved; no ranking run under Outcome A)
- [x] Test proxy naming for orderbook/depth/aggressor/liquidation/spread tokens.
- [x] Replace asserted/declared Stage-0 audit booleans with checks derived from materialized evidence; later-stage checks remain explicitly pending.

## G. Required deliverables

- [x] `feature_source_lineage.parquet`
- [x] `feature_coverage_by_month_side_symbol.parquet`
- [x] `feature_availability_report.md`
- [x] `retention_feature_dictionary.json`
- [x] `retention_feature_groups.json`
- [x] `retention_conditional_oof_predictions.parquet`
- [x] `retention_conditional_results.parquet`
- [x] `retention_conditional_calibration.parquet`
- [x] `retention_feature_stability.parquet`
- [x] `retention_compact_feature_manifest.json` (zero survivors; Stage 2 NOT_RUN)
- [x] `retention_leave_group_out_results.parquet` (gated NOT_RUN record)
- [x] `stage_b_incremental_retention_results.parquet` (gated NOT_RUN record)
- [x] `stage_b_incremental_retention_summary.md` (Outcome-A NOT_RUN explanation)
- [x] `feature_disposition.yaml` with one complete record per F0-F8 group and requested development evidence.
- [x] `correctness_test_report.json` generated fail-closed from actual Stage-1 evidence.
- [x] `run_manifest.json` with matching code/data/output hashes, periods, folds, paired seeds, counts, frozen contracts, search breadth and limitations.

## H. Final twelve-question report

- [x] 1. List every newly materialised OHLCV/OI/funding mechanism.
- [x] 2. Distinguish genuinely new fields from pre-existing transformations.
- [x] 3. State and prove causality and live reproducibility.
- [x] 4. Quantify compatible historical coverage after lineage filtering.
- [x] 5. State whether any mechanism improved strict-OOF `retain | clear`.
- [x] 6. Quantify stability by side, month, symbol and regime.
- [x] 7. Compare the compact combination with individual groups, or state why the compact arm is correctly not run.
- [x] 8. State whether the better retention head improved frozen Stage B, or state why Stage B is correctly not run.
- [x] 9. Compare against the frozen base control, or state why the comparison is gated off.
- [x] 10. State global top-10 and causal-threshold exact-net results, or state why they are gated off.
- [x] 11. Give exactly one supported terminal disposition.
- [x] 12. Identify the remaining information gap and the evidence for it.

Completion requires every applicable item above to have direct evidence. Conditional Stage-2/Stage-B items may be recorded as correctly not run only when the preceding predeclared gate fails and the stop decision is fully evidenced.
