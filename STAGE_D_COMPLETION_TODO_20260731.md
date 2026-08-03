# Stage-D completion todo and acceptance ledger

Authoritative specification: `/Users/remyroche/.codex/attachments/6b4dbb8a-b97e-41a0-91f2-ca193188944a/pasted-text-1.txt`.

Status: `[x]` proven complete; `[ ]` incomplete/unproven. Conditional stages may be checked only when an explicit admissibility gate proves `NOT_RUN` is the required outcome.

## A. Frozen scope and prohibited changes

- [x] Preserve both terminal Stage-B/Stage-C decisions and do not reopen Stage B.
- [x] Implement only `EXIT_NOW` versus `CONTINUE_FROZEN_POLICY` at first exact H0 clear.
- [x] Prove no entry target/model/threshold, base/residual model, auxiliary entry stack, sizing, portfolio constraint, quota, entry gate, stop/trailing geometry, partial exit, RL or production policy changed.
- [x] Freeze candidate identity, side, executable entry, H12 endpoint, clear/adverse geometry, cost model, execution policy, 1m path source, product separation and split conventions.

## B. Source and lineage audit

- [x] Read and validate every required Stage-C/target/path artifact and seal before editing.
- [x] Inventory every OI/funding raw source and derived sidecar with provider/product, event/observation/ingestion/availability timestamps, cadence, revision, missingness, fill, maximum staleness and live parity.
- [x] Give each OI/funding source exactly one allowed lineage disposition.
- [x] Admit A6/A7 only with proven observation/availability timestamps, bounded staleness, correct product and live parity; otherwise reject.
- [x] Produce `oi_funding_lineage_report.md`, `oi_funding_source_ledger.parquet`, `oi_funding_availability_tests.json`.

## C. Stage D0 action population and counterfactuals

- [x] Use exact H0 clear-first population only and reproduce frozen clear-first identity support.
- [x] Require unambiguous first clear, actionable next executable exit, complete paths for both actions and complete causal action features.
- [x] Persist candidate/side, entry, first-clear, action-decision, action-execution, horizon-end, label-availability, execution-policy, cost-model and path-source IDs.
- [x] Set `action_decision_ts = first_clear_ts`; use the repository causal next-executable action convention.
- [x] Materialize gross/cost/net for exit-now and continue, with cost deducted exactly once and shared entry cost not double counted.
- [x] Prove continuation reproduces the frozen policy and `delta_continue_bps = net_continue_bps - net_exit_now_bps`.
- [x] Persist optional binary diagnostics only; primary target is conditional expected incremental bps.
- [x] Manually reconcile a representative sample against exact 1m paths before modeling.
- [x] Produce population manifest, counterfactual parquet and correctness audit markdown.

## D. Stage D1 deterministic baselines

- [x] Evaluate B0 always continue and B1 always exit on identical action rows (canonical v4; v3 superseded after bootstrap audit).
- [x] Report gross, cost and net overall/by side/month/time-to-clear/volatility; volume and regime carry explicit unavailable/rejected dispositions.
- [x] Quantify signed baseline uplift, loss avoided, false-exit opportunity cost and whether mechanical first-clear exit is superior without mislabeling signed uplift as giveback.

## E. Causal action features A0-A9

- [x] A0 minimal action state: frozen entry controls, side, time/bars to clear, gross/net-at-action, row cost and policy geometry.
- [x] A1 path geometry to clear, including observed-to-date MFE/MAE, efficiency, direction, slopes, acceleration, jump concentration and giveback-to-date.
- [x] A2 completed-candle/rejection structure through action time only.
- [x] A3 volume confirmation/churn/shock features disposition: `REJECTED_SOURCE_UNAVAILABLE`; immutable exact-1m paths contain no volume and no aligned replacement passed lineage.
- [x] A4 volatility/instability/climax features, including side-adverse semivolatility.
- [x] A5 action-timestamp eligible-universe market/cross-sectional features with universe size and membership digest.
- [x] A6 OI path features only if lineage admitted; otherwise `REJECTED_LINEAGE`.
- [x] A7 funding/crowding path features only if lineage admitted; otherwise `REJECTED_LINEAGE` and no future settlement/payment.
- [x] A8 regime fields only from strict candidate/action-level OOF/prequential sidecar; disposition: `REJECTED_OOF_LINEAGE`.
- [x] A9 small fixed transparent composites using only admitted components and no final-OOS fitted weights; blocked-component composites recorded `NOT_RUN`.
- [x] Every feature has exact formula, lookback/window, units/range, minimum observations, side normalization, availability, missingness/staleness, factual/proxy status and lineage.
- [x] Every path feature stops at action decision; future MFE/MAE/outcome/path is rejected.
- [x] Generators are deterministic, vectorized/batched and memory bounded.
- [x] Produce feature dictionary/groups/lineage/coverage artifacts.

## F. Validation and experiment sequence

- [x] Use strict chronological OOF/prequential action blocks with resolved labels only and purge at least the remaining maximum label horizon.
- [x] Hold eligible IDs, folds, model class/hyperparameters, paired seeds, labels, cost/policy and evaluator fixed across arms.
- [x] Fit side-local where architectural parity requires it and map outputs into comparable incremental-bps units before pooling.
- [x] Run D0 A0 control, cumulative D1-D9 in declared order, with blocked groups recorded rather than silently skipped.
- [x] Run mechanism-only incremental comparisons against A0 where practical.
- [x] Perform all filtering/clipping/correlation reduction/feature selection on training rows only.
- [x] Select groups on development evidence only; freeze before final OOS.
- [x] If groups survive, build compact model with controlled feature count and run identical-row leave-one-group-out.
- [x] Replay thresholds 0/25/50 bps only; select any margin on development only; never use global top-k action selection.

## G. Metrics and admission gates

- [x] Report MAE/Huber, Spearman delta IC, ROC/PR, Brier/log loss, calibration slope/intercept and realized delta by predicted decile.
- [x] Report rows, action rates, B0/B1/learned net, paired uplift versus both, gross/cost, side/month/latest/worst/time-to-clear and paired UTC-day bootstrap.
- [x] Report givebacks exited, retained cases falsely exited, opportunity cost of false exits and loss avoided by correct exits.
- [x] Require each group to pass development prediction, calibration, month/symbol/side support and learned-policy economics gates.
- [x] Require final model positive paired uplift versus both baselines, non-negative latest period, no failing side, stable incremental-bps calibration, support and no lineage violation.
- [x] Never interpret action-layer uplift as entry-system profitability or deployability.

## H. Named correctness tests

- [x] `test_action_population_is_exact_clear_first_population`
- [x] `test_first_clear_timestamp_matches_frozen_label_pack`
- [x] `test_action_decision_precedes_action_execution`
- [x] `test_action_features_available_by_action_decision`
- [x] `test_path_features_stop_at_action_decision`
- [x] `test_future_mfe_mae_are_rejected`
- [x] `test_exit_now_counterfactual_cost_applied_once`
- [x] `test_continue_counterfactual_matches_frozen_policy`
- [x] `test_delta_equals_continue_minus_exit`
- [x] `test_action_arms_use_identical_candidate_ids`
- [x] `test_folds_use_resolved_action_labels_only`
- [x] `test_scalers_fit_on_training_data_only`
- [x] `test_feature_selection_uses_training_data_only`
- [x] `test_cross_sectional_universe_is_timestamp_eligible`
- [x] `test_oi_requires_verified_availability_timestamp`
- [x] `test_funding_requires_verified_availability_timestamp`
- [x] `test_no_unbounded_oi_or_funding_forward_fill`
- [x] `test_transition_features_require_oof_lineage`
- [x] `test_side_outputs_are_mapped_to_incremental_bps`
- [x] `test_action_threshold_uses_development_data_only`
- [x] `test_no_entry_or_portfolio_policy_is_changed`

## I. Required deliverables

- [x] `stage_d_action_population_manifest.json`
- [x] `stage_d_action_counterfactuals.parquet`
- [x] `stage_d_action_counterfactual_audit.md`
- [x] OI/funding three-artifact provenance pack.
- [x] Action feature dictionary/groups/lineage/coverage pack.
- [x] Action OOF predictions/model results/calibration/stability/bootstrap pack.
- [x] `stage_d_compact_feature_manifest.json`
- [x] `stage_d_leave_group_out_results.parquet`
- [x] `stage_d_action_policy_replay.parquet`
- [x] `stage_d_feature_disposition.yaml`
- [x] `STAGE_D_FINAL_REPORT.md`
- [x] Evidence-driven `correctness_test_report.json`.
- [x] Sealed `run_manifest.json` with matching input/source/code/output hashes, periods, folds, seeds, counts, gates, blocked stages and limitations.

## J. Final thirteen-question report

- [x] 1. Is always exiting at first clear better than always continuing?
- [x] 2. How large is the giveback cost under the frozen policy?
- [x] 3. Does observed path-to-clear improve continuation prediction?
- [x] 4. Which mechanisms add information?
- [x] 5. Are improvements stable by side, month, symbol and time-to-clear?
- [x] 6. Were OI and funding admitted or rejected by causal lineage?
- [x] 7. Does the learned action policy improve net versus both baselines?
- [x] 8. How much loss is avoided by correct exits?
- [x] 9. How much retained upside is sacrificed by false exits?
- [x] 10. Does the latest period pass?
- [x] 11. What is the paired day-block uncertainty?
- [x] 12. What is the sole terminal decision?
- [x] 13. What remains blocked because no entry model has passed?

Completion requires direct evidence for every applicable item. Use exactly one terminal model decision; `OI_FUNDING_CAUSAL_LINEAGE_UNRESOLVED` may accompany it only as a data-lineage disposition.
