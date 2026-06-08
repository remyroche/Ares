# Extreme Price Movements Inference Mismatch Investigation

Status: updated, 2026-06-04.

## 2026-06-04 Requested Todo Status

1. Rerun shadow/live inference: checked. Latest run-scoped ledger rows for `20260525_010004_nopenalty` include feature/mask counts, candidate decisions, source-parity coverage, rank rejection/acceptance reasons, and open-position decisions. The latest observed signal bar was `2026-06-04T17:00:00Z`; one shadow/live row was accepted (`USDC/USD:USD`, short asset-OI), and the other current-run decisions were rank rejected. Selected base/meta feature snapshots and raw/rank-normalized prediction fields are persisted in the prediction ledger.
2. Compare actual training path vs actual live path: checked. The comparison used the training-path feature store produced by `run_pipeline.py features`, not a parity-only feature helper. For the reconciled live batch, overlapping selected decision features matched the training feature store with `0.0` max absolute delta, and logged model-input scoring reproduced base score, meta score, policy rank, auction rank, and normalized rank with `0.0` max absolute delta.
3. Localize policy-score mismatch: checked. `simple_policy_optimiser/rank_reference/*.parquet` is generated from the verified policy-OOS prediction handoff, not from the current deployed final-fit live scorer. The earlier non-zero final-fit replay-vs-rank-reference deltas are therefore a score-source/OOS-contract distinction, not a feature-generation mismatch.
4. Fix policy-OOS contract: checked. Policy-OOS generation is trained-universe filtered, manifests now declare train-meta-state provenance, rank-normalization metadata, source model-state hashes, source-artifact preflight status, and execution-source metadata. The optimiser fails closed if policy rows are outside the trained/inference universe or if manifest provenance is invalid.
5. Regenerate verified policy candidates: checked. The current verified candidate set uses the t+10 execution model and includes gross/net returns, delay-gap fields, spread/slippage/fee drag fields, and adverse-move rejection simulations. Current active candidate count is `27,485`, with `25,564` rows using delayed 1m proxy candles and `1,921` rows using the theoretical 15m fallback.
6. Rerun `simple_policy_optimiser`: checked for the guarded verified policy-OOS source. Deployment rank bands and gates were reconciled against the inference rank-reference logic. Threshold-band economics remain a risk: the global deployment-rank subset is positive and near the target (`14,692` rows, `60.2%` net hit, `270.5` bps mean net), but some per-strategy threshold bands are weak and long-dist remains below a clean 60% net-hit threshold in the per-strategy candidate view.
7. Historical inference replay: checked for the four active deployed heads. Replay through the actual inference candidate path has exact selected-feature parity for all four active strategies (`21,500` compared feature cells, no missing rows, no mismatches above `1e-6`, max absolute delta `0.0`). The first remaining divergence is the expected final-fit live scorer versus frozen policy-OOS score source.
8. Execution realism audit: checked. OOS execution was recomputed with t+10 entry, delay attribution, spread/slippage/fees, liquidity/source coverage, rejected rows by gate, and market/stop-order live behavior. The global accepted execution-attribution subset is still positive after costs (`3,028` trades, `72.1%` net hit, `255.36` bps mean net, mean friction about `20.28` bps). Delay cost is small on average in this replay (`-0.31` bps), but live adverse-signal gaps must continue to be logged because individual trades can be materially worse.
9. Data-source audit: checked, with residual risks. Local live-hourly versus historical-hourly parity is high (`306` sampled symbols, `3` mismatched rows, `0.98%` mismatch). Live hourly versus execution-1m aggregate mismatch is mostly explained by incomplete 1m windows; complete 60-minute execution windows had only `2` mismatches in the sampled May window. The 1m policy loader reads complete cached t+10 windows from disk and only downloads missing minute ranges when enabled.
10. Broader runtime guards: checked for the high-risk contracts now identified. Runtime/training-live validation now covers artifact hashes, feature schema/source parity, trained-universe coverage, rank-reference provenance, rank-reference universe coverage, stale feature-store coverage, and scorer/provenance mismatch. A new guard rejects rank references without a valid policy-OOS contract for active strategies.

## 2026-06-04 Incremental Update

- Fixed a reproducibility bug in `simple_policy_optimiser._filter_policy_quote_rows(...)`: copied Kraken perp policy-OOS files such as `BTC/USD:USD` were dropped when `EPM_EXCHANGE` was absent and the default perp quote resolved to `USDC`. The filter now infers a homogeneous perp quote from symbols like `BASE/USD:USD` when the configured quote matches no rows.
- Added regression coverage: `tests/test_simple_policy_optimiser_deployment.py::test_policy_quote_filter_infers_homogeneous_kraken_perp_quote`.
- Re-ran deterministic rank-reference source reconciliation for all seven policy-OOS strategies. Result: `rank_reference/*.parquet` is an exact zero-diff transform of `policy_oos_trained_universe_verify/policy_oos_*_clf.parquet`; row counts, timestamp/symbol keys, `calibrated_score`, and `rank_pct` all match exactly.
- Added `policy_source_reconciliation_manifest.json`, which records the rank-source reconciliation, current run-scoped live-ledger summary, and hash checks proving `data_perp/artifacts`, `data_perp/exchanges/krakenfutures/artifacts`, and `data_perp/exchanges/krakenfutures_perp/artifacts` point to byte-identical model/policy-OOS artifacts for the sampled files.
- Current live proof: the run-scoped ledger for `20260525_010004_nopenalty` now contains `20` clean current-run rows for signal bar `2026-06-04T08:00:00Z`, all with selected base/meta feature-value JSON and finite base/meta/rank-normalized scores. All twenty were rejected by rank threshold, so no trade was opened in those shadow cycles.
- Latest live/training-path reconciliation: after bringing `data_perp/features/20260523_015947` current through the actual training-path incremental feature generator for signal bar `2026-06-04T15:00:00Z`, the strict shadow inference batch reconciles exactly. Across the four logged decision rows (`SYN/USD:USD`, `TLM/USD:USD`, `WLD/USD:USD`, `ZRO/USD:USD`), `968` selected feature values match the training feature store with `0.0` max absolute delta; recomputed base scores, meta scores, policy rank, auction rank, and normalized rank all match the live ledger with `0.0` max absolute delta. Evidence: `live_vs_training_path_feature_values/live_ledger_reconciliation_20260604T162233Z_summary.json`.
- Live-vs-policy replay reconciliation: exact matches are `0/10` because the deployable replay artifact ends at `2026-05-21T23:00:00Z` while the live ledger rows are from `2026-06-04T08:00:00Z`. This is a timestamp-window mismatch, not a score/path failure.
- Current score-provenance conclusion: the policy rank reference is not stale. Remaining non-zero comparisons between live final-fit inference scores and policy-OOS reference scores are expected unless the same held-out policy model state and same timestamp/symbol rows are compared. The correct contract is: policy optimisation uses held-out train-meta-state policy-OOS predictions; live inference uses deployed final-fit scoring and maps those scores through the saved policy-OOS rank distribution.

## 2026-06-04 Live / Training Path Reconciliation

- A strict shadow cycle was run for `20260525_010004_nopenalty` after updating the authoritative training-path feature store `data_perp/features/20260523_015947` to the latest needed closed hour with `run_pipeline.py features` and explicit `EPM_FEATURE_END_TS=2026-06-04T15:00:00Z`.
- The earlier strict live guard failure was correct: the selected-feature store was stale because the default feature pipeline end lag targeted an older window. Once the training-path feature store was brought current, live inference loaded the strict selected-feature cache instead of recomputing model features.
- Live inference used incremental hourly OHLCV fast path for the scoped symbols (`149` skipped existing, `0` fetched), mask snapshot cache hit for the latest hour, and strict selected-feature loading from `data_perp/features/20260523_015947`.
- Latest mask support was non-empty:
  - long-dist: `6/149`
  - long high-vol: `20/149`
  - short Bollinger/OI: `3/149`
  - short asset-OI: `59/149`
- Source-parity filtering accepted `118/149` symbols and rejected `31` for missing/stale `perp_volume` inputs. This remains a data-coverage issue to monitor, but it did not prevent the four logged scored rows from reconciling.
- Candidate flow for the reconciled batch: `7` mask-passing candidates, `4` scored/logged decisions, `0` accepted trades. The four scored decisions were rejected because their rank-normalized/auction scores were below the deployed thresholds:
  - `TLM/USD:USD`: long high-vol, base `0.474695`, meta `0.548094`, normalized rank `0.416106`, threshold `0.6322`.
  - `SYN/USD:USD`: short asset-OI, base `0.507579`, meta `0.401238`, normalized rank `0.220678`, threshold `0.58`.
  - `ZRO/USD:USD`: short asset-OI, base `0.429122`, meta `0.112066`, normalized rank `0.002646`, threshold `0.58`.
  - `WLD/USD:USD`: short Bollinger/OI, base `0.456412`, meta `0.535247`, normalized rank `0.400499`, threshold `0.68`.
- Exact reconciliation summary for signal bar `2026-06-04T15:00:00Z`:
  - rows compared: `4`
  - selected feature values compared to training store: `968`
  - feature max absolute delta: `0.0`
  - base score max absolute delta: `0.0`
  - meta score max absolute delta: `0.0`
  - policy-rank max absolute delta: `0.0`
  - auction-rank max absolute delta: `0.0`
  - normalized-rank max absolute delta: `0.0`
- Evidence files:
  - `live_vs_training_path_feature_values/live_ledger_reconciliation_20260604T162233Z_summary.json`
  - `live_vs_training_path_feature_values/live_ledger_vs_training_feature_store_20260604T162233Z.csv`
  - `live_vs_training_path_feature_values/live_ledger_vs_selected_model_scores_20260604T162233Z.csv`
- Code fixes required to make this proof valid:
  - `model_orchestrator.predict_meta(...)` now exposes the exact batch meta model input matrix used for prediction.
  - `run_inference.py` now logs decision feature snapshots from that exact meta input matrix instead of reconstructing meta diagnostics later from candidate dictionaries.
  - `run_inference.py` now appends prediction-ledger rows for global-auction/rank skipped candidates when all-candidate logging is enabled, fixing missing scored rows such as the previously absent WLD decision.
- Remaining efficiency findings from this run:
  - Full-union training-path feature update is still expensive even with incremental deltas. Chunk compute was dominated by `compute_features_hourly.base_features` and `compute_features_hourly.pre_position_sizer`.
  - DuckDB delta saves still cost roughly one minute per large symbol chunk when writing hundreds of feature keys.
  - Persisted causal-transform state was present but could not be used in pure append mode when the writer re-emitted warmup rows; it fell back to full batched transform for that chunk.
- Follow-up feature-generation optimization on `2026-06-04T15:00:00Z`:
  - `save_features(...)` now slices rows by each symbol cutoff before building per-symbol payload DataFrames, avoiding thousands of warmup rows of allocation for one-row/hourly appends.
  - Chunked feature generation now avoids an extra full copy of every panel field before compute.
  - Feature timing now separates `composite_features` and `position_sizer_features`; the old `pre_position_sizer` label was misleading because it measured the prior composite block.
  - The cache precheck now applies the same naturally sparse latest-VWAP exemption as final snapshot validation. This stopped an endless one-key repair loop for `loc_vwap_dev_z_24`.
  - Cached runs with `--skip-feature-postsave-checks` now skip feature-health report regeneration as well as snapshot validation.
  - Verification: first rerun repaired `35` symbols for one key with `compute=4.6s` and `save=0.0s`; second rerun reproduced the stale sparse-key loop; after the sparse-key patch, final rerun reported `Features already exist and cover full target period: 954 features × 233 symbols. Skipping recomputation.` and exited after the `36.1s` cache scan with health reports skipped.
  - A persisted feature-cache scan manifest now records the scan result keyed by expected feature contract, required symbol/time bounds, and a cheap input-file signature. The first run writes the manifest after the normal scan; the next identical cached run hit `Feature cache scan manifest hit` and skipped per-file parquet schema/target-row reads. The cached command dropped from roughly `43s` after health-report skipping to roughly `9s` end to end, dominated by startup and close-only precheck loading.

## Executive Summary

- Root cause: current evidence points to a policy/inference universe-contract mismatch, not model failure. The policy rank-reference was not stale: it was generated row-for-row from `policy_oos_predictions/*_clf.parquet`. The bug was that policy-OOS generation allowed a broader 179-237-symbol policy universe while the trained/inference OOF union contains only 152 symbols. Each old per-strategy policy source had 74-87 symbols outside the trained universe, and those rows could not be scored by strict trained-universe replay. Policy-OOS generation now filters to the deployable trained universe and `simple_policy_optimiser` now fails closed if policy rows contain symbols outside that universe.
- Current confidence: high for current policy-OOS artifact provenance, threshold-band economics, and replay/rank contract instrumentation; medium for the broader live degradation question until fresh run-scoped live decisions, execution realism, and data-source audits are completed.
- Current trained-universe verification: regenerated policy-OOS files under `policy_oos_trained_universe_verify/` now have zero symbols outside the trained universe. A guarded full-budget optimiser run from those rows exported `27,485` cross-strategy candidates across seven strategies, with `100%` cached t+10 path coverage and no Kraken 1m download attempts (`EPM_SIMPLE_POLICY_1M_DOWNLOAD=0`).
- Current deployability blocker: the guarded full-budget deployment payload selected zero strategies because all seven strategies were rejected for `missing_lgbm_mask_contract`. The portfolio replay still ran over all candidate rows and accepted `3,236` trades with objective `10024.095596`; this is now classified as invalid deployable evidence because it was not constrained to a non-empty live-deployable strategy set. Code now fails closed before portfolio replay when deployment selection is empty.
- Threshold-band result after trained-universe filtering: threshold-level rows are only weakly positive and often below the 60% net-hit target. Per-strategy threshold hit rates are `39.33%` to `56.98%`, while final selected policy rows are positive EV across all seven strategies. Threshold-band gross return minus the 20 bps stress buffer is negative at every deployment threshold. The current policy result therefore remains partly dependent on higher-rank/constrained selection, not only marginal threshold acceptance.
- Strict post-policy frozen holdout result: using the saved six-head thresholds, saved policy-rank references, train-meta-frozen model state, saved portfolio policy config, fixed t+10 Kraken 1m delayed-entry candles, and a +1/+2/+3 minute fallback on `2026-05-22T00:00:00Z` through `2026-05-27T17:00:00Z` produced `729` auction-floor candidates and `73` accepted replay trades with `2.19%` mean accepted net return, `11.70` trades/day, final wallet `11583.00`, and max drawdown `-0.13%`. This is a short untouched policy-layer holdout with strong portfolio result, but not yet a complete live execution proof.
- Current policy-source localization: `simple_policy_optimiser/rank_reference/*.parquet` is generated from `policy_oos_predictions/*_clf.parquet`. Joining strategy references to policy-OOS rows on `timestamp` and `symbol` gives exact row equality for all seven deployed policy-OOS strategies; every `rank_reference.calibrated_score` equals `policy_oos.clf` exactly, and `rank_pct` matches the percentile transform within float precision.
- Current deployment blocker status: the earlier missing-LGBM-mask blocker is fixed for the newly written deployment contract. The corrected optimiser run reports `lgbm_mask_contract_covered=4` for the four selected strategies, all with `regime_mask_source=embedded_lgbm_final_rule_registry`. The remaining live proof is a fresh run-scoped cycle and decision reconciliation, not another mask-contract repair.
- Current deployed-feature parity proof: strict historical replay through the actual inference candidate path now has exact feature parity for all four deployed strategies selected by the current policy package. Across `21,500` compared feature cells, inference and training-path selected features have `0` missing rows, `0` mismatches above `1e-6`, and `0.0` max absolute difference. The replay also produces finite predictions for all sampled decision rows.
- Current score-provenance interpretation: the remaining non-zero policy-score and rank-normalized-score deltas are expected for this comparison because the saved policy rank-reference is generated from the frozen train-meta policy-OOS source, while strict replay scores the deployed final-fit inference bundle. That is an OOS-contract distinction, not a feature-generation parity failure.
- Guard status: `simple_policy_optimiser` now requires policy-OOS manifests to prove train-meta-frozen provenance, a non-final-fit source, the expected candidate/execution source, rank-normalization metadata, source-model hash, and a passing artifact preflight. It also validates that every loaded policy-OOS row is inside the trained/inference universe before optimisation. The rank-reference loader now fails closed when a regenerated manifest declares an invalid explicit policy-OOS contract. Strict training-live artifact validation now covers `base_models_intermediate.pkl`, `trained_state.pkl`, `model_state_meta.pkl`, the native model directory tree, the rank-reference manifest, and the cross-strategy auction reference. It also fails closed when readable policy rank-reference symbols are outside the trained/inference universe. The current deployed parity contract correctly fails offline because it lacks newer hash entries; regenerated contracts must also pass the rank-reference universe guard.
- Live-run status: a bounded `--live-test --run-once --perps` direct module run reached Kraken exchange creation but failed inside the sandbox at DNS resolution for `futures.kraken.com`; escalated retries timed out in approval review. Separately, the LaunchAgent log repeatedly reports `can't open input file: /Users/remyroche/Documents/Ares/scripts/run_live_test_supervised.sh` even though that path exists in the workspace, so the launchd plist/runtime environment, permissions, or stale launch state should be checked before relying on the monitor.
- Confirmed so far: the repo already has policy-rank reference logic, feature-universe parity tests, prediction ledger diagnostics, and historical/live replay scripts. The investigation must use those concrete paths rather than a generic raw-probability/top-k framing.
- Confirmed historical finding: the earlier four-head policy rank-reference was not stale and was not copied from `20260523_015947`; it reproduced to float precision when using the exchange-scoped offline selected-feature handoff.
- Important terminology correction: runtime columns still use legacy score-transform names, but the deployment decision is rank-normalized through saved policy-rank reference distributions.
- Important execution correction: new investigation reports should focus on market/marketable entry behavior, protective stop behavior, spread, slippage, fees, rejected orders, stale data, and delay.

## Commands Run

- Repository inspection:
  - `find extreme_price_movements -maxdepth 3 -type f | sort`
  - `rg -n "rank...|simple_policy|prediction_ledger|execution|slippage|spread" ...`
  - focused reads of `run_pipeline.py`, `simple_policy_optimiser.py`, `inference/run_inference.py`, `inference/policy_rank_reference.py`, `inference/model_orchestrator.py`, and existing parity tests.
- Skill validation:
  - `python3 /Users/remyroche/.codex/skills/.system/skill-creator/scripts/quick_validate.py /Users/remyroche/.codex/skills/extreme-price-movements-inference-mismatch`
- OOS reproduction: pending.
- Inference reproduction: pending.
- Historical replay: in progress; all four active strategy heads have policy-rank sample parity against saved final-fit rank references when the replay uses the same feature handoff, including through the actual inference candidate/mask feature-load path.
- Deployed four-head strict feature parity rerun:
  - `scripts/historical_inference_parity.py --feature-load-path inference_candidate --policy-artifact-root extreme_price_movements/reports/inference_mismatch_investigation/simple_policy_optimiser_trained_universe_full_deployable --rank-reference-dir extreme_price_movements/reports/inference_mismatch_investigation/simple_policy_optimiser_trained_universe_full_deployable/simple_policy_optimiser/rank_reference ...`
  - output directories: `historical_parity_trained_universe_deployable_refs_long_dist_features`, `historical_parity_trained_universe_deployable_refs_long_bars`, `historical_parity_trained_universe_deployable_refs_short_bollinger`, and `historical_parity_trained_universe_deployable_refs_short_oi`.
- Rank-reference provenance:
  - current/source rank-reference parquet comparison;
  - current `meta_oof` timestamp-range check;
  - log inspection of `logs/unified_20260528_121034.log`;
  - direct final-fit model reproduction of three saved long-dist rank-reference rows through the policy feature handoff.
- Rank-reference universe audit:
  - `python3 ... policy_rank_reference_universe_audit.csv` audit under `extreme_price_movements/reports/inference_mismatch_investigation/`.
- Trained-universe policy-OOS regeneration:
  - `env PYTHONUNBUFFERED=1 PYTHONPATH=. EPM_DATA_ROOT=data_perp EPM_EXCHANGE=krakenfutures EPM_MARKET_MODE=perps EPM_MODEL_BACKEND=lgbm_pipeline EPM_DISABLE_REGIME_ADAPTORS=1 EPM_SIMPLE_POLICY_REGIME_ADAPTOR=0 python3 scripts/generate_policy_oos_predictions.py --data-root data_perp --run-id 20260525_010004_nopenalty --market-mode perps --output-dir extreme_price_movements/reports/inference_mismatch_investigation/policy_oos_trained_universe_verify`
  - audit output: `extreme_price_movements/reports/inference_mismatch_investigation/policy_oos_trained_universe_verify_audit.csv`.
- Guarded diagnostic optimiser run:
  - `env PYTHONUNBUFFERED=1 PYTHONPATH=. EPM_DATA_ROOT=data_perp EPM_EXCHANGE=krakenfutures EPM_MARKET_MODE=perps EPM_MODEL_BACKEND=lgbm_pipeline EPM_DISABLE_REGIME_ADAPTORS=1 EPM_SIMPLE_POLICY_REGIME_ADAPTOR=0 EPM_SIMPLE_POLICY_OOS_PREDICTIONS_DIR=extreme_price_movements/reports/inference_mismatch_investigation/policy_oos_trained_universe_verify EPM_SIMPLE_POLICY_OUTPUT_RUN_ROOT=extreme_price_movements/reports/inference_mismatch_investigation/simple_policy_optimiser_trained_universe_verify EPM_SIMPLE_POLICY_RUN_PORTFOLIO_REPLAY=0 SIMPLE_POLICY_N_TRIALS=20 python3 -u extreme_price_movements/simple_policy_optimiser.py --data_root data_perp --run_id 20260525_010004_nopenalty --market-mode perps`
  - metrics output: `extreme_price_movements/reports/inference_mismatch_investigation/simple_policy_optimiser_trained_universe_verify_metrics.csv`.
- Test command:
  - `python3 -m pytest tests/test_policy_rank_reference.py tests/test_live_feature_universe_parity.py tests/test_replay_live_signal_predictions.py -q`
  - `python3 -m pytest tests/test_replay_live_signal_predictions.py tests/test_live_feature_universe_parity.py -q`
  - `python3 -m pytest tests/test_live_feature_universe_parity.py -q`
  - `python3 -m py_compile scripts/historical_inference_parity.py`
  - `python3 -m py_compile extreme_price_movements/data_store.py extreme_price_movements/inference/feature_generator.py scripts/historical_inference_parity.py`
  - `python3 -m py_compile scripts/evaluate_frozen_policy_holdout.py`
  - `git diff --check -- scripts/historical_inference_parity.py tests/test_replay_live_signal_predictions.py`
  - `EPM_EXCHANGE=kraken python3 scripts/historical_inference_parity.py --data-root data_perp --artifact-data-root data_perp --run-id 20260525_010004_nopenalty --market-mode perps --sample-source policy_rank_reference --strategy-id <long_dist_head> --sample-rows 3 --lookback-hours 720 --basket-mode sample --output-dir extreme_price_movements/reports/inference_mismatch_investigation/historical_parity_smoke`
  - `EPM_EXCHANGE=kraken python3 scripts/historical_inference_parity.py --data-root data_perp --artifact-data-root data_perp --run-id 20260525_010004_nopenalty --market-mode perps --sample-source policy_rank_reference --strategy-id <long_dist_head> --sample-rows 3 --lookback-hours 720 --basket-mode full --output-dir extreme_price_movements/reports/inference_mismatch_investigation/historical_parity_single_strategy_full`
  - `EPM_EXCHANGE=kraken python3 scripts/historical_inference_parity.py --data-root data_perp --artifact-data-root data_perp --run-id 20260525_010004_nopenalty --market-mode perps --sample-source policy_rank_reference --strategy-id <long_dist_head> --sample-rows 3 --lookback-hours 720 --basket-mode full --skip-feature-comparison --output-dir extreme_price_movements/reports/inference_mismatch_investigation/historical_parity_single_strategy_cache_fix`
  - same command with output dir `historical_parity_single_strategy_cache_hit`
- Frozen policy holdout:
  - `EPM_DATA_ROOT=data_perp EPM_ARTIFACT_SOURCE_RUN_ID=20260523_015947 EPM_EXCHANGE=krakenfutures EPM_MARKET_MODE=perps python3 -u scripts/evaluate_frozen_policy_holdout.py --data-root data_perp --run-id 20260525_010004_nopenalty --market-mode perps --artifact-source-run-id 20260523_015947 --predict-start 2026-05-22T00:00:00Z --predict-end 2026-05-27T17:00:00Z --output-dir data_perp/artifacts/20260525_010004_nopenalty/policy_holdout_frozen_replay_network`
  - Direct Kraken 1m probe for `CYBER/USD:USD`, `CTSI/USD:USD`, and `LQTY/USD:USD` over `2026-05-22T00:00:00Z` to `2026-05-22T00:15:00Z`, before and after preserving explicit zero-volume carry candles.
  - `EPM_DATA_ROOT=data_perp EPM_ARTIFACT_SOURCE_RUN_ID=20260523_015947 EPM_EXCHANGE=krakenfutures EPM_MARKET_MODE=perps python3 -u scripts/evaluate_frozen_policy_holdout.py --data-root data_perp --run-id 20260525_010004_nopenalty --market-mode perps --artifact-source-run-id 20260523_015947 --predict-start 2026-05-22T00:00:00Z --predict-end 2026-05-27T17:00:00Z --output-dir data_perp/artifacts/20260525_010004_nopenalty/policy_holdout_frozen_replay_1m_fallback`
  - `python3 -m pytest tests/test_simple_policy_optimiser_deployment.py::test_delayed_entry_uses_nearby_1m_fallback tests/test_kraken_charts_ohlcv.py -q`
- Live-vs-replay decision reconciliation:
  - `python3 scripts/reconcile_live_replay_decisions.py --output-dir extreme_price_movements/reports/inference_mismatch_investigation --report-name live_vs_frozen_holdout_decision_reconciliation.md`
  - `python3 scripts/reconcile_live_replay_decisions.py --replay-decisions data_perp/artifacts/20260525_010004_nopenalty/portfolio_policy_replay/per_candidate_replay_decisions.parquet --candidates data_perp/artifacts/20260525_010004_nopenalty/simple_policy_optimiser/simple_policy_candidates.parquet --output-dir extreme_price_movements/reports/inference_mismatch_investigation/main_replay_live_reconciliation --report-name live_vs_main_replay_decision_reconciliation.md`
- Run-scoped shadow live check:
  - `EPM_RUN_SCOPED_PREDICTION_LEDGER=1 python3 -u -m extreme_price_movements.inference.run_inference --shadow --run-once --perps --data-root data_perp --run-id 20260525_010004_nopenalty`
  - `python3 -m pytest tests/test_simple_policy_optimiser_deployment.py::test_deployment_payload_requires_current_trained_meta_model tests/test_simple_policy_optimiser_deployment.py::test_deployment_payload_rejects_missing_lgbm_masks_for_lgbm_backend tests/test_simple_policy_optimiser_deployment.py::test_deployment_payload_embeds_market_specific_lgbm_mask_contract tests/test_simple_policy_optimiser_deployment.py::test_deployment_payload_persists_realized_holding_time_metrics extreme_price_movements/tests/test_inference_step_parity.py::test_load_lgbm_strategy_masks_prefers_embedded_strategy_contract extreme_price_movements/tests/test_inference_step_parity.py::test_load_lgbm_strategy_masks_fallback_filters_to_selected_strategies extreme_price_movements/tests/test_inference_step_parity.py::test_lgbm_strategy_mask_coverage_fails_closed_for_missing_selected_strategy -q`
  - `python3 -m py_compile scripts/historical_inference_parity.py extreme_price_movements/simple_policy_optimiser.py`
  - `python3 -m py_compile extreme_price_movements/inference/policy_rank_reference.py`
  - `python3 -m pytest tests/test_historical_inference_parity_feature_scope.py tests/test_simple_policy_optimiser_deployment.py::test_policy_oos_contract_requires_predictions_after_train_end tests/test_simple_policy_optimiser_deployment.py::test_policy_oos_contract_requires_manifest_provenance tests/test_simple_policy_optimiser_deployment.py::test_policy_oos_contract_requires_scoring_contract -q`
  - `python3 -m pytest tests/test_policy_rank_reference.py::test_persist_policy_rank_reference_manifest tests/test_policy_rank_reference.py::test_policy_rank_reference_bad_policy_oos_contract_fails_closed tests/test_historical_inference_parity_feature_scope.py tests/test_simple_policy_optimiser_deployment.py::test_policy_oos_contract_requires_predictions_after_train_end tests/test_simple_policy_optimiser_deployment.py::test_policy_oos_contract_requires_manifest_provenance tests/test_simple_policy_optimiser_deployment.py::test_policy_oos_contract_requires_scoring_contract -q`

## Artifact Evidence

- Artifact manifest: `artifact_manifest.json`.
- Model artifact run id: `20260525_010004_nopenalty`.
- Source feature run id used by policy handoff: `20260523_015947`.
- Current artifact manifest scope: six-strategy policy-OOS and portfolio replay artifacts as of 2026-06-02; `24` files with hashes, sizes, mtimes, row counts, timestamp ranges, and strategy samples.
- Policy-OOS preflight: valid. Source model fit end `2024-09-08T04:00:00+00:00`; policy fit end `2026-01-19T06:00:00+00:00`; policy prediction window starts `2026-01-22T10:00:00+00:00`.
- Policy-OOS handoff: seven `policy_oos_<strategy>_clf.parquet` files, generated by `scripts/generate_policy_oos_predictions.py`, `generated_from_final_fit_bundle=false`, `model_provenance=train_meta_frozen_model_state`.
- Trained-universe policy-OOS repair evidence: `policy_oos_trained_universe_verify/manifest.json` reports `trained_universe_symbols=152`. The regenerated files dropped `74-87` non-trained symbols per strategy and the follow-up audit reports `outside_trained_universe_symbols=0` for all seven policy-OOS files.
- Guarded diagnostic optimiser evidence: `simple_policy_optimiser_trained_universe_verify/` was written through the isolated `EPM_SIMPLE_POLICY_OUTPUT_RUN_ROOT` path so it did not overwrite active deployment artifacts. It produced `27,485` cross-strategy candidate rows after the normalized-rank floor. The run used `SIMPLE_POLICY_N_TRIALS=20` and `EPM_SIMPLE_POLICY_RUN_PORTFOLIO_REPLAY=0`, so it validates the contract and gives directional threshold economics but is not the final policy optimisation.
- Diagnostic threshold economics by strategy:
  - long asset-vol/compression: `358` threshold trades, `56.98%` hit, `1.33` bps mean net; final selected rows `62.93%` hit, `31.60` bps mean net.
  - long high-vol pullback/funding: `489` threshold trades, `53.17%` hit, `1.10` bps mean net; final selected rows `51.96%` hit, `22.04` bps mean net.
  - long high-vol location/range: `312` threshold trades, `56.09%` hit, `1.92` bps mean net; final selected rows `54.71%` hit, `27.16` bps mean net.
  - long-dist: `410` threshold trades, `50.98%` hit, `0.36` bps mean net; final selected rows `48.80%` hit, `21.50` bps mean net.
  - long-loc: `414` threshold trades, `44.69%` hit, `0.65` bps mean net; final selected rows `46.53%` hit, `25.47` bps mean net.
  - short asset-OI: `295` threshold trades, `52.54%` hit, `1.20` bps mean net; final selected rows `59.22%` hit, `29.81` bps mean net.
  - short Bollinger/price-RV: `239` threshold trades, `39.33%` hit, `1.49` bps mean net; final selected rows `69.26%` hit, `20.61` bps mean net.
- Guarded full-budget optimiser evidence: `simple_policy_optimiser_trained_universe_full/` was run from the same trained-universe policy-OOS source with `EPM_SIMPLE_POLICY_1M_DOWNLOAD=0` and portfolio replay enabled. Candidate export again produced `27,485` cross-strategy rows. Per-strategy t+10 path matrices reported `100.00%` coverage, and the rank-reference universe audit reports zero outside-trained symbols for all seven per-strategy references and `cross_strategy_auction.parquet`.
- Full-budget portfolio replay diagnostic: the run completed with `accepted=true`, `3,236` accepted trades, objective `10024.095596`, baseline trade count `2,685`, and baseline objective `2354.607861`, but the deployment payload selected zero strategies because all seven were rejected for `missing_lgbm_mask_contract`. This replay is therefore not valid deployable evidence. Because this run also exposed that portfolio replay did not respect `EPM_SIMPLE_POLICY_OUTPUT_RUN_ROOT`, the replay report was written under `data_perp/artifacts/20260525_010004_nopenalty/portfolio_policy_replay/portfolio_policy_replay_report.json`; the isolation bug is now patched and covered by a regression test.
- Full-budget per-strategy threshold economics are saved at `simple_policy_optimiser_trained_universe_full_metrics.csv`. Deployment threshold bands are weakly positive on mean net trade but remain below the 60% local hit-rate target for all seven strategies. Their gross return minus the configured 20 bps stress buffer is negative at the threshold band for all seven strategies. Final selected rows are positive after that buffer, with final mean net trade from `18.94` to `33.32` bps and final hit rates from `47.28%` to `67.95%`.
- Policy-OOS scoring contract: per-file manifests are now required to declare `prediction_source=generated_from_train_meta_state:*`, `candidate_rows_source=policy_slice_feature_events`, `executable_path_source=simple_policy_optimiser_recomputes_from_ohlcv_and_execution_1m`, a rank-normalization declaration, a source model-state hash, and a passing source-artifact preflight.
- Deployed four-head strict replay feature parity:
  - long-dist: `5,680` feature rows, `0` missing inference rows, `0` missing training rows, `0` mismatches above `1e-6`, max abs diff `0.0`, `20` prediction rows.
  - long-bars: `5,720` feature rows, `0` missing inference rows, `0` missing training rows, `0` mismatches above `1e-6`, max abs diff `0.0`, `20` prediction rows.
  - short Bollinger/price-RV: `6,060` feature rows, `0` missing inference rows, `0` missing training rows, `0` mismatches above `1e-6`, max abs diff `0.0`, `20` prediction rows.
  - short asset-OI: `4,040` feature rows, `0` missing inference rows, `0` missing training rows, `0` mismatches above `1e-6`, max abs diff `0.0`, `20` prediction rows.
  - Legacy summary fields named `policy_calibrated_score_max_abs_diff` are policy-score diffs; legacy `policy_rank_pct_max_abs_diff` is the rank-normalized-score diff. These are non-zero because the comparison is final-fit deployed scoring versus frozen train-meta policy-OOS rank-reference rows.
- Active-artifact simple-policy candidates from the earlier unfiltered run: `48,913` rows across seven optimized strategies, generated with configured delayed entry `10` minutes and rank-band audit artifacts under `simple_policy_optimiser/rank_threshold_band_report.*`. These artifacts are now known to predate the trained-universe policy-OOS filter and should not be treated as deployable until regenerated from the guarded source.
- Earlier delayed-entry coverage: `45,632/48,913` candidate rows (`93.29%`) used complete cached 1m t+10 execution windows; `3,281` rows fell back to `theoretical_15m_open`. This remains useful execution-data coverage evidence, but the candidate set itself must be regenerated from trained-universe policy-OOS rows before final deployment.
- Frozen replay support added: `portfolio_policy_replay` can now load a saved `optimized_portfolio_policy_config.json` and fit its priority EV curve from a separate reference candidate table, enabling post-policy holdout replay without re-optimising on the holdout.
- Frozen post-policy holdout evidence:
  - Summary: `data_perp/artifacts/20260525_010004_nopenalty/policy_holdout_frozen_replay_1m_fallback/summary.json`.
  - Candidate table: `data_perp/artifacts/20260525_010004_nopenalty/policy_holdout_frozen_replay_1m_fallback/simple_policy_holdout_candidates.parquet`.
  - Candidate metadata: `data_perp/artifacts/20260525_010004_nopenalty/policy_holdout_frozen_replay_1m_fallback/simple_policy_holdout_candidates_metadata.json`.
  - Portfolio replay: `data_perp/artifacts/20260525_010004_nopenalty/policy_holdout_frozen_replay_1m_fallback/portfolio_policy_replay/portfolio_policy_replay_report.json`.
  - Prediction window: `2026-05-22T00:00:00+00:00` to `2026-05-27T17:00:00+00:00`, after policy optimiser prediction end `2026-05-22T00:00:00+00:00`.
  - Source validation: `policy_holdout_temporal_disjoint=true`, `policy_holdout_fit_predict_disjoint=true`, `policy_holdout_train_base_meta_fit_disjoint=true`, and `policy_holdout_train_base_meta_fit_overlap_rows=0`.
- Exchange-scoped feature handoff: `data_perp/exchanges/krakenfutures/features/20260525_010004_nopenalty -> data_perp/features/20260523_015947`.
- Policy rank-reference row counts and hashes: recorded in `artifact_manifest.json`.
- Current `meta_oof` files end at `2026-01-19 03:00:00+00:00`, before the policy rank-reference starts. The policy slice was generated from final full-fit inference models, not current `meta_oof`.
- Standalone artifact manifest now records hashes, sizes, mtimes, row counts, timestamp ranges, and schema samples for trained state, meta state, exchange/training contracts, base/meta OOF handoffs, per-strategy rank references, cross-strategy auction reference, policy params, and simple-policy candidates.
- The parity contract points to `data_perp/exchanges/krakenfutures/artifacts/20260525_010004_nopenalty`; locally this is a symlink to `data_perp/artifacts/20260525_010004_nopenalty`, so the exchange-scoped and root artifact paths resolve to the same package.
- Rank-reference manifest: generated by `simple_policy_optimiser`, schema `policy_rank_reference_v1`, seven strategies, per-strategy score column `calibrated_score`, per-strategy rank column `rank_pct`, cross-strategy rank column `normalized_rank_score`. Each per-strategy manifest entry now carries a `policy_oos_contract` declaring train-meta-state generation, source fit end, and rank-normalization. A top-level aggregate policy-OOS contract is still absent and should be added as a cleanup guard.
- Policy-OOS source-artifact preflight now writes `data_perp/artifacts/20260525_010004_nopenalty/policy_oos_predictions/preflight_report.json` and currently passes because both required sidecars are present:
  - `data_perp/artifacts/20260525_010004_nopenalty/base_models_intermediate.manifest.json`
  - `data_perp/artifacts/20260525_010004_nopenalty/models/model_state_meta.manifest.json`
- Runtime scorer artifact risk: `load_inference_config(...)` loads `config["model_bundle"]` through `load_model_bundle(...)`, which uses `models/native` for alpha models; `_effective_runtime_model_bundle(...)` then overlays that runtime bundle onto `load_full_state(...)`. Policy-OOS generation instead declares `base_models_intermediate.pkl` as the base source. The live scorer contract therefore requires either a score-equivalence proof between native alpha models and base-intermediate alpha models or an explicit runtime switch to the policy-OOS-safe base source.
- Deployment mask contract evidence:
  - `data_perp/artifacts/20260525_010004_nopenalty/policy_params/strategy_for_inference.json` selects six strategies, but each selected row has `lgbm_regime_mask={}` and `regime_mask_source=missing_lgbm_mask_contract`.
  - `extreme_price_movements/offline_optimisers/reports/inference_candidate_mask_best_params_perps.csv` contains four old rows only, with strategy ids beginning `loc_ema_stack_pos_24_...`, `dist_rolling_7d_high_...`, `loc_prev_week_range_pos_48_...`, and `dist_weekly_vwap_...`; none match the six selected deployment ids.
  - A CSV scan under `data_perp/artifacts/20260523_015947` found no exact selected six strategy ids in `canonical_key`, `strategy_id`, or `base_event_trigger` columns. Exact hits under `20260525_010004_nopenalty` were downstream policy reports only, not mask registries.
  - `data_perp/artifacts/20260523_015947/policy_oos_retrain_strategy_source_perps.csv` lists seven retrain source rows and calls these ids `canonical_key`, but those values are not parseable canonical rule expressions; direct `split_composite_key(...)` and `extract_feature_names_from_key(...)` return no rule structure/features for sampled selected ids.

## Pipeline Map

See `pipeline_map.md`.

## First Divergence

- Divergence stage: live-style recomputed features versus offline selected-feature policy handoff.
- First affected timestamp: 2026-01-28 12:00:00 UTC sample batch.
- First affected symbols: AAVE/USD:USD, ANIME/USD:USD, APT/USD:USD in the sampled long-dist policy-rank rows.
- OOS/policy-reference value: saved policy rank-reference scores from `simple_policy_optimiser/rank_reference`.
- Inference replay value: final-fit base and meta predictions now score successfully.
- Explanation: the previous missing `ret4h_bench_resid` blocker was caused by the sample replay basket excluding the BTC benchmark required by benchmark-residual features. The following meta blocker was caused by replay not carrying enough warmup and by market-wide regime score columns being represented as Series instead of per-symbol DataFrames. Those are fixed. A later apparent large mismatch was caused by replay recomputing live-style features while the saved rank reference used the offline selected-feature handoff. When the replay is pointed to the same source feature handoff, all four heads reproduce saved policy scores to float precision.
- Evidence file/test: `historical_parity_*_source_features_no_rolling/summary.json`, `rank_reference_provenance.md`, `replay_diff_report.md`, `artifact_manifest.json`, and command output.

## Current Findings

### Feature Parity

- Existing test coverage verifies that live model features are computed on the full tradable universe, not only post-mask candidates.
- Existing tests prevent stale orderbook features from being forward-filled into the live snapshot.
- Confirmed high-risk area: cross-sectional and market-wide features whose values depend on the live universe and cached raw inputs. The saved policy reference used offline selected-feature parquet rows, while the live replay recomputes features from OHLCV/live-style source caches.
- Exchange-scoped policy feature handoff: `data_perp/exchanges/krakenfutures/features/20260525_010004_nopenalty -> ../../../features/20260523_015947`.
- Tiny historical replay on one long-dist head found zero mismatches above `1e-6` among comparable cells after adding required BTC benchmark context. This remains a smoke result because most training reference cells are unavailable for the selected policy-rank rows.
- Full-warmup full-basket replay for the same long-dist head loaded 236 trained symbols and produced base plus meta predictions for the three sampled rows.
- Replay feature generation now uses `effective_lookback_hours=7272` for panel loading, so long-window benchmark residuals and transformed features have the same warmup budget the script previously only logged.
- Market-wide regime features such as `regime_trend_score` and `regime_vol_score` are now broadcast to the live/replay symbol universe before model scoring instead of being lost as Series-shaped placeholders.

### Top-K / Ranking

- The live gate uses saved policy-rank reference percentiles. This is covered by `tests/test_policy_rank_reference.py`.
- The current code still uses a legacy transformed-score column name. The investigation will refer to this as the score used for rank-normalization unless an active score-transform artifact is proven to materially alter ordering.
- The cross-strategy auction rank reference is persisted by `persist_auction_rank_reference(...)` and should be included in replay parity.
- The first replay attempt without `EPM_EXCHANGE=kraken` silently targeted the Binance USD-M data component and loaded no panel. Replay commands must set the exchange explicitly.
- Current long-dist rank-reference is not a stale source-run copy: current sha16 `0a8636c05192e906`, source-run sha16 `ca67ab8f11efd35f`, joined score mean absolute difference `0.14120904469502452`, exact score matches `0`.
- Direct final-fit reproduction through the policy feature handoff matched the first three saved long-dist rank-reference scores with max absolute difference `2.6175391298899342e-08`.
- 2026-06-03 correction: that earlier direct reproduction did not prove the full current policy/inference contract. A stricter replay using the current trained/inference universe found that sampled long-dist rank-reference rows such as `CAKE/USD:USD`, `INIT/USD:USD`, `SUN/USD:USD`, `RSR/USD:USD`, and `SPK/USD:USD` are outside the 152-symbol trained OOF union, so strict replay could not score them and reported `missing_feature_row`.
- Current rank-reference universe audit is saved at `policy_rank_reference_universe_audit.csv`. It shows `cross_strategy_auction.parquet` has `237` symbols with `88` outside the trained universe. Per-strategy rank-reference files have `179-236` symbols and `74-87` outside the trained universe. The source policy-OOS predictions already contain the same class of mismatch: `policy_oos_universe_audit.csv` shows `179-236` symbols per source file and `74-87` outside the trained universe. The repair point was therefore policy-OOS generation/source filtering, not only rank-reference persistence.
- 2026-06-03 repair: `scripts/generate_policy_oos_predictions.py` now filters policy-OOS rows to `load_trained_symbol_universe(...)`, records the trained-universe filter in manifests, and aborts if a strategy has no deployable rows. `simple_policy_optimiser.py` now validates policy-OOS rows against the same trained universe before optimisation. The regenerated trained-universe policy-OOS audit reports zero outside-universe symbols for all seven strategies.
- All-four-head policy-rank replay through `scripts/historical_inference_parity.py` now matches saved rank-normalized references:
  - long-dist: score max abs diff `1.1398097299331056e-08`, rank-pct max abs diff `3.936542928001385e-05`.
  - long-loc: score max abs diff `1.2172013008626692e-08`, rank-pct max abs diff `4.864759680869857e-05`.
  - short-dist: score max abs diff `7.229255527541056e-09`, rank-pct max abs diff `3.943217665616783e-05`.
  - short-loc: score max abs diff `1.180954556367908e-08`, rank-pct max abs diff `3.471498993268263e-05`.
- The same four-head parity now also holds when replay calls the actual `_select_candidates_and_load_features(...)` inference path rather than the direct feature loader:
  - long-dist strict path: mask pass `12/236`, source eligible `165/236`, policy score max abs diff `1.1398097299331056e-08`.
  - long-loc strict path: mask pass `4/236`, source eligible `167/236`, policy score max abs diff `1.2172013008626692e-08`.
  - short-dist strict path: mask pass `19/236`, source eligible `156/236`, one additional candidate rejected for missing model feature coverage (`HMSTR/USD:USD` missing `unwind_score`), policy score max abs diff `7.229255527541056e-09`.
  - short-loc strict path: mask pass `115/236`, source eligible `164/236`, eight additional candidates rejected for missing model feature coverage, policy score max abs diff `1.180954556367908e-08`.
- After the cache split, the same strict inference candidate path was rerun with predictions enabled for 14 policy-rank sample rows per head:
  - long-dist: candidate side/count `long=12`, policy score max abs diff `2.6175391298899342e-08`, rank max abs diff `3.936542928001385e-05`.
  - long-loc: candidate side/count `long=15`, policy score max abs diff `1.2620761713488804e-08`, rank max abs diff `4.8647596808781834e-05`.
  - short-dist: candidate side/count `short=9`, policy score max abs diff `1.3113076846593685e-08`, rank max abs diff `3.943217665616783e-05`.
  - short-loc: candidate side/count `short=84`, policy score max abs diff `1.5324575497466242e-08`, rank max abs diff `3.471498993268263e-05`.
  - This confirms that the cache split did not alter score/rank parity.
- 2026-06-04 deployed four-head parity rerun against the current four-strategy deployment package proves exact selected-feature parity through the live candidate path:
  - long-dist: `feature_rows=5680`, missing inference/training `0/0`, mismatches above `1e-6=0`, max abs diff `0.0`, prediction rows `20`.
  - long-bars: `feature_rows=5720`, missing inference/training `0/0`, mismatches above `1e-6=0`, max abs diff `0.0`, prediction rows `20`.
  - short Bollinger/price-RV: `feature_rows=6060`, missing inference/training `0/0`, mismatches above `1e-6=0`, max abs diff `0.0`, prediction rows `20`.
  - short asset-OI: `feature_rows=4040`, missing inference/training `0/0`, mismatches above `1e-6=0`, max abs diff `0.0`, prediction rows `20`.
- The final long-bars mismatch was a feature-store issue, not model drift: stale DuckDB delta rows for `dist_ema50_atr` at duplicate timestamps overrode populated base Parquet values during selected-feature reads. The store now gives existing populated base cells precedence over stale deltas and prevents incoming deltas from overwriting populated historical cells.
- The replay also exposed that the inference policy loaders were still resolving thresholds/selection/mask cores from the canonical artifact path instead of the isolated policy-artifact root supplied to the replay. The loaders now search the same policy artifact bases used by parity tooling.
- Strict-path evidence directories:
  - `historical_parity_long_dist_inference_candidate_path`
  - `historical_parity_long_loc_inference_candidate_path`
  - `historical_parity_short_dist_inference_candidate_path`
  - `historical_parity_short_loc_inference_candidate_path`
  - `historical_parity_long_dist_inference_candidate_after_cache_split`
  - `historical_parity_long_loc_inference_candidate_after_cache_split`
  - `historical_parity_short_dist_inference_candidate_after_cache_split`
  - `historical_parity_short_loc_inference_candidate_after_cache_split`

### Ranking / Threshold Reconciliation

See `topk_reconciliation_report.md`.

- Per-strategy deployment thresholds currently used by the six-head policy params:
  - long asset-vol/compression: `0.68`.
  - long high-vol pullback/funding: `0.65`.
  - long high-vol location/range: `0.63`.
  - long local BB/channel: `0.64`.
  - short asset-OI: `0.60`.
  - short Bollinger/price-RV: `0.68`.
- The local-candidate net-hit guard is enabled at `min_net_hit_rate=0.50`, `min_rows=50`, rank column `auction_rank_score`.
- `rank_threshold_band_report.csv` from the earlier active-artifact run proves marginal threshold economics separately from cumulative higher-rank economics. The old 0.80-0.85 global auction band was positive on mean and median net return, but hit rate was only `57.41%`, so the trade-off was EV-positive but not uniformly above the 60% hit-rate target. The new trained-universe diagnostic confirms the same concern more strongly at per-strategy threshold boundaries.
- Remaining ranking work is final portfolio decision reconciliation: rank parity is proven, but final live trade parity also depends on source-panel eligibility, sparse-feature rejection, per-symbol/per-strategy concurrency, global auction, portfolio capacity, and stale/adverse price gap checks.

### Portfolio Decision Reconciliation

See `portfolio_decision_reconciliation.md` and `live_vs_frozen_holdout_decision_reconciliation.md`.

- Added `scripts/reconcile_live_replay_decisions.py` to compare live prediction-ledger rows against replay decision artifacts with both exact keys (`signal_bar_ts/timestamp`, symbol, side, strategy) and loose symbol/side timestamp keys.
- The current live ledger is not a valid current-package parity sample. It contains `149` rows across three artifact generations: `112` rows from `20260321_140000`, `36` from `20260523_015947`, and only `1` from `20260525_010004_nopenalty`.
- Against the frozen post-policy holdout replay there are `0/149` exact matches and `0/149` loose signal-bar/symbol/side matches.
- Against the main policy replay there are `0/149` exact matches and `1/149` loose signal-bar/symbol/side match; the one current-artifact live row still has no exact replay match.
- Current live gate counts are useful diagnostically but not comparable to the current six-head replay: `93` portfolio rejections, `22` rank rejections, `19` liquidity rejections, `4` price-gap rejections, and `11` traded rows.
- Conclusion: final live-vs-replay decision parity requires a fresh run-scoped ledger for the current six-head package. The mixed historical ledger cannot be used to diagnose current-package live degradation.

### Current Deployment Mask Contract

- A one-shot shadow run with `EPM_RUN_SCOPED_PREDICTION_LEDGER=1` resolved the ledger to `data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/20260525_010004_nopenalty/prediction_ledger.parquet`, so the mixed-ledger issue is fixed for future live-test cycles.
- The earlier deployment failed closed because selected strategies had empty `lgbm_regime_mask` contracts and the legacy perps mask CSV did not contain the selected ids.
- Patch applied: both policy deployment writers now pass `market_mode` into the LGBM mask-contract loader and reject deployable rows with `missing_lgbm_mask_contract` under the LGBM backend.
- Current corrected run: `policy_params/best_policy_params_perps.json` and `strategy_for_inference_perps.json` select four strategies, all with embedded LGBM final-rule registry masks. Reconciliation reports `policy_optimized=7`, `trained_meta_model_covered=7`, `lgbm_mask_contract_covered=4`, `deployment_selected=4`, and `deployment_rejected=3`.
- Remaining action: run a fresh run-scoped live-test/shadow cycle with this corrected deployment and reconcile the generated ledger rows against the current replay decisions.

### Frozen Post-Policy Holdout

- Added `scripts/evaluate_frozen_policy_holdout.py` to score a post-policy holdout without changing the already selected simple-policy or portfolio parameters.
- The script uses label-backed rows from the source run, train-meta-frozen `model_state_meta.pkl`, saved per-strategy policy-rank references, the saved cross-strategy auction rank reference, saved deployment thresholds, and the saved `optimized_portfolio_policy_config.json`.
- It explicitly does not rank-normalize within the holdout and does not re-optimise portfolio parameters on holdout rows.
- Holdout window: `2026-05-22T00:00:00+00:00` through `2026-05-27T17:00:00+00:00`.
- Candidate/replay result after preserving Kraken explicit zero-volume 1m candles:
  - `1,335` local candidates before auction floor.
  - `729` candidates after global auction floor.
  - `73` accepted portfolio replay trades.
  - `11.70` trades/day.
  - `2.19%` mean accepted net return and `2.39%` mean accepted gross return.
  - final wallet `11583.00` from `10000`.
  - max drawdown `-0.13%`.
  - full-stop rate `28.77%`, timeout rate `16.44%`.
- Per-strategy local holdout candidates before auction floor:
  - long asset-vol/compression: threshold `0.68`, `172` rows, `52.33%` net hit, `1.57%` mean net.
  - long high-vol pullback/funding: threshold `0.65`, `266` rows, `51.13%` net hit, `1.27%` mean net.
  - long high-vol location/range: threshold `0.63`, `243` rows, `36.63%` net hit, `0.99%` mean net.
  - long local BB/channel: threshold `0.64`, `196` rows, `46.43%` net hit, `1.13%` mean net.
  - short asset-OI: threshold `0.60`, `364` rows, `64.56%` net hit, `1.83%` mean net.
  - short Bollinger/price-RV: threshold `0.68`, `94` rows, `41.49%` net hit, `1.31%` mean net.
- Interpretation: the frozen policy remains EV-positive on this short untouched policy-layer holdout, but several long heads and the short Bollinger head do not meet the desired 60% local net-hit target on this short slice. The portfolio replay is still strong because the global auction and portfolio constraints select a smaller subset.
- Execution-data result: the prior `0`-row 1m fetch was a loader bug, not missing Kraken data. Kraken returns explicit zero-volume 1m carry candles for inactive minutes; `_fetch_kraken_futures_charts_ohlcv(...)` was dropping those rows before delayed-entry replay. After preserving them for `1m` and allowing t+10 plus +1/+2/+3 fallback, the frozen holdout candidate metadata reports `entry_execution_source_counts={"delayed_1m_intraminute_proxy": 703, "theoretical_15m_open": 26}` and `delay_window_complete_rows=703` with the expected 11 candles. All 703 delayed rows used exact t+10 (`entry_delay_fallback_minutes=0`); the remaining 26 rows still lack usable t+10 through t+13 candles. Median `entry_gap_bps` and `delay_window_range_bps` are both `0.0`; max absolute entry movement is about `90.7 bps` on this short holdout.
- Accepted-trade execution-source check: `71/73` accepted trades use the delayed 1m proxy and `2/73` use theoretical 15m open. The delayed-only subset remains essentially unchanged: `63.38%` net hit, `2.19%` mean net, and `2.40%` mean gross.

### Timestamp / Horizon / Tradability

- `simple_policy_optimiser.py` applies a delayed-entry execution model and writes delay/adverse-move reports.
- Live inference writes signal, decision, expected entry, actual entry, and friction fields to `prediction_ledger.parquet`.
- Initial timestamp/horizon audit has been started in `timestamp_horizon_report.md`.
- Current code default and current regenerated artifact: simple policy optimisation uses `t+10m` delayed-entry 1m candles, applies that model before deployment threshold discovery, and persists delay-window fields.
- Live/shadow execution logs signal/decision/entry timestamps plus adverse gap and entry-delay effects.
- The next concrete check is to quantify whether OOS delayed-entry assumptions match live `signal_to_entry_seconds`, `decision_to_entry_seconds`, adverse gap, spread, and slippage distributions.

### Execution Realism

- Partial-fill analysis is out of scope for this system.
- Relevant market/stop execution questions:
  - Is the market/marketable entry price measured against the same reference used by OOS replay?
  - Are spread, slippage, fees, adverse hourly-close gap, and delayed-entry effects included in live EV gates?
  - Are stop trigger/fill gaps logged sufficiently to improve simple policy replay?
- Execution realism audit artifacts now exist:
  - `execution_assumption_matrix.csv`
  - `execution_realism_oos_breakdown.csv`
  - `execution_delay_sensitivity.csv`
  - `adverse_entry_gap_rejection_sensitivity.csv`
  - `slippage_sensitivity.csv`
  - `spread_cost_sensitivity.csv`
  - `live_execution_ledger_summary.json`
  - `oos_vs_inference_execution_reconciliation.md`
  - `fill_quality_report.md`
- Proper OOS delay sensitivity from the current regenerated `execution_attribution/` artifact shows the delayed-entry proxy is not the main OOS degradation:
  - global delayed-entry net mean `255.36 bps`, no-delay same-exit net mean `255.06 bps`, delay cost `-0.31 bps`.
  - long-dist delay cost `-0.81 bps`, long-loc `-0.32 bps`, short-dist `+0.14 bps`, short-loc `-0.15 bps`.
  - gross-to-net friction is consistently about `20.2 bps`.
- Candidate-table adverse-gap rejection simulations on the regenerated t+10 artifact are currently non-informative: rejecting already adverse `>=0.5%`, `>=1.0%`, or `>=1.5%` hourly-close-to-delayed-entry moves rejects `0` rows in the saved simulation table. This confirms the gate is harmless on this replay slice but does not yet validate it under live adverse gaps.
- Current execution realism scope is constrained by available data:
  - Quote/orderbook snapshots are not available, so spread/slippage fitting should use the live observations persisted around signal, order, and fill time plus fee fields from the executor.
  - Intra-delay path modelling is intentionally skipped for now because it would require denser data and another model.
  - Stop/trailing path realism is intentionally limited to the fields currently logged; reconstructing stop/trailing behavior from candle paths requires hours of post-entry data and is out of scope for the current pass.
- A systemic policy handoff issue is now confirmed: all four deployed strategy IDs have `meta_oof` files under `20260525_010004_nopenalty`, but `_load_label_events_for_strategy(...)` returns zero label-backed rows for the same deployed IDs:
  - long-loc: `148,389` `meta_oof` rows, `0` matching label rows.
  - long-dist: `112,154` `meta_oof` rows, `0` matching label rows.
  - short-loc: `123,570` `meta_oof` rows, `0` matching label rows.
  - short-dist: `133,512` `meta_oof` rows, `0` matching label rows.
  The final-fit model-generation fallback was therefore scoring feature-only policy events. Those rows can produce t+10 delayed-entry proxies, but they do not carry the forward return/path columns required to score executable exits. Feature-only replay is now opt-in only via `EPM_SIMPLE_POLICY_ALLOW_FEATURE_ONLY_REPLAY=1` and is labelled as diagnostic, not executable policy evidence.
- `meta_oof` is not a valid substitute for policy-OOS. The four current `meta_oof` files end inside the train_base/train_meta window:
  - long-loc: `2023-03-12 23:00 UTC` to `2026-01-19 05:00 UTC`.
  - long-dist: `2023-03-13 02:00 UTC` to `2026-01-19 03:00 UTC`.
  - short-loc: `2023-03-13 05:00 UTC` to `2026-01-19 06:00 UTC`.
  - short-dist: `2023-03-13 01:00 UTC` to `2026-01-19 04:00 UTC`.
  The strict `policy_optimiser` consumer plan predicts from `2026-01-22 10:00 UTC` through `2026-05-22 00:00 UTC`, after fit end `2026-01-19 06:00 UTC`.
- The optimiser now treats `artifacts/<run_id>/policy_oos_predictions/policy_oos_<strategy>_clf.parquet` as the normal policy-OOS handoff. It validates that prediction timestamps are after the policy fit end and inside the policy prediction window. `meta_oof`, feature-only rows, and final-fit generated predictions are diagnostic-only unless explicit override environment variables are set.
- Live ledger coverage is still insufficient for a full live execution realism verdict:
  - `decision_ts - signal_bar_ts` median is about `3895s` across 149 ledger rows, far above the optimiser's assumed 10-minute delayed-entry proxy.
  - `signal_to_entry_seconds` and `decision_to_entry_seconds` are absent in the prediction ledger; the one traded AR row has `entry_delay_adverse_bps=330.97`.
  - `realized_fee_bps` is absent from the prediction ledger.
  - This points to live observability/reconciliation as the next concrete fix, not OOS delay modelling as the primary proven root cause.
- Live entry observability patch now persists entry fee components and an entry-fee bps estimate in prediction-ledger rows when the live executor returns exchange fee information. This improves future reconciliation; historical rows already written without these fields remain incomplete.

### Data Source / Microstructure

- See `data_source_reconciliation.md`.
- A broad local parity rerun over all discovered overlap compared 306 symbols from `2026-05-01` through `2026-05-31`.
- Live hourly versus cached historical hourly is mostly clean where overlap exists: 302/306 rows match exactly within tolerance, with four mismatching symbols at the same cached historical timestamp (`2026-05-22 05:00:00 UTC`): ETH, GOOGLX, LTC, and ROSE.
- This does not support a general live-hourly data corruption root cause, but the historical-hourly comparison is still shallow in time because the cached historical reference currently contributes only one overlapping row per symbol.
- Live hourly versus execution 1m aggregate is not clean: 3,833/8,986 overlapping symbol-hour rows mismatch, and only 226/306 symbols have execution-1m overlap in the audited window.
- The execution 1m mismatch is now explained: the local execution-1m store is sparse delayed-entry sample data, not complete 1m history. Across the 8,986 overlapping symbol-hours, every hour has exactly one saved 1m row and zero hours have 60-minute completeness.
- The earlier active simple-policy candidate artifact was regenerated with the t+10 model, but it predates the trained-universe filter. Its execution 1m path remains useful as a data-collection risk signal because `3,281/48,913` rows still fell back to theoretical 15m-open execution. Future final policy optimisation should rerun this coverage check on the guarded trained-universe candidate set with `EPM_SIMPLE_POLICY_1M_DOWNLOAD=0` when the cache is expected to be complete.
- Live data diagnostics still need closed-candle equality over deeper historical-hourly overlap, stale/missing perp volume, OI/funding source age, orderbook proxy freshness, and spread/slippage snapshots.

### Portfolio Decision Reconciliation

See `portfolio_decision_reconciliation.md`.

- Earlier active optimiser candidate rows: `48,913`, date range `2026-01-22 10:00:00 UTC` through `2026-05-21 23:00:00 UTC`. This artifact predates the trained-universe policy-OOS filter.
- Earlier global auction replay reports `3,525` trades, `3.73%` mean net return/trade, `29.25` trades/day, final wallet `820,652,624`, and max drawdown `-1.90%`. Execution-attribution reporting covers `3,028` accepted trades after its reporting filters. These metrics are now historical evidence, not final deployable metrics, until the optimiser is rerun from guarded trained-universe policy-OOS rows.
- Guarded trained-universe optimiser rows: `27,485` cross-strategy candidates in the isolated full-budget report output. Portfolio replay completed with `3,236` accepted trades and objective `10024.095596`, but this replay is now marked invalid for deployment because the deployment payload had zero selected strategies and seven `missing_lgbm_mask_contract` rejections. The live-baseline replay over the same candidates had `2,685` trades and objective `2354.607861`.
- Execution-attribution accepted trades are distributed across all four deployed heads:
  - long-dist: `895`, `56.42%` net hit, `2.60%` mean net return.
  - long-loc: `464`, `67.03%` net hit, `3.09%` mean net return.
  - short-dist: `690`, `79.86%` net hit, `2.61%` mean net return.
  - short-loc: `979`, `83.45%` net hit, `2.21%` mean net return.
- The prior symptom where portfolio acceptance was almost entirely one strategy is not present in the current execution-attribution or global auction outputs.
- The local live ledger is not a clean current-run comparison table because most rows still reference `20260523_015947`; a fresh live-test ledger with current artifact diagnostics is needed before final live-vs-replay gate parity can be claimed.

## Fixes Applied

1. Reusable investigation skill terminology corrected:
   - Removed passive-limit execution language that does not apply to market/stop operation.
   - Replaced generic score-adjustment terminology with rank-normalization/rank-normalizer/rank-normalized terminology.
   - Preserved partial-bar language because it refers to data availability, not execution fills.
   - Validation passed.

2. Historical replay benchmark residual context and warmup fixed:
   - Added canonical BTC benchmark context when any required trained feature ends with `_bench_resid`.
   - Reloads the historical panel before feature computation if the required context expands the basket.
   - Uses the computed effective warmup window for panel loading; before this patch the script calculated `effective_lookback_hours` but still loaded only raw `--lookback-hours`.
   - Verified on the long-dist strategy: `ret4h_bench_resid` is no longer missing and base predictions are produced.

3. Historical replay feature cache and duplicate-compute path fixed:
   - Enabled rolling transformed-feature cache for historical replay while keeping latest-only snapshot cache disabled.
   - Fixed tail append semantics so keys absent from cache keep their full computed tail instead of being sliced away by the cache cursor.
   - Fixed merge semantics so per-symbol DataFrame materializations replace non-DataFrame placeholders from the shared feature path.
   - Verified on the long-dist strategy: cold replay did one feature tail build and completed feature load/compute in `428.0s`; immediate rerun loaded `79,296` rolling cached rows across `336` partitions and completed feature load/compute in `14.8s`.

4. Policy prediction-source provenance reporting fixed:
   - `simple_policy_optimiser.py` now records requested precomputed OOF separately from the actual prediction source.
   - Top-level policy metrics now include actual source strings and labels by strategy.
   - Per-strategy policy metrics now set `uses_precomputed_meta_oof` from the actual source, not the request flag.
   - This is reporting/provenance only; it does not alter scores, ranks, thresholds, model inputs, or trading behavior.

5. Offline feature handoff and lazy replay parity fixed:
   - `feature_generator` now falls back from exchange-scoped artifact roots to the parent data root when loading selected offline features, and it honors `EPM_ARTIFACT_SOURCE_RUN_ID` as a feature source override.
   - Lazy selected-feature caches are preserved through slicing, merging, stale-source checks, feature matrix extraction, and historical feature comparison instead of materializing hundreds of wide matrices for small parity probes.
   - `data_store.LazyFeatureDict` now supports timestamp/symbol value lookup without assembling the full feature DataFrame.
   - Verified on all four active heads with policy-rank samples from the saved reference.

6. Causal-transform state-current raw-feature bug fixed:
   - If live causal transform state is already current but the caller has raw recomputed features, the transform path no longer treats those raw features as transformed.
   - This prevents huge raw values such as `grind_score=482204.9375` from being fed into model parity audits when the selected-feature cache path is unavailable.

7. Strict inference candidate path parity and unscorable-candidate handling fixed:
   - `scripts/historical_inference_parity.py` can now call `_select_candidates_and_load_features(...)` through `--feature-load-path inference_candidate`, which exercises the live candidate masks, source-panel eligibility filter, selected-feature handoff, and model feature validation path.
   - `_latest_only_features(...)` now slices lazy selected features at one timestamp without assembling hundreds of full feature DataFrames.
   - `validate_required_feature_frames(...)` now understands `LazyFeatureDict` coverage without materializing it.
   - Candidates that pass masks/source checks but lack required model feature columns are now rejected before scoring instead of aborting the whole inference cycle. This was reproduced on short-dist (`HMSTR/USD:USD` missing `unwind_score`) and short-loc (eight sparse-feature candidates).
   - Verified on all four heads through the actual inference candidate path.

8. Live LGBM mask market-basket resolution fixed:
   - Strict live recompute exposed a systemic long-dist pre-mask divergence: the selected-feature policy handoff had candidates, but the live fast mask path returned `0/236`.
   - Root cause: the fast LGBM mask feature path computed `mkt_ret_eq_24h` over the full 236-symbol live universe because `market_basket` config uses spot-style symbols such as `BTC/USDT`, while Kraken perp columns use symbols such as `BTC/USD:USD`. The regular training/backtest feature path already resolves these by base asset.
   - Added shared base-symbol basket resolution and reused it in both the fast LGBM mask path and the regular basket/universe feature path.
   - Verified with a targeted unit test and strict first-hour live recompute: long-dist mask support recovered from `0/236` to `40/236` in the fast pre-pass and `24/236` after the full model-feature pass.
   - The explicit `market_basket` feature families now share the same resolver. Universe-wide features such as cross-sectional medians, dispersion, ranks, and peer context intentionally use all available columns and are not basket-symbol mapped.

9. Live/replay transformed-feature cache contract split and mask-only replay path fixed:
   - Persisted live transformed-feature cache keys now include an explicit cache namespace and the fitted feature-transform contract hash, in addition to symbols and required-feature hashes.
   - Mask-only features and model features now write/read separate cache namespaces (`mask` vs `model`), so cross-key fallback cannot reuse a model cache for a mask contract or vice versa.
   - Top-level live cache controls are preserved when `runtime_cfg` is present; before this, caller-level knobs such as cache namespace could be silently dropped inside `load_or_compute_features(...)`.
   - `scripts/historical_inference_parity.py --feature-load-path inference_candidate --skip-predictions` now stops after the LGBM pre-mask path instead of materializing the full 600+ feature model frame.
   - Verified on the long-dist strict first-hour replay: inference candidate feature load/compute dropped from the prior full model recompute path (`~537s`, near `10GB` RSS at merge) to a mask-only path of `7.4s` after panel/model-state load, with `40/236` long-dist mask support recovered.

10. Prediction-ledger live entry fee diagnostics fixed:
   - `_prediction_ledger_row(...)` now persists entry notional, base amount, entry fee quote/cost/currency/source, and derived `entry_fee_bps`.
   - When no round-trip `realized_fee_bps` exists yet, the ledger stores the known entry fee bps as the current realized-fee proxy for that open entry.
   - This matches the live executor, which already returns exchange entry fee fields and entry timing diagnostics.
   - Verified with a focused regression test.

11. Policy-OOS source-artifact provenance preflight added:
   - `scripts/generate_policy_oos_predictions.py` now runs a metadata-only preflight before loading or scoring model artifacts.
   - The preflight requires base and meta artifact manifests proving fit start/end, allowed training slice role, non-final-fit source, matching slice-plan hash, and a feature-contract hash.
   - If provenance is missing or unsafe, the generator writes `policy_oos_predictions/preflight_report.json` and exits before model loading/scoring.
   - Current `20260525_010004_nopenalty` artifacts pass preflight. `policy_oos_predictions/preflight_report.json` has `valid=true`, no errors, source model fit end `2024-09-08T04:00:00+00:00`, and policy prediction start `2026-01-22T10:00:00+00:00`.

12. Frozen policy holdout evaluator added:
   - Added `scripts/evaluate_frozen_policy_holdout.py`.
   - It scores post-policy label-backed rows with the train-meta-frozen model state, maps scores through saved policy-window rank references, applies saved deployment thresholds, applies the saved cross-strategy auction rank reference, and replays the saved portfolio policy config without holdout optimisation.
   - Fixed a positional-index bug in the delayed-entry helper call by resetting per-strategy local candidate indexes before delayed-entry execution modelling.
   - Verified with `python3 -m py_compile scripts/evaluate_frozen_policy_holdout.py` and `git diff --check`.

13. Kraken Futures 1m zero-volume carry candles preserved:
   - `_fetch_kraken_futures_charts_ohlcv(...)` no longer drops explicit zero-volume carry candles when `timeframe="1m"`.
   - Root cause: Kraken returns flat zero-volume 1m candles for inactive minutes; the generic suspicious-carry filter removed them before delayed-entry replay, causing rows to fall back to `theoretical_15m_open`.
   - The filter is still applied for coarser chart timeframes.
   - Added `tests/test_kraken_charts_ohlcv.py` to preserve this behavior.
   - Verified by direct Kraken probes for `CYBER/USD:USD`, `CTSI/USD:USD`, and `LQTY/USD:USD`, and by frozen holdout metadata showing `701/729` rows now use `delayed_1m_intraminute_proxy`.

14. Deployment LGBM regime-mask provenance handoff fixed:
   - A run-scoped shadow live check for `20260525_010004_nopenalty` failed closed before scoring because all six selected deployment strategies had empty `lgbm_regime_mask` contracts.
   - The fallback perps mask registry contained only four stale/older strategies and correctly filtered to `0` rows for the selected six.
   - Root cause: `policy_oos_retrain_strategy_source_perps.csv` carried lossy safe ids in `canonical_key` but did not carry the original parseable `base_event_trigger`. The original rules were still present in `data_perp/artifacts/20260523_015947/strategy_selection/candidate_strategies.json`.
   - `offline_optimisers/params_store.py` now preserves explicit `strategy_id` as the model/deployment identity and uses `base_event_trigger` as the parseable LGBM mask rule when present. Legacy registries without explicit ids keep the old sanitized-id behavior.
   - `policy_optimiser.py` and `simple_policy_optimiser.py` now require LGBM mask contracts for LGBM deployments, pass `market_mode` into the loader, and reject deployable rows with `missing_lgbm_mask_contract` instead of emitting monitor-only packages that fail at runtime.
   - Backfilled `policy_oos_retrain_strategy_source_perps.csv` with `strategy_id`, `base_event_trigger`, `trade_side`, `feature_keys_json`, and `market_mode=perps`, keeping the old `canonical_key` safe-id column for compatibility.
   - Backfilled the current `20260525_010004_nopenalty` deployment JSON copies with embedded parseable LGBM mask contracts for all six selected strategies.
   - Verified through `_load_lgbm_strategy_mask_rows(...)` and `_validate_lgbm_strategy_mask_coverage(...)`: selected cores `6`, embedded aliases `12`, coverage `ok`, and every loaded rule is parseable.
   - A fresh one-shot shadow cycle then loaded the six-strategy portfolio contract, training-live parity contract, six policy-param rows, 12 strategy aliases, and 12 embedded LGBM mask aliases successfully. It did not score features because the market kill switch blocked new entries on `MARKET_AVG_1H_MOVE_GT_5PCT` (`market_avg_1h_abs_move=5.02%`, halt until `2026-06-02T22:15:00Z`). This is a separate protective gate, not the previous mask-contract failure.

15. Incremental feature repair/backfill inefficiency reduced:
   - `feature_backfill_key_batch_size` now defaults to `0`, meaning a repair chunk computes all missing feature keys once per symbol chunk instead of recomputing the same expensive feature graph once per 100-key batch.
   - The chunked backfill path treats `key_batch_size=0` as no key batching.
   - The runtime source filter now pre-filters requested repair keys that the active portability/source policy would reject anyway. In the completed Kraken perps repair, this skipped unavailable orderbook/volume-source keys before feature computation instead of computing them and dropping them later.
   - The latest feature repair completed for `287` symbols and wrote `2,631` feature keys. It was a cache repair/backfill, not pure append-only: cache scan found `917/954` available keys, with `35` missing files and `252` missing-column symbols, then backfilled through `2026-05-30T18:00:00Z`.
   - Remaining hotspot: each symbol chunk still took several minutes (`~539-683s` compute and `~289-430s` save), so full repair is still heavy. The next efficiency target is true append-only rolling/EWMA state for rows after cache coverage is complete, plus avoiding wide parquet rewrites during repair saves.

16. Fresh live/shadow validation localized the current blocker:
   - A run-scoped one-shot shadow cycle for `20260525_010004_nopenalty` now loads the corrected four-strategy deployment package, policy thresholds, training-live parity contract, and embedded LGBM masks.
   - Hourly OHLCV refresh was incremental for the target hour: `149` scoped symbols, `fetch=0`, `skipped_existing=149`, elapsed `~2.6s`; panel load remains `~17.6s`.
   - The LGBM pre-mask path now works and no longer falls back to the full 900+ feature builder. Cold mask-tail compute used the dedicated 12-key `live_lgbm_mask_fast_path`, wrote a 12-feature snapshot/rolling cache through `2026-06-02T19:00:00Z`, and subsequent mask evaluation produced non-empty support for all four deployed strategies:
     - long-dist: `1/149`
     - long-loc: `100/149`
     - short-dist: `4/149`
     - short-loc: `2/149`
   - Source-parity filtering then accepted `118/149` symbols and rejected `31` for missing/stale `perp_volume` inputs (`quote_volume` missing, `volume` stale).
   - The actual live blocker is the model selected-feature cache, not mask support: `data_perp/features/20260523_015947` is stale for sampled deployed symbols (`BTC`/`ETH` max `2026-05-27T18:00:00Z`; `SOL`/`DOGE`/`KAITO` max `2026-05-30T18:00:00Z`) while live scoring needed `2026-06-02T19:00:00Z`.
   - Model-feature loading now preserves `LazyFeatureDict` instead of materializing all 739 loaded feature frames before scoring.
   - Runtime now fails closed for the `model` feature namespace when the selected-feature cache is stale, instead of silently recomputing a full 747-feature live tail and entering late. The one-shot failed with:
     `Live model feature cache is stale or incomplete ... cached_last_ts=2026-05-27 18:00:00+00:00 target_end_ts=2026-06-02 19:00:00+00:00 required_features=747`.
   - This is the correct safety behavior for live trading: model scoring must use training-path selected features that are current, or explicitly run in replay/audit mode with `live_model_feature_tail_recompute_enabled`.

17. Policy-OOS trained-universe contract fixed:
   - `scripts/generate_policy_oos_predictions.py` now loads the deployable trained/inference universe with `load_trained_symbol_universe(...)`, filters every generated policy-OOS source to that universe, records dropped rows/symbols in the manifest, and aborts if a strategy has no deployable policy-OOS rows left.
   - `simple_policy_optimiser.py` now validates loaded policy-OOS rows against the same trained universe and fails closed when a verified policy source includes non-deployable symbols.
   - `simple_policy_optimiser.py` now supports `EPM_SIMPLE_POLICY_OUTPUT_RUN_ROOT`, allowing audit optimiser runs to write policy params, candidates, metrics, and rank references into an isolated report root without overwriting active deployment artifacts.
   - `policy_rank_reference.py` now supports an explicit output directory for diagnostic rank-reference persistence while preserving the existing default artifact path.
   - Verification: regenerated trained-universe policy-OOS files have zero outside-universe symbols across all seven strategies. A guarded isolated optimiser diagnostic produced `27,485` candidate rows and per-strategy threshold metrics without touching active deployment artifacts.

18. Portfolio replay output isolation fixed:
   - `simple_policy_optimiser.py` now passes `output_dir=EPM_SIMPLE_POLICY_OUTPUT_RUN_ROOT/portfolio_policy_replay` into `run_portfolio_policy_replay(...)` when an output override is configured.
   - `portfolio_policy_replay.py` now accepts `persist_live_artifacts`; default production behavior still persists live portfolio policy artifacts, while isolated audit runs can write replay reports without mutating `data_perp/artifacts/<run_id>/policy_params`.
   - Verification: a fixed-config smoke replay wrote `portfolio_policy_replay_report.json`, `per_candidate_replay_decisions.parquet`, and the optimized policy config under `portfolio_replay_isolation_smoke/`. Focused portfolio replay tests now cover both live persistence and no-live-persistence modes.

19. Empty deployment selection now fails closed before replay:
   - The guarded full-budget run exposed a stronger contract bug: all seven verified policy strategies were rejected for `missing_lgbm_mask_contract`, leaving `strategies=[]`, yet portfolio replay still ran over all candidate rows.
   - `simple_policy_optimiser.py` now raises before portfolio replay when candidate rows exist but no deployable strategies are selected, including rejection-reason counts in the error.
   - Verification: added a focused regression test for the missing-mask/no-selected-strategy case.

20. Policy-artifact root lookup fixed for inference parity and isolated optimiser outputs:
   - `_load_normalized_threshold_map(...)`, `_load_policy_selection_rules(...)`, `_load_embedded_lgbm_strategy_mask_rows(...)`, and `_load_selected_strategy_cores(...)` now search the same `_policy_artifact_bases(...)` roots used by replay/parity code.
   - Root cause: isolated policy runs could write valid thresholds, selected strategies, and mask contracts under a report root, but inference helpers still preferred the canonical artifact path. This caused stale fourth-strategy contract reads during parity replay.
   - Verification: the long-bars replay loaded the isolated `strategy_for_inference_perps.json` and reported embedded LGBM final-rule registry mask coverage before scoring.

21. Feature-store stale-delta overwrite fixed:
   - `data_store._merge_duplicate_feature_rows(...)` now preserves existing populated base cells before delta cells when duplicate timestamps are merged.
   - `append_symbol_features(...)` now masks incoming delta values where the existing tail already has a populated value, so delta stores can append new rows or fill missing cells but cannot silently overwrite base-history feature values.
   - Root cause: selected-feature reads merged base Parquet and DuckDB delta rows at duplicate timestamps; stale delta rows could win after index sorting and produce large feature mismatches such as `dist_ema50_atr=-2.053749` versus the training-path value near `-0.021476`.
   - Verification: `tests/test_training_feature_availability.py::test_feature_delta_append_visible_to_selected_loader` passes, and the long-bars strict replay now reports `feature_mismatches_gt_1e_6=0` and `feature_max_abs_diff=0.0`.

## Regression Tests Added

- Updated `tests/test_live_feature_universe_parity.py` so missing live orderbook source remains a hard error unless training health proves the required feature was neutral/constant.
- Added historical replay tests for benchmark-residual context symbol insertion.
- Added replay/runtime feature tests for rolling cache config, market-wide regime broadcasting, non-DataFrame placeholder replacement, and tail append behavior for newly materialized keys.
- Added lazy selected-feature tests proving merge and candidate matrix extraction do not assemble wide feature frames.
- Added strict inference-path tests for latest-only lazy feature slicing, lazy feature contract validation, and sparse lazy candidate rejection.
- Added shared basket-resolution tests proving spot-style config symbols map to perp columns by base asset, and that the live LGBM mask fast path computes `mkt_ret_eq_24h` on the configured basket rather than the full universe.
- Added cache namespace and mask-only replay tests proving model features are not materialized when the caller only needs mask support, and proving mask/model transformed caches have distinct exact-contract keys.
- Added prediction-ledger row test proving live traded rows persist realized entry price, entry fee diagnostics, and signal/decision-to-entry timing fields.
- Added deployment contract tests proving LGBM deployments reject missing regime masks and embed market-specific parseable mask contracts.
- Added a params-store handoff regression test proving explicit `strategy_id` is preserved while `base_event_trigger` is used as the parseable LGBM mask rule.
- Added a feature-store regression test proving DuckDB feature deltas honor timestamp filters during selected-feature reads. This fixed the replay model-feature load from minutes-per-25-files to ~20 seconds for 285 keys across 152 symbols at one exact timestamp.
- Added training-live parity contract tests requiring base/native/meta/rank artifacts and rejecting policy rank-reference symbols outside the trained/inference universe.
- Added an optimiser-side guard so `simple_policy_optimiser` refuses verified policy-OOS prediction sources whose symbols are outside the trained/inference universe instead of silently admitting non-deployable rows.
- Added a policy-OOS generator regression test proving non-trained symbols are dropped before policy source files are written.
- Added a portfolio replay regression test proving isolated audit replays do not write live `policy_params/optimized_portfolio_policy_config.json` or `training_live_parity_contract.json`.
- Added an optimiser deployment regression test proving portfolio replay refuses an empty selected-strategy set.

## Next Steps

1. Regenerate and validate the training-live parity contract so it includes the latest artifact hashes and the new policy-root/delta-store fixes.
2. Re-run the run-scoped shadow cycle and verify it reaches model scoring without `live_model_feature_tail_recompute_enabled`.
3. Reconcile live final decision rows against the repaired `portfolio_policy_replay/per_candidate_replay_decisions.parquet`.
4. Reconcile timestamps and label horizons end to end, especially signal-bar close, decision time, expected t+10 entry, actual entry, stop/trailing start, and evaluation window.
5. Extend live-vs-historical OHLCV parity to deeper historical-hourly overlap and close the remaining execution-1m coverage gap.
6. Reconcile stop trigger/fill gaps and rejected market/stop order history from trade-executor logs or exchange order history.
7. Decide whether the current marginal threshold-band weakness is acceptable only with portfolio/global-auction constraints, or whether per-strategy deployment gates should explicitly require stronger local threshold economics before live deployment.
