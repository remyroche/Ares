# Extreme Price Movements Inference Mismatch Investigation

Status: updated, 2026-06-02.

## Executive Summary

- Root cause: current evidence points to pipeline/evidence-contract issues, not model failure. The first confirmed mismatch was feature-handoff/source parity. The current six-head package has verified policy-OOS prediction artifacts generated from train-meta-frozen model state, while the deployed scorer is a final-fit bundle. Those are intentionally different evidence roles and now need explicit provenance guards so policy evidence cannot be mixed with a different deployed scoring contract.
- Current confidence: high for current policy-OOS artifact provenance, threshold-band economics, and replay/rank contract instrumentation; medium for the broader live degradation question until fresh run-scoped live decisions, execution realism, and data-source audits are completed.
- Current policy result after strict policy-OOS regeneration: `48,913` exported candidates across seven optimized strategies. The deployment contract selects four strategies (`2` long, `2` short) and rejects three long strategies only because of the top-2-per-side cap. Global auction replay reports `3,525` trades, `3.73%` mean net return/trade, `29.25` trades/day, and max drawdown `-1.90%`; execution-attribution reporting covers `3,028` accepted trades with `72.13%` net hit and `2.55%` mean net return/trade.
- Threshold-band result: the global 0.80-0.85 auction-rank band is already positive (`57.41%` net hit, `2.82%` mean net, `1.21%` median net); higher ranks improve hit rate and EV, but the lower band is not negative-then-rescued.
- Strict post-policy frozen holdout result: using the saved six-head thresholds, saved policy-rank references, train-meta-frozen model state, saved portfolio policy config, fixed t+10 Kraken 1m delayed-entry candles, and a +1/+2/+3 minute fallback on `2026-05-22T00:00:00Z` through `2026-05-27T17:00:00Z` produced `729` auction-floor candidates and `73` accepted replay trades with `2.19%` mean accepted net return, `11.70` trades/day, final wallet `11583.00`, and max drawdown `-0.13%`. This is a short untouched policy-layer holdout with strong portfolio result, but not yet a complete live execution proof.
- Current policy-source localization: `simple_policy_optimiser/rank_reference/*.parquet` is generated from `policy_oos_predictions/*_clf.parquet`. Joining strategy references to policy-OOS rows on `timestamp`, `symbol`, and `strategy_id` gives exact score equality; applying the rank-normalization score transform also reproduces the saved reference scores exactly.
- Current deployment blocker status: the earlier missing-LGBM-mask blocker is fixed for the newly written deployment contract. The corrected optimiser run reports `lgbm_mask_contract_covered=4` for the four selected strategies, all with `regime_mask_source=embedded_lgbm_final_rule_registry`. The remaining live proof is a fresh run-scoped cycle and decision reconciliation, not another mask-contract repair.
- Guard status: `simple_policy_optimiser` now requires policy-OOS manifests to prove train-meta-frozen provenance, a non-final-fit source, the expected candidate/execution source, rank-normalization metadata, source-model hash, and a passing artifact preflight. The rank-reference loader now fails closed when a regenerated manifest declares an invalid explicit policy-OOS contract.
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
- Rank-reference provenance:
  - current/source rank-reference parquet comparison;
  - current `meta_oof` timestamp-range check;
  - log inspection of `logs/unified_20260528_121034.log`;
  - direct final-fit model reproduction of three saved long-dist rank-reference rows through the policy feature handoff.
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
- Policy-OOS scoring contract: per-file manifests are now required to declare `prediction_source=generated_from_train_meta_state:*`, `candidate_rows_source=policy_slice_feature_events`, `executable_path_source=simple_policy_optimiser_recomputes_from_ohlcv_and_execution_1m`, a rank-normalization declaration, a source model-state hash, and a passing source-artifact preflight.
- Current simple-policy candidates: `48,913` rows across seven optimized strategies, generated with configured delayed entry `10` minutes and rank-band audit artifacts under `simple_policy_optimiser/rank_threshold_band_report.*`.
- Current delayed-entry coverage: `45,632/48,913` candidate rows (`93.29%`) used complete cached 1m t+10 execution windows; `3,281` rows fell back to `theoretical_15m_open`. All delayed rows had exact `10.0` minute entry delay and complete 11-candle windows. This is now t+10 evidence, but the fallback rows remain an execution-data coverage gap.
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
- `rank_threshold_band_report.csv` now proves marginal threshold economics separately from cumulative higher-rank economics. The 0.80-0.85 global auction band is positive on mean and median net return, but hit rate is only `57.41%`, so the trade-off is EV-positive but not uniformly above the 60% hit-rate target.
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
- The saved simple-policy candidate artifact has now been regenerated with the t+10 model. The execution 1m path remains a data-collection risk because `3,281/48,913` rows still fell back to theoretical 15m-open execution and future spread/slippage fine-tuning depends on this path.
- Live data diagnostics still need closed-candle equality over deeper historical-hourly overlap, stale/missing perp volume, OI/funding source age, orderbook proxy freshness, and spread/slippage snapshots.

### Portfolio Decision Reconciliation

See `portfolio_decision_reconciliation.md`.

- Current optimiser candidate rows: `48,913`, date range `2026-01-22 10:00:00 UTC` through `2026-05-21 23:00:00 UTC`.
- Global auction replay reports `3,525` trades, `3.73%` mean net return/trade, `29.25` trades/day, final wallet `820,652,624`, and max drawdown `-1.90%`. Execution-attribution reporting covers `3,028` accepted trades after its reporting filters.
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

## Next Steps

1. Bring `data_perp/features/20260523_015947` current through the actual training-path incremental feature generator, through at least the latest closed hourly candle needed by live scoring.
2. Rerun the run-scoped shadow cycle and verify it reaches model scoring without `live_model_feature_tail_recompute_enabled`.
3. Reconcile live final decision rows against `portfolio_policy_replay/per_candidate_replay_decisions.parquet` from that fresh current-artifact ledger.
4. Reconcile timestamps and label horizons end to end, especially signal-bar close, decision time, expected t+10 entry, actual entry, stop/trailing start, and evaluation window.
5. Extend live-vs-historical OHLCV parity to deeper historical-hourly overlap.
6. Close the remaining execution-1m coverage gap: `3,281/48,913` current candidates still used theoretical 15m-open fallback instead of complete t+10 1m windows.
7. Reconcile stop trigger/fill gaps and rejected market/stop order history from trade-executor logs or exchange order history.
8. Add a top-level policy-OOS contract entry to the rank-reference manifest, not only per-strategy entries.
9. Convert the current report evidence into the final root-cause section once fresh live execution observability and data-source audits are complete.
