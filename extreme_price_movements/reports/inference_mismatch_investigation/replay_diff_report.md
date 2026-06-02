# Replay Diff Report

Status: updated after benchmark residual, replay cache, feature-handoff, and lazy-materialization fixes, 2026-06-01.

## Command

```bash
EPM_EXCHANGE=kraken python3 scripts/historical_inference_parity.py \
  --data-root data_perp \
  --artifact-data-root data_perp \
  --run-id 20260525_010004_nopenalty \
  --market-mode perps \
  --sample-source policy_rank_reference \
  --strategy-id long_dist_weekly_vwap_0_074823022_loc_prev_week_range_pos_48_0_48354843_mkt_ret_eq_24h_-0_43956268_volume_autocorr_48_-0_38378653 \
  --sample-rows 3 \
  --lookback-hours 720 \
  --basket-mode sample \
  --output-dir extreme_price_movements/reports/inference_mismatch_investigation/historical_parity_smoke
```

The same command without `EPM_EXCHANGE=kraken` incorrectly resolved the market data component to Binance USD-M and loaded no OHLCV panel.

## Initial Result

- Runtime: 15.5s.
- Samples: 3 rows from one policy rank-reference timestamp.
- Basket mode: sample-only, so cross-asset features are not a parity proof.
- Feature comparison: 1,827 checked cells, 24 common rows, 0 mismatches above `1e-6` among comparable values.
- Prediction comparison: 3 rows written, but alpha/model scoring did not produce comparable base/meta prediction diffs.
- First concrete blocker: alpha model feature alignment reported missing trained feature `ret4h_bench_resid`.

## Fix

The replay harness now adds the canonical Kraken perps BTC benchmark (`BTC/USD:USD`) to the diagnostic basket whenever a trained feature contract requires a `*_bench_resid` feature. If this expands the basket after the first panel load, the harness reloads the historical panel before feature computation. This fixes the missing `ret4h_bench_resid` blocker without changing model, ranking, trading, or live inference behavior.

The harness also now uses its computed effective warmup window for panel loading. Previously it logged `effective_lookback_hours` but loaded only the raw `--lookback-hours`, which could leave long-window residual/transform features non-finite even when the policy optimiser had valid rows.

Regression coverage:

- `tests/test_replay_live_signal_predictions.py::test_historical_parity_adds_benchmark_context_for_residual_features`
- `tests/test_replay_live_signal_predictions.py::test_historical_parity_keeps_sample_basket_without_residual_features`

## Retest

- Sample-basket replay: `ret4h_bench_resid` is no longer missing; base predictions are produced for the three sampled rows.
- Full-basket replay on the same long-dist strategy: loaded 236 trained symbols, feature compute completed in 53.1s, and base predictions were produced for the three sampled rows.
- Feature comparison in both reruns: 1,836 checked cells, 30 common rows, 0 mismatches above `1e-6` among comparable values.

## Previous Divergence

The full-warmup single-strategy replay now produces both base and meta predictions for the three sampled long-dist policy-rank rows. The previous non-finite meta blocker disappeared after using the effective warmup window and materializing the market-wide regime score columns as per-symbol feature frames.

Current observed prediction differences versus the saved policy rank-reference are still material:

- `policy_calibrated_score_max_abs_diff`: `0.1342082198043293`
- `policy_rank_pct_max_abs_diff`: `0.2880368460418061`
- `prediction_rows`: `3`

These are not yet evidence of model decay. They are a replay-vs-policy-reference mismatch that must be localized next: feature values, model/rank-normalizer inputs, rank-reference distribution, or policy reference generated from a different artifact/config snapshot.

## Resolved Divergence

The material score/rank mismatch above was caused by using the wrong feature handoff for the comparison. The saved policy rank-reference was generated from the exchange-scoped offline selected-feature path:

`data_perp/exchanges/krakenfutures/features/20260525_010004_nopenalty -> data_perp/features/20260523_015947`

When the historical replay uses that same selected-feature source, all four active heads reproduce the saved policy reference to float precision:

| head | score max abs diff | rank-pct max abs diff | prediction rows |
| --- | ---: | ---: | ---: |
| long-dist | 1.1398097299331056e-08 | 3.936542928001385e-05 | 3 |
| long-loc | 1.2172013008626692e-08 | 4.864759680869857e-05 | 3 |
| short-dist | 7.229255527541056e-09 | 3.943217665616783e-05 | 3 |
| short-loc | 1.180954556367908e-08 | 3.471498993268263e-05 | 3 |

Evidence directories:

- `historical_parity_long_dist_source_features_no_rolling`
- `historical_parity_long_loc_source_features_no_rolling`
- `historical_parity_short_dist_source_features_no_rolling`
- `historical_parity_short_loc_source_features_no_rolling`

This proves the rank-reference parquet is coherent with the final-fit model bundle and the policy feature handoff. It does not prove OOS performance, because final-fit model predictions are not strict policy-OOS evidence if the final-fit bundle trained through the policy period. Policy metrics require a dedicated `policy_oos_predictions/` handoff whose model training end precedes the prediction slice.

## Strict Inference Candidate Path Update

The replay harness now supports `--feature-load-path inference_candidate`, which calls the actual live candidate path, `_select_candidates_and_load_features(...)`. This verifies more than direct model scoring: it exercises LGBM strategy masks, source-panel eligibility filtering, selected-feature fallback from exchange scope to parent data root, lazy feature extraction, and strict model-feature coverage validation.

Strict-path results for three sampled policy-rank rows per head:

| head | mask pass | source eligible | extra sparse-feature rejects | score max abs diff | rank-pct max abs diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| long-dist | 12/236 | 165/236 | 0 | 1.1398097299331056e-08 | 3.936542928001385e-05 |
| long-loc | 4/236 | 167/236 | 0 | 1.2172013008626692e-08 | 4.864759680869857e-05 |
| short-dist | 19/236 | 156/236 | 1 | 7.229255527541056e-09 | 3.943217665616783e-05 |
| short-loc | 115/236 | 164/236 | 8 | 1.180954556367908e-08 | 3.471498993268263e-05 |

The short-dist replay initially failed because `HMSTR/USD:USD` passed the short-dist mask and source checks but lacked the sparse required `unwind_score` feature column. The fix rejects such unscorable candidates before model validation instead of aborting the entire cycle. Short-loc exposed the same class at larger scale: eight candidates passed masks/source checks but lacked one or more required model feature columns.

Evidence directories:

- `historical_parity_long_dist_inference_candidate_path`
- `historical_parity_long_loc_inference_candidate_path`
- `historical_parity_short_dist_inference_candidate_path`
- `historical_parity_short_loc_inference_candidate_path`

Remaining strict-path limitation: the replay is still using the offline selected-feature handoff for prediction parity. This proves the live candidate path can reproduce the policy reference when pointed at the same feature source, but it does not prove live recomputed feature generation is identical to the offline selected-feature generation.

## Rank-Reference Provenance Update

The saved policy rank-reference is not stale and is not a copy from `20260523_015947`.

- Current long-dist rank-reference sha16: `0a8636c05192e906`, rows: `25,403`.
- `20260523_015947` long-dist rank-reference sha16: `ca67ab8f11efd35f`, rows: `246,714`.
- Joined current/source rows: `21,149`.
- Current/source score mean absolute difference: `0.14120904469502452`; max absolute difference: `0.5993017554283142`.
- Exact score matches at `1e-12`: `0`.

The current `meta_oof` parquet files under `20260525_010004_nopenalty` end inside the train_base/train_meta window around `2026-01-19`, while the strict policy optimiser prediction slice starts at `2026-01-22 10:00:00+00:00`. Logs from `logs/unified_20260528_121034.log` show all four precomputed meta OOF files had no rows in the policy slice, after which `simple_policy_optimiser` generated policy-slice predictions from final full-fit inference models and persisted rank references for all four strategies. That handoff is useful for final-fit/live parity diagnostics, but it is not rigorous OOS evidence.

Using the exact exchange-scoped policy feature handoff,
`data_perp/exchanges/krakenfutures/features/20260525_010004_nopenalty -> ../../../features/20260523_015947`, the first three long-dist policy rank-reference scores reproduce to float precision from the final-fit model bundle:

- AAVE/USD:USD: abs diff `1.139810e-08`.
- SPK/USD:USD: abs diff `2.617539e-08`.
- PORTAL/USD:USD: abs diff `1.244253e-08`.

So the material replay-vs-policy-reference mismatch is specifically live-style recomputed features versus the offline selected-feature policy handoff, not stale rank-reference parquet. Detailed evidence is in `rank_reference_provenance.md`.

## Replay Efficiency Fix

The historical replay path had two confirmed efficiency bugs:

1. The harness disabled the rolling transformed-feature cache, so a replay could not hit the cheap timestamp-partitioned cache even after a previous run had generated identical transformed features.
2. The tail merge dropped newly materialized market-wide regime keys because `compute_features_hourly` emitted `regime_trend_score` and `regime_vol_score` as Series, while inference expects per-symbol DataFrames. The merge refused to replace the Series placeholders, the keys were treated as missing, and `_backfill_missing_requested_keys()` launched a second `compute_features_hourly()` pass. In the reproduced run this pushed RSS close to 10 GB.

Patches:

- `scripts/historical_inference_parity.py` now enables the rolling replay cache while keeping latest-only snapshot cache disabled.
- `feature_generator._merge_missing_feature_dicts()` now replaces non-DataFrame placeholders with DataFrame feature panels.
- `feature_generator._slice_tail_features_for_cache_append()` keeps full tail history for feature keys absent from the cache while still appending only new rows for keys already cached.
- Market-wide features from `mkt_gates` are broadcast to the full symbol universe before model scoring.

Retest:

- Cold single-strategy full-basket replay: feature load/compute completed in `428.0s`; one tail compute only; no second shared-feature backfill; persisted `336` timestamp partitions, `615` features, `79,296` rows.
- Immediate rerun: loaded the rolling cache in `12.183s`; feature load/compute completed in `14.8s`; total replay elapsed `43.9s`; no `compute_features_hourly_tail` run.

Remaining efficiency target: the cold replay still spends most time in base feature generation, especially OI/perp derivative and CausalTransform stages. That is lower priority than the current parity divergence because cache-hit replay is now cheap enough for iterative diagnostics.

## Additional Replay/Inference Efficiency Fix

The selected-feature cache hit still had an avoidable materialization path: merge and validation helpers converted `LazyFeatureDict` selected-feature payloads into hundreds of full DataFrames even for three-row audits. This made a cheap cache hit look like another heavy replay.

Patches:

- `LazyFeatureDict.latest_values_at()` supports timestamp/symbol lookups without wide DataFrame assembly.
- `get_features_for_candidates()` uses the lazy lookup path when available.
- `_merge_missing_feature_dicts()`, `_drop_stale_live_sensitive_features()`, `_ensure_required_symbol_columns()`, and `_slice_feature_window()` preserve lazy payloads instead of eagerly iterating all items.
- `historical_inference_parity._compare_features()` compares selected timestamp/symbol values through the lazy lookup path.
- `load_cached_features_for_inference()` falls back from exchange-scoped artifact roots to the parent data root and accepts `EPM_ARTIFACT_SOURCE_RUN_ID`/`--feature-source-run-id` as the selected-feature source.

Retest:

- Long-dist selected-feature replay completed in `104.2s`.
- Long-loc selected-feature replay completed in `106.0s`.
- Short-dist selected-feature replay completed in `127.0s`.
- Short-loc selected-feature replay completed in `111.1s`.
- No full historical tail recompute occurred in these source-feature parity runs.

Remaining efficiency target: loading 236 symbol parquet files for the selected-feature handoff still costs about 28-36s per process. A multi-head replay should share the loaded feature payload and model state across heads instead of launching one process per head.

## Additional Strict-Path Efficiency / Safety Fix

The actual inference candidate path still had two avoidable strict-replay problems:

1. Latest-only mask evaluation converted lazy selected features into normal DataFrames. `_latest_only_features(...)` now reads a single timestamp by raw array offset, with timestamp positions computed once per symbol.
2. Strict model-feature validation treated a `LazyFeatureDict` as empty. `validate_required_feature_frames(...)` now checks lazy key and symbol coverage without materializing the payload.

The strict path now completes all four heads. The remaining cold runtime is dominated by selected parquet loading and orderbook summary materialization, not latest-only mask extraction.
