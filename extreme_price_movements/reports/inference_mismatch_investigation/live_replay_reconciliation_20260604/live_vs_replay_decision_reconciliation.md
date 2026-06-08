# Live vs Replay Decision Reconciliation

Status: updated, 2026-06-02.

## Inputs

- Live ledger: `data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/20260525_010004_nopenalty/prediction_ledger.parquet`
- Replay decisions: `data_perp/artifacts/20260525_010004_nopenalty/portfolio_policy_replay/per_candidate_replay_decisions.parquet`
- Replay candidates: `data_perp/artifacts/20260525_010004_nopenalty/simple_policy_optimiser/simple_policy_candidates.parquet`

## Summary

- Live ledger rows: `87`.
- Replay decision rows: `14731`.
- Live exact replay matches on signal timestamp, symbol, side, and strategy: `0`.
- Live loose replay matches on signal timestamp, symbol, and side: `0`.
- Live rows from `20260525_010004_nopenalty`: `87`.
- Current-run exact replay matches: `0`.

## Live Artifact Mix

Model artifact run ids:

| model_artifact_run_id     |   rows |
|:--------------------------|-------:|
| 20260525_010004_nopenalty |     87 |

Policy artifact run ids:

| policy_artifact_run_id    |   rows |
|:--------------------------|-------:|
| 20260525_010004_nopenalty |     87 |

## Live Gate Distribution

| portfolio_decision   |   rows |
|:---------------------|-------:|
| rank_rejected        |     86 |
| traded               |      1 |

Portfolio reject reasons:

| portfolio_reject_reason                  |   rows |
|:-----------------------------------------|-------:|
| rank_below_dynamic_threshold             |     80 |
| missing_policy_rank_reference_percentile |      6 |
| NA                                       |      1 |

Liquidity reject reasons:

| liquidity_reject_reason   |   rows |
|:--------------------------|-------:|
| NA                        |     87 |

## Replay Gate Distribution

| rejection_reason                    |   rows |
|:------------------------------------|-------:|
| symbol_already_open                 |   5360 |
| below_dynamic_threshold             |   3136 |
| symbol_in_cooldown                  |   2822 |
| accepted                            |   2513 |
| max_new_entries_per_bar_reached     |    490 |
| max_concurrent_positions_reached    |    280 |
| max_concurrent_per_strategy_reached |    130 |

## Replay Strategy Summary

| side   | strategy_id                                                                                                                                                                                     |   rows |   accepted |   mean_net_return |   net_hit_rate |
|:-------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------:|-----------:|------------------:|---------------:|
| long   | long_bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828 |   6315 |       1137 |         0.025337  |       0.577672 |
| long   | long_dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_leverage_build_score_0_45107844_return_autocorr_48_1_18643_rolling_range_20_-0_25967735                                        |   5195 |        771 |         0.0242983 |       0.534937 |
| short  | short_asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                               |   2747 |        511 |         0.0372189 |       0.768839 |
| short  | short_bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385_price_rv_15d_robust_z_0_060036644                                                                                           |    474 |         94 |         0.0203422 |       0.704641 |

## Current-Run Live Rows

| signal_bar_ts             | symbol      | side   | strategy_id                                                                                                                                                                                | portfolio_decision   | was_traded   |   normalized_rank_score |   final_threshold |   expected_total_entry_friction_bps |   entry_delay_adverse_bps | replay_exact_match   |
|:--------------------------|:------------|:-------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------|:-------------|------------------------:|------------------:|------------------------------------:|--------------------------:|:---------------------|
| 2026-06-04 08:00:00+00:00 | SOL/USD:USD | long   | bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828 | rank_rejected        | False        |               0.569382  |            0.6322 |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | TRX/USD:USD | long   | bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828 | rank_rejected        | False        |               0.657462  |            0.6322 |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | XRP/USD:USD | long   | bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828 | rank_rejected        | False        |               0.464025  |            0.6322 |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | SNX/USD:USD | short  | asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                                | rank_rejected        | False        |               0.0823561 |            0.58   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | SOL/USD:USD | short  | asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                                | rank_rejected        | False        |               0.0631093 |            0.58   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | UNI/USD:USD | short  | asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                                | rank_rejected        | False        |               0.0574624 |            0.58   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | WLD/USD:USD | short  | asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                                | rank_rejected        | False        |               0.148516  |            0.58   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | XRP/USD:USD | short  | asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                                | rank_rejected        | False        |               0.0565096 |            0.58   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | YGG/USD:USD | short  | asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                                | rank_rejected        | False        |               0.0207475 |            0.58   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | WLD/USD:USD | short  | bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385_price_rv_15d_robust_z_0_060036644                                                                                            | rank_rejected        | False        |               0.273593  |            0.68   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | SOL/USD:USD | long   | bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828 | rank_rejected        | False        |               0.569382  |            0.6322 |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | TRX/USD:USD | long   | bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828 | rank_rejected        | False        |               0.657462  |            0.6322 |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | XRP/USD:USD | long   | bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828 | rank_rejected        | False        |               0.464025  |            0.6322 |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | SNX/USD:USD | short  | asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                                | rank_rejected        | False        |               0.0823561 |            0.58   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | SOL/USD:USD | short  | asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                                | rank_rejected        | False        |               0.0631093 |            0.58   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | UNI/USD:USD | short  | asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                                | rank_rejected        | False        |               0.0574624 |            0.58   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | WLD/USD:USD | short  | asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                                | rank_rejected        | False        |               0.148516  |            0.58   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | XRP/USD:USD | short  | asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                                | rank_rejected        | False        |               0.0565096 |            0.58   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | YGG/USD:USD | short  | asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                                | rank_rejected        | False        |               0.0207475 |            0.58   |                                 nan |                       nan | False                |
| 2026-06-04 08:00:00+00:00 | WLD/USD:USD | short  | bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385_price_rv_15d_robust_z_0_060036644                                                                                            | rank_rejected        | False        |               0.273593  |            0.68   |                                 nan |                       nan | False                |

## Interpretation

- The current live ledger is not a clean live-vs-replay parity table for the six-head package because it mixes three artifact generations.
- Only `87` live row references `20260525_010004_nopenalty` in either model or policy artifact fields, and it does not match the frozen replay window/strategy set.
- Replay decisions reject mainly on portfolio state gates (`symbol_already_open`, dynamic thresholds, position/concurrency, cooldown). Live rejects include rank, min-notional sizing, spread, stale ticker, and stale adverse price movement gates that are not represented one-for-one in the portfolio replay artifact.
- Therefore the next valid test is a fresh live-test cycle using the current six-head package with the ledger cleared or namespaced to this run, then replaying the same signal bars with the same initial portfolio state.
