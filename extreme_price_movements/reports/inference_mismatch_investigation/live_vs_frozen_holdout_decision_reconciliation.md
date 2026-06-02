# Live vs Replay Decision Reconciliation

Status: updated, 2026-06-02.

## Inputs

- Live ledger: `data_perp/exchanges/krakenfutures/live_state/prediction_ledger.parquet`
- Replay decisions: `data_perp/artifacts/20260525_010004_nopenalty/policy_holdout_frozen_replay_1m_fallback/portfolio_policy_replay/per_candidate_replay_decisions.parquet`
- Replay candidates: `data_perp/artifacts/20260525_010004_nopenalty/policy_holdout_frozen_replay_1m_fallback/simple_policy_holdout_candidates.parquet`

## Summary

- Live ledger rows: `149`.
- Replay decision rows: `729`.
- Live exact replay matches on signal timestamp, symbol, side, and strategy: `0`.
- Live loose replay matches on signal timestamp, symbol, and side: `0`.
- Live rows from `20260525_010004_nopenalty`: `1`.
- Current-run exact replay matches: `0`.

## Live Artifact Mix

Model artifact run ids:

| model_artifact_run_id     |   rows |
|:--------------------------|-------:|
| 20260321_140000           |    112 |
| 20260523_015947           |     36 |
| 20260525_010004_nopenalty |      1 |

Policy artifact run ids:

| policy_artifact_run_id    |   rows |
|:--------------------------|-------:|
| 20260321_140000           |    112 |
| 20260523_015947           |     36 |
| 20260525_010004_nopenalty |      1 |

## Live Gate Distribution

| portfolio_decision   |   rows |
|:---------------------|-------:|
| portfolio_rejected   |     93 |
| rank_rejected        |     22 |
| liquidity_rejected   |     19 |
| traded               |     11 |
| price_gap_rejected   |      4 |

Portfolio reject reasons:

| portfolio_reject_reason                 |   rows |
|:----------------------------------------|-------:|
| rank_below_dynamic_threshold            |     46 |
| invalid_requested_position_size         |     32 |
| NA                                      |     30 |
| below_live_test_min_notional_after_caps |     25 |
| symbol_already_active                   |     12 |
| stale_entry_price_moved_too_far         |      4 |

Liquidity reject reasons:

| liquidity_reject_reason   |   rows |
|:--------------------------|-------:|
| NA                        |    130 |
| spread_above_hard_max     |     15 |
| stale_ticker              |      4 |

## Replay Gate Distribution

| rejection_reason                    |   rows |
|:------------------------------------|-------:|
| below_dynamic_threshold             |    178 |
| symbol_already_open                 |    176 |
| max_new_entries_per_bar_reached     |    121 |
| symbol_in_cooldown                  |     97 |
| accepted                            |     73 |
| max_concurrent_per_strategy_reached |     50 |
| max_concurrent_positions_reached    |     34 |

## Replay Strategy Summary

| side   | strategy_id                                                                                                                                                                                     |   rows |   accepted |   mean_net_return |   net_hit_rate |
|:-------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------:|-----------:|------------------:|---------------:|
| long   | long_asset_vol_level_pct_0_20587213_compression_score_-0_99787366                                                                                                                               |    134 |          9 |       0.013279    |       0.5      |
| long   | long_bars_in_high_vol_state_log_norm_-0_38407263_pullback_depth_1_8353814_pullback_depth_-0_90359813_asset_funding_rate_abs_mean_7d_0_000011711979_variance_ratio_10_48_-0_27143899             |     76 |          5 |       0.00974579  |       0.355263 |
| long   | long_bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828 |    243 |         29 |       0.00993596  |       0.366255 |
| long   | long_loc_bb_channel_pos_48_0_63256526_zscore_price_200_-0_64289832_xasset_asset_minus_basket_fund_z_0_1707595                                                                                   |    122 |          8 |       0.0100447   |       0.508197 |
| short  | short_asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                               |    141 |         21 |       0.0222708   |       0.702128 |
| short  | short_bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385_price_rv_15d_robust_z_0_060036644                                                                                           |     13 |          1 |      -0.000311193 |       0.230769 |

## Current-Run Live Rows

| signal_bar_ts             | symbol     | side   | strategy_id                                                                                                                                              | portfolio_decision   | was_traded   |   normalized_rank_score |   final_threshold |   expected_total_entry_friction_bps |   entry_delay_adverse_bps | replay_exact_match   |
|:--------------------------|:-----------|:-------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------|:-------------|------------------------:|------------------:|------------------------------------:|--------------------------:|:---------------------|
| 2026-05-28 11:00:00+00:00 | AR/USD:USD | short  | loc_ema_stack_pos_24_0_43357179_compression_ratio_-0_33411601_mkt_ret_eq_24h_1_1280091_mkt_ret_eq_24h_-0_81129736_up_down_semivol_ratio_tanh_-0_39156261 | traded               | True         |                0.913143 |              0.78 |                             21.9566 |                   330.969 | False                |

## Interpretation

- The current live ledger is not a clean live-vs-replay parity table for the six-head package because it mixes three artifact generations.
- Only `1` live row references `20260525_010004_nopenalty` in either model or policy artifact fields, and it does not match the frozen replay window/strategy set.
- Replay decisions reject mainly on portfolio state gates (`symbol_already_open`, dynamic thresholds, position/concurrency, cooldown). Live rejects include rank, min-notional sizing, spread, stale ticker, and stale adverse price movement gates that are not represented one-for-one in the portfolio replay artifact.
- Therefore the next valid test is a fresh live-test cycle using the current six-head package with the ledger cleared or namespaced to this run, then replaying the same signal bars with the same initial portfolio state.
