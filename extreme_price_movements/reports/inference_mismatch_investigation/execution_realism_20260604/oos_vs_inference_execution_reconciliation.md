# OOS vs Inference Execution Reconciliation

Run: `20260525_010004_nopenalty`

Candidate source: `data_perp/artifacts/20260525_010004_nopenalty/simple_policy_optimiser/simple_policy_candidates.parquet`
Policy params: `data_perp/artifacts/20260525_010004_nopenalty/simple_policy_optimiser/deployment/best_policy_params_perps.json`

## Deployment Thresholds

| strategy_id                                                                                                                                                                                     | strategy_short   |   deployment_rank_threshold |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|----------------------------:|
| long_bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828 | long_dist        |                        0.64 |
| long_dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_leverage_build_score_0_45107844_return_autocorr_48_1_18643_rolling_range_20_-0_25967735                                        | long_dist        |                        0.7  |
| short_asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                               | short_dist       |                        0.59 |
| short_bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385_price_rv_15d_robust_z_0_060036644                                                                                           | short_dist       |                        0.81 |

## Candidate Artifact Entry Delay

```json
{
  "available": true,
  "entry_execution_source_counts": {
    "delayed_1m_intraminute_proxy": 25564,
    "theoretical_15m_open": 1921
  },
  "max_minutes": 10.0,
  "median_minutes": 10.0,
  "min_minutes": 10.0,
  "minutes_value_counts": {
    "10.0": 27485
  },
  "non_null": 27485
}
```

## OOS Candidate Execution Breakdown

- `all_local_candidates`: n=27485, net hit=0.611, gross hit=0.615, mean net=282.2 bps, mean gross=302.5 bps, same-exit price-gap net=195.8 bps, same-exit price-gap effect=86.42 bps, friction=20.30 bps.
- `passes_current_deployment_rank`: n=14692, net hit=0.602, gross hit=0.604, mean net=270.5 bps, mean gross=290.8 bps, same-exit price-gap net=110.2 bps, same-exit price-gap effect=160.32 bps, friction=20.29 bps.

## Proper Delay Sensitivity

| strategy_short   | variant                |   trades |   hit_rate |   mean_return_bps |   gross_hit_rate |   gross_return_bps |   delay_cost_bps_mean |   friction_drag_bps_mean |   delayed_1m_fill_rate |
|:-----------------|:-----------------------|---------:|-----------:|------------------:|-----------------:|-------------------:|----------------------:|-------------------------:|-----------------------:|
| global           | delayed_entry_net      |     3028 |   0.721268 |           255.363 |         0.73745  |            275.638 |             -0.30699  |                  20.2756 |               0.913804 |
| global           | no_delay_same_exit_net |     3028 |   0.720608 |           255.056 |       nan        |            275.331 |              0        |                  20.2756 |               0.913804 |
| long_dist        | delayed_entry_net      |      895 |   0.564246 |           260.478 |         0.602235 |            280.759 |             -0.812872 |                  20.2808 |               1        |
| long_dist        | no_delay_same_exit_net |      895 |   0.564246 |           259.665 |       nan        |            279.946 |              0        |                  20.2808 |               1        |
| long_loc         | delayed_entry_net      |      464 |   0.670259 |           309.385 |         0.672414 |            329.714 |             -0.322109 |                  20.3297 |               0.931034 |
| long_loc         | no_delay_same_exit_net |      464 |   0.668103 |           309.063 |       nan        |            329.392 |              0        |                  20.3297 |               0.931034 |
| short_dist       | delayed_entry_net      |      690 |   0.798551 |           260.532 |         0.807246 |            280.812 |              0.140169 |                  20.2808 |               0.747826 |
| short_dist       | no_delay_same_exit_net |      690 |   0.798551 |           260.672 |       nan        |            280.953 |              0        |                  20.2808 |               0.747826 |
| short_loc        | delayed_entry_net      |      979 |   0.834525 |           221.44  |         0.842697 |            241.681 |             -0.152505 |                  20.2417 |               0.94382  |
| short_loc        | no_delay_same_exit_net |      979 |   0.833504 |           221.287 |       nan        |            241.529 |              0        |                  20.2417 |               0.94382  |

## Per Strategy-ID Candidate Metrics

| strategy_id                                                                                                                                                                                     |    n |   net_hit_rate |   gross_hit_rate |   net_bps_mean |   gross_bps_mean |   friction_drag_bps_mean |   delayed_1m_fill_rate |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----:|---------------:|-----------------:|---------------:|-----------------:|-------------------------:|-----------------------:|
| long_bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828 | 6315 |       0.577672 |         0.581789 |        253.37  |          273.644 |                  20.2736 |               0.983848 |
| long_dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_leverage_build_score_0_45107844_return_autocorr_48_1_18643_rolling_range_20_-0_25967735                                        | 5195 |       0.534937 |         0.5359   |        242.983 |          263.246 |                  20.2632 |               0.678537 |
| short_asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                               | 2747 |       0.768839 |         0.768839 |        372.189 |          392.582 |                  20.3926 |               0.991991 |
| short_bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385_price_rv_15d_robust_z_0_060036644                                                                                           |  435 |       0.708046 |         0.708046 |        206.51  |          226.737 |                  20.2267 |               0.995402 |

## Delay Window and Liquidity Summary

| strategy_id                                                                                                                                                                                     |     n |   delayed_1m_rows |   theoretical_15m_open_rows |   delay_window_candle_count_median |   entry_gap_bps_mean |   entry_slippage_proxy_bps_mean |   delay_max_adverse_bps_mean |   delay_max_favorable_bps_mean |   liquidity_capacity_weight_mean |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------:|------------------:|----------------------------:|-----------------------------------:|---------------------:|--------------------------------:|-----------------------------:|-------------------------------:|---------------------------------:|
| global                                                                                                                                                                                          | 27485 |             25564 |                        1921 |                                 11 |           -0.240094  |                      0.00788483 |                    0.183112  |                      0.376987  |                                1 |
| long_asset_vol_level_pct_0_20587213_compression_score_-0_99787366                                                                                                                               |  4795 |              4746 |                          49 |                                 11 |           -0.142136  |                      0.0104782  |                    0.144856  |                      0.296033  |                                1 |
| long_bars_in_high_vol_state_log_norm_-0_38407263_pullback_depth_1_8353814_pullback_depth_-0_90359813_asset_funding_rate_abs_mean_7d_0_000011711979_variance_ratio_10_48_-0_27143899             |  3494 |              3456 |                          38 |                                 11 |            0.0175807 |                      0          |                    0.0175807 |                      0         |                                1 |
| long_bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828 |  6315 |              6213 |                         102 |                                 11 |           -0.206892  |                      0.00848056 |                    0.0563471 |                      0.304232  |                                1 |
| long_dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_leverage_build_score_0_45107844_return_autocorr_48_1_18643_rolling_range_20_-0_25967735                                        |  5195 |              3525 |                        1670 |                                 11 |            0.0100204 |                      0          |                    0.0129692 |                      0.0129882 |                                1 |
| long_loc_bb_channel_pos_48_0_63256526_zscore_price_200_-0_64289832_xasset_asset_minus_basket_fund_z_0_1707595                                                                                   |  4465 |              4427 |                          38 |                                 11 |           -0.993731  |                      0.0223963  |                    0.704733  |                      1.34876   |                                1 |
| short_asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597                                                                                               |  2747 |              2725 |                          22 |                                 11 |           -0.0202467 |                      0          |                    0         |                      0.0202467 |                                1 |
| short_bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385_price_rv_15d_robust_z_0_060036644                                                                                           |   474 |               472 |                           2 |                                 11 |            0.382555  |                      0          |                    0.883838  |                      0.572485  |                                1 |

## Live Ledger Coverage

```json
{
  "decision_minus_signal_bar_seconds_mean": 4349.2840630229875,
  "decision_minus_signal_bar_seconds_median": 4699.887213,
  "decision_minus_signal_bar_seconds_non_null": 87,
  "decision_minus_signal_bar_seconds_p90": 5984.012470800001,
  "decision_to_entry_seconds_mean": 4.472333,
  "decision_to_entry_seconds_median": 4.472333,
  "decision_to_entry_seconds_non_null": 1,
  "decision_to_entry_seconds_p90": 4.472333,
  "entry_delay_adverse_bps_mean": 30.498500000000206,
  "entry_delay_adverse_bps_median": 30.498500000000206,
  "entry_delay_adverse_bps_non_null": 1,
  "entry_delay_adverse_bps_p90": 30.498500000000206,
  "expected_fill_slippage_bps_mean": 7.521449055468699,
  "expected_fill_slippage_bps_median": 7.521449055468699,
  "expected_fill_slippage_bps_non_null": 1,
  "expected_fill_slippage_bps_p90": 7.521449055468699,
  "expected_total_entry_friction_bps_mean": 8.021474056718706,
  "expected_total_entry_friction_bps_median": 8.021474056718706,
  "expected_total_entry_friction_bps_non_null": 1,
  "expected_total_entry_friction_bps_p90": 8.021474056718706,
  "ledger_exists": true,
  "path": "data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/20260525_010004_nopenalty/prediction_ledger.parquet",
  "portfolio_decisions": {
    "rank_rejected": 86,
    "traded": 1
  },
  "portfolio_reject_reasons": {
    "missing": 1,
    "missing_policy_rank_reference_percentile": 6,
    "rank_below_dynamic_threshold": 80
  },
  "realized_fee_bps_non_null": 0,
  "rows": 87,
  "signal_to_entry_seconds_mean": 5102.368352,
  "signal_to_entry_seconds_median": 5102.368352,
  "signal_to_entry_seconds_non_null": 1,
  "signal_to_entry_seconds_p90": 5102.368352,
  "slippage_bps_mean": 7.521449055468699,
  "slippage_bps_median": 7.521449055468699,
  "slippage_bps_non_null": 1,
  "slippage_bps_p90": 7.521449055468699,
  "spread_bps_mean": 1.000050002500015,
  "spread_bps_median": 1.000050002500015,
  "spread_bps_non_null": 1,
  "spread_bps_p90": 1.000050002500015,
  "ticker_spread_bps_mean": 1.000050002500015,
  "ticker_spread_bps_median": 1.000050002500015,
  "ticker_spread_bps_non_null": 1,
  "ticker_spread_bps_p90": 1.000050002500015,
  "traded_rows": 1
}
```

## Live Market/Stop Order Contract

- Live entries refuse to place an unprotected order unless exact simple-policy stop params and barrier context are loaded for the strategy.
- In live mode, the entry order may be forced to `market`; the execution path extracts the realized exchange fill and stores it as `entry_price`/`realized_entry_price`.
- Theoretical/policy/ohlcv entry prices are retained separately as audit fields (`theoretical_entry_price`, `policy_entry_price`, `ohlcv_entry_price`) and used to compute `entry_delay_adverse_bps` and entry-price deltas.
- Initial STOP_LOSS and trailing/replace decisions in `simple_policy_stop.py` use the live position state's `entry_price`, which is the realized fill for live entries. This avoids stops that are accidentally too close to a worse live fill, but means optimiser replay assumptions must be tuned to match live fill distributions.
- Position monitoring can classify rejected protective stops through `trigger_price_rejected` or `order_rejected`; the prediction ledger only contains portfolio-level rejection reasons, so exchange-level rejection counts still require trade-executor/order logs.

## Findings

- OOS simple-policy candidates contain delayed-entry gross/net returns, theoretical entry, delayed entry, entry-gap, expected friction, fee, slippage, and orderbook-slippage fields.
- The candidate artifact delay summary above is measured directly from `delayed_entry_ts - timestamp`; if it differs from the current code default, the artifact must be regenerated before treating its policy metrics as current-code evidence.
- The proper no-delay-vs-delayed comparison is sourced from `execution_attribution/global_summary.csv` and `execution_attribution/per_strategy.csv`. The candidate-table same-exit price-gap columns are diagnostic only and must not be interpreted as a valid no-delay policy replay.
- The final candidate parquet contains the t+10 delay-window fields (`delay_close_gap_bps`, `delay_max_adverse_bps`, `delay_max_favorable_bps`, `delay_window_range_bps`, and `delay_window_candle_count`); the delay-window summary above is computed directly from those fields.
- Live ledger currently has sparse realized entry timing: `signal_to_entry_seconds` and `decision_to_entry_seconds` are mostly absent for untraded rows. That is acceptable for rejected candidates but means execution-delay realism must be evaluated on traded rows plus trade logs, not solely the prediction ledger.
- Rejected order analysis is represented through portfolio/liquidity rejection reasons. Exchange-level rejected market/stop order counts require trade-executor logs or exchange order history, which are not fully represented in `prediction_ledger.parquet`.
