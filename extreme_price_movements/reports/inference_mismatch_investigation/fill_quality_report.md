# OOS vs Inference Execution Reconciliation

Run: `20260525_010004_nopenalty`

Candidate source: `data_perp/exchanges/krakenfutures/artifacts/20260525_010004_nopenalty/simple_policy_optimiser/simple_policy_candidates.parquet`
Policy params: `data_perp/exchanges/krakenfutures/artifacts/20260525_010004_nopenalty/simple_policy_optimiser/deployment/best_policy_params_perps.json`

## Deployment Thresholds

| strategy_id                                                                                                                                                           | strategy_short   |   deployment_rank_threshold |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|----------------------------:|
| long_dist_weekly_vwap_0_074823022_loc_prev_week_range_pos_48_0_48354843_mkt_ret_eq_24h_-0_43956268_volume_autocorr_48_-0_38378653                                     | long_dist        |                        0.88 |
| long_loc_prev_week_range_pos_48_0_42586401_loc_vwap_dev_z_24_0_10701825_zscore_price_50_1_0128103_mkt_ret_eq_24h_-0_78752208_up_down_return_mass_ratio_tanh_1_1231147 | long_loc         |                        0.85 |
| short_dist_rolling_7d_high_0_13977644_mkt_ret_eq_24h_-0_56630391_rolling_range_20_-0_40672407                                                                         | short_dist       |                        0.59 |
| short_loc_ema_stack_pos_24_0_43357179_compression_ratio_-0_33411601_mkt_ret_eq_24h_1_1280091_mkt_ret_eq_24h_-0_81129736_up_down_semivol_ratio_tanh_-0_39156261        | short_loc        |                        0.78 |

## Candidate Artifact Entry Delay

```json
{
  "available": true,
  "entry_execution_source_counts": {
    "delayed_1m_intraminute_proxy": 18534,
    "theoretical_15m_open": 1492
  },
  "max_minutes": 7.0,
  "median_minutes": 7.0,
  "min_minutes": 7.0,
  "minutes_value_counts": {
    "7.0": 20026
  },
  "non_null": 20026
}
```

## OOS Candidate Execution Breakdown

- `all_local_candidates`: n=20026, net hit=0.612, gross hit=0.635, mean net=200.8 bps, mean gross=221.0 bps, same-exit price-gap net=11.2 bps, same-exit price-gap effect=189.62 bps, friction=20.22 bps.
- `passes_current_deployment_rank`: n=16707, net hit=0.651, gross hit=0.670, mean net=209.5 bps, mean gross=229.7 bps, same-exit price-gap net=-18.4 bps, same-exit price-gap effect=227.87 bps, friction=20.23 bps.

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

## Live Ledger Coverage

```json
{
  "decision_minus_signal_bar_seconds_mean": 3834.8637891610733,
  "decision_minus_signal_bar_seconds_median": 3894.565717,
  "decision_minus_signal_bar_seconds_non_null": 149,
  "decision_minus_signal_bar_seconds_p90": 4889.144609800004,
  "decision_to_entry_seconds_non_null": 0,
  "entry_delay_adverse_bps_mean": 330.9693107380918,
  "entry_delay_adverse_bps_median": 330.9693107380918,
  "entry_delay_adverse_bps_non_null": 1,
  "entry_delay_adverse_bps_p90": 330.9693107380918,
  "expected_fill_slippage_bps_mean": 0.0,
  "expected_fill_slippage_bps_median": 0.0,
  "expected_fill_slippage_bps_non_null": 18,
  "expected_fill_slippage_bps_p90": 0.0,
  "expected_total_entry_friction_bps_mean": 25.550657757510578,
  "expected_total_entry_friction_bps_median": 23.328276253229518,
  "expected_total_entry_friction_bps_non_null": 18,
  "expected_total_entry_friction_bps_p90": 47.17693998034903,
  "ledger_exists": true,
  "path": "data_perp/exchanges/krakenfutures/live_state/prediction_ledger.parquet",
  "portfolio_decisions": {
    "liquidity_rejected": 19,
    "portfolio_rejected": 93,
    "price_gap_rejected": 4,
    "rank_rejected": 22,
    "traded": 11
  },
  "portfolio_reject_reasons": {
    "below_live_test_min_notional_after_caps": 25,
    "invalid_requested_position_size": 32,
    "missing": 30,
    "rank_below_dynamic_threshold": 46,
    "stale_entry_price_moved_too_far": 4,
    "symbol_already_active": 12
  },
  "realized_fee_bps_non_null": 0,
  "rows": 149,
  "signal_to_entry_seconds_non_null": 0,
  "slippage_bps_mean": 0.0,
  "slippage_bps_median": 0.0,
  "slippage_bps_non_null": 1,
  "slippage_bps_p90": 0.0,
  "spread_bps_mean": 2305.357124067476,
  "spread_bps_median": 94.67646375661843,
  "spread_bps_non_null": 40,
  "spread_bps_p90": 11101.89266269121,
  "ticker_spread_bps_mean": 43.91314954866991,
  "ticker_spread_bps_median": 43.91314954866991,
  "ticker_spread_bps_non_null": 1,
  "ticker_spread_bps_p90": 43.91314954866991,
  "traded_rows": 11
}
```

## Findings

- OOS simple-policy candidates contain delayed-entry gross/net returns, theoretical entry, delayed entry, entry-gap, expected friction, fee, slippage, and orderbook-slippage fields.
- The candidate artifact delay summary above is measured directly from `delayed_entry_ts - timestamp`; if it differs from the current code default, the artifact must be regenerated before treating its policy metrics as current-code evidence.
- The proper no-delay-vs-delayed comparison is sourced from `execution_attribution/global_summary.csv` and `execution_attribution/per_strategy.csv`. The candidate-table same-exit price-gap columns are diagnostic only and must not be interpreted as a valid no-delay policy replay.
- The final candidate parquet does not contain the full delay-window breakdown columns (`delay_close_gap_bps`, `delay_max_adverse_bps`, `delay_max_favorable_bps`), so this audit cannot yet reproduce the full within-window path decomposition from that parquet alone.
- Live ledger currently has sparse realized entry timing: `signal_to_entry_seconds` and `decision_to_entry_seconds` are mostly absent for untraded rows. That is acceptable for rejected candidates but means execution-delay realism must be evaluated on traded rows plus trade logs, not solely the prediction ledger.
- Rejected order analysis is represented through portfolio/liquidity rejection reasons. Exchange-level rejected market/stop order counts require trade-executor logs or exchange order history, which are not fully represented in `prediction_ledger.parquet`.
- Quote/orderbook snapshots are not available for fitting spread/slippage curves. The supported fitting source is the live observation set around signal, order, and fill time, plus persisted fee and entry-gap diagnostics.
- Intra-delay path modelling is intentionally skipped in this pass because it requires denser path data and another model. The policy replay remains based on the configured delayed-entry proxy minute.
- Stop/trailing path realism is intentionally limited to the fields currently logged; full stop/trailing reconstruction requires hours of post-entry data and is out of scope for the current pass.
- Feature-only final-fit score dumps are diagnostic only. Executable policy optimisation must use rows with label/forward-path columns; otherwise delayed entry can be filled but exits cannot be scored.
