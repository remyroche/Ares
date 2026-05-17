# Live vs OOS Gap Report

## Required interpretations
1. If signal_forward_return is good but fill_forward_return is bad: execution/timing gap.
2. If signal_forward_return is already bad: model/rank/live-feature drift.
3. If signal_forward_return is good for rejected candidates but not traded candidates: selection/gating/portfolio constraints issue.
4. If fill_forward_return is good but realized_trade_return is bad: exit/stop/slippage/cost issue.

## Summary
- **rows**: 34
- **traded_rows**: 16
- **rejected_rows**: 18
- **mean_oos_expected_net_bps**: 48.91588982632931
- **mean_signal_forward_net_bps**: -34.34653425537004
- **mean_realized_trade_net_bps**: -187.5592199963025

## Metadata
- **primary_horizon_bars**: [24]
- **bar_minutes**: [60]

## Diagnostic coverage
- **replay_rows**: 34
- **rows_with_oos_join**: 34
- **rows_with_signal_forward**: 14
- **rows_with_fill_forward**: 17
- **rows_with_realized_exits**: 2
- **rows_with_realized_trade_net**: 2
- **diagnostic_complete_rows**: 14
- **missing_forward_outcome_rows**: 20
- **unresolved_trade_rows**: 6
- **ledger_decision_ts_non_null**: 34
- **ledger_signal_bar_ts_non_null**: 34
- **ledger_feature_source_max_ts_non_null**: 0
- **ledger_feature_available_ts_non_null**: 0

## Unit warnings

## Feature parity
- **rows**: 0
- **matches**: 0
- **mismatches**: 0
- **missing**: 0
- **lookahead**: 0

## Selection summary
- **mean_traded_signal_forward_bps**: -24.49290607206922
- **mean_rejected_signal_forward_bps**: -162.44370063828063
- **rejected_positive_signal_count**: 0
- **rejected_positive_signal_sum_bps**: 0.0
- **selection_opportunity_cost_bps**: -137.9507945662114

## Four-element diagnosis
- **signal_forward_good_fill_forward_bad**: {'rows': 0, 'mean_oos_expected_net_bps': nan, 'mean_signal_forward_net_bps': nan, 'mean_fill_forward_net_bps': nan, 'mean_realized_trade_net_bps': nan, 'mean_gap_oos_vs_realized_bps': nan}
- **signal_forward_bad**: {'rows': 7, 'mean_oos_expected_net_bps': 48.51938380549352, 'mean_signal_forward_net_bps': -304.4806247969949, 'mean_fill_forward_net_bps': -306.2572275866449, 'mean_realized_trade_net_bps': -209.517045454546, 'mean_gap_oos_vs_realized_bps': 255.87943650669274}
- **rejected_candidate_signal_forward_good**: {'rows': 0, 'mean_oos_expected_net_bps': nan, 'mean_signal_forward_net_bps': nan, 'mean_fill_forward_net_bps': nan, 'mean_realized_trade_net_bps': nan, 'mean_gap_oos_vs_realized_bps': nan}
- **fill_forward_good_realized_trade_bad**: {'rows': 1, 'mean_oos_expected_net_bps': 50.13712837050359, 'mean_signal_forward_net_bps': 389.0227656670353, 'mean_fill_forward_net_bps': 379.9759223117816, 'mean_realized_trade_net_bps': -165.601394538059, 'mean_gap_oos_vs_realized_bps': 215.73852290856257}

## IC metrics
- **overall_ic**: 0.389010989010989
- **ic_mean_across_symbols**: nan
- **ic_std_across_symbols**: nan
- **ic_n_symbols**: 0
- **ic_mean_across_weeks**: 0.65
- **ic_std_across_weeks**: 0.35
- **ic_n_weeks**: 2
- **ic_mean_across_months**: 0.389010989010989
- **ic_std_across_months**: 0.0
- **ic_n_months**: 1

## Classification counts
- **missing_forward_outcome**: 20
- **prediction_or_live_feature_drift**: 7
- **unresolved_trade**: 6
- **exit_stop_slippage_cost_gap**: 1

## Recommended next action
Prioritize model/rank drift and live feature parity investigation.
