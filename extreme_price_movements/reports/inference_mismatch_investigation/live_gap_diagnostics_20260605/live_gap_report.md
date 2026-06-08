# Live vs OOS Gap Report

## Required interpretations
1. If signal_forward_return is good but fill_forward_return is bad: execution/timing gap.
2. If signal_forward_return is already bad: model/rank/live-feature drift.
3. If signal_forward_return is good for rejected candidates but not traded candidates: selection/gating/portfolio constraints issue.
4. If fill_forward_return is good but realized_trade_return is bad: exit/stop/slippage/cost issue.

## Summary
- **rows**: 106
- **traded_rows**: 7
- **rejected_rows**: 99
- **mean_oos_expected_net_bps**: 41.992391068664084
- **mean_signal_forward_net_bps**: nan
- **mean_realized_trade_net_bps**: -337.45217206014536

## Metadata
- **primary_horizon_bars**: []
- **bar_minutes**: []

## Diagnostic coverage
- **replay_rows**: 106
- **rows_with_oos_join**: 106
- **rows_with_signal_forward**: 0
- **rows_with_fill_forward**: 0
- **rows_with_realized_exits**: 2
- **rows_with_realized_trade_net**: 2
- **diagnostic_complete_rows**: 0
- **missing_forward_outcome_rows**: 106
- **unresolved_trade_rows**: 0
- **ledger_decision_ts_non_null**: 106
- **ledger_signal_bar_ts_non_null**: 106
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
- **mean_traded_signal_forward_bps**: nan
- **mean_rejected_signal_forward_bps**: nan
- **rejected_positive_signal_count**: 0
- **rejected_positive_signal_sum_bps**: 0.0
- **selection_opportunity_cost_bps**: nan

## Four-element diagnosis
- **signal_forward_good_fill_forward_bad**: {'rows': 0, 'mean_oos_expected_net_bps': nan, 'mean_signal_forward_net_bps': nan, 'mean_fill_forward_net_bps': nan, 'mean_realized_trade_net_bps': nan, 'mean_gap_oos_vs_realized_bps': nan}
- **signal_forward_bad**: {'rows': 0, 'mean_oos_expected_net_bps': nan, 'mean_signal_forward_net_bps': nan, 'mean_fill_forward_net_bps': nan, 'mean_realized_trade_net_bps': nan, 'mean_gap_oos_vs_realized_bps': nan}
- **rejected_candidate_signal_forward_good**: {'rows': 0, 'mean_oos_expected_net_bps': nan, 'mean_signal_forward_net_bps': nan, 'mean_fill_forward_net_bps': nan, 'mean_realized_trade_net_bps': nan, 'mean_gap_oos_vs_realized_bps': nan}
- **fill_forward_good_realized_trade_bad**: {'rows': 0, 'mean_oos_expected_net_bps': nan, 'mean_signal_forward_net_bps': nan, 'mean_fill_forward_net_bps': nan, 'mean_realized_trade_net_bps': nan, 'mean_gap_oos_vs_realized_bps': nan}

## IC metrics
- **overall_ic**: nan
- **ic_mean_across_symbols**: nan
- **ic_std_across_symbols**: nan
- **ic_n_symbols**: 0
- **ic_mean_across_weeks**: nan
- **ic_std_across_weeks**: nan
- **ic_n_weeks**: 0
- **ic_mean_across_months**: nan
- **ic_std_across_months**: nan
- **ic_n_months**: 0

## Classification counts
- **missing_forward_outcome**: 106

## Recommended next action
Collect more replay rows with forward and realized outcomes before drawing conclusions.
