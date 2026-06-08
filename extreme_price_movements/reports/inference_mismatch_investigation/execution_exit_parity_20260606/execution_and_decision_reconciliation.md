# Execution and Decision Reconciliation

## Spread / Slippage
- Rows: `255`
- Traded rows: `12`
- Policy vs live friction delta: `{'n': 14, 'mean': 0.0, 'median': 0.0, 'p90': 0.0, 'max': 0.0}`
- Live total entry friction: `{'n': 14, 'mean': 25.134291639172158, 'median': 12.245627719811168, 'p90': 56.99311918145675, 'max': 83.63692339268339}`

## Backtest / Live Open Decision
- Ledger rows: `255`
- Live traded: `12`
- Replay accepted: `5`
- Decision mismatches: `17`
- Gap classes: `{'match': 238, 'live_accept_replay_reject': 12, 'replay_accept_live_reject': 5}`
- Gap explanations: `{'rank_threshold': 235, 'live_traded': 12, 'live_stale_signal_or_data_gate': 2, 'live_reject:global_auction_symbol_entry_block:recent_losing_trade_cooldown': 2, 'live_reject:local_stop_min_distance_invalid': 2, 'live_reject:global_auction_adverse_hourly_close_gap:adverse_hourly_close_gap_too_large': 1, 'live_reject:global_auction_symbol_entry_block:symbol_already_active': 1}`
- Direct rank-gate would open: `25`
- Direct rank-gate mismatches: `13`
- Direct rank-gate gap explanations: `{'match': 242, 'rank_threshold': 5, 'live_stale_signal_or_data_gate': 2, 'live_reject:global_auction_symbol_entry_block:recent_losing_trade_cooldown': 2, 'live_reject:local_stop_min_distance_invalid': 2, 'live_reject:global_auction_adverse_hourly_close_gap:adverse_hourly_close_gap_too_large': 1, 'live_reject:global_auction_symbol_entry_block:symbol_already_active': 1}`
- Exact portfolio-state replayable rows: `0`
- Exact portfolio-state replayable traded rows: `0`

## Replay Field Coverage
- Ledger rows: `447`
- Live traded rows: `12`
- Exact portfolio-state replayable rows: `0`
- Exact portfolio-state replayable rate: `0.0`
- Failed field checks: `14`
- Failed traded-field checks: `6`
- Critical missing rows: `2727`
- Worst missing: `[{'field_group': 'exact_portfolio_state_replay', 'scope': 'all', 'accepted_alternatives': 'portfolio_state_snapshot_json|open_positions_before_json|active_positions_before_json', 'missing_rows': 447, 'coverage_rate': 0.0}, {'field_group': 'exact_portfolio_state_replay', 'scope': 'all', 'accepted_alternatives': 'portfolio_state_snapshot_hash|portfolio_state_hash', 'missing_rows': 447, 'coverage_rate': 0.0}, {'field_group': 'exact_portfolio_state_replay', 'scope': 'all', 'accepted_alternatives': 'open_positions_before|open_positions_before_count', 'missing_rows': 447, 'coverage_rate': 0.0}, {'field_group': 'exact_portfolio_state_replay', 'scope': 'all', 'accepted_alternatives': 'cooldowns_before_json|recent_losing_trade_cooldown_state_json', 'missing_rows': 447, 'coverage_rate': 0.0}, {'field_group': 'exact_portfolio_state_replay', 'scope': 'all', 'accepted_alternatives': 'portfolio_priority', 'missing_rows': 447, 'coverage_rate': 0.0}, {'field_group': 'exact_portfolio_state_replay', 'scope': 'all', 'accepted_alternatives': 'wallet_before|wallet_value', 'missing_rows': 440, 'coverage_rate': 0.015659955257270694}, {'field_group': 'spread_slippage_cost_attribution', 'scope': 'traded', 'accepted_alternatives': 'fee_bps|entry_fee_bps|realized_fee_bps', 'missing_rows': 12, 'coverage_rate': 0.0}, {'field_group': 'order_fill_identity', 'scope': 'traded', 'accepted_alternatives': 'position_id', 'missing_rows': 12, 'coverage_rate': 0.0}, {'field_group': 'entry_timing_attribution', 'scope': 'traded', 'accepted_alternatives': 'signal_bar_close_ts', 'missing_rows': 7, 'coverage_rate': 0.4166666666666667}, {'field_group': 'entry_timing_attribution', 'scope': 'traded', 'accepted_alternatives': 'decision_price_to_fill_bps|actual_fill_vs_expected_bps', 'missing_rows': 7, 'coverage_rate': 0.4166666666666667}]`

Note: decision replay uses live ledger candidates and deployed portfolio-policy gates. It is a final gate parity audit, not a PnL backtest.
