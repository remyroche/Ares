# Execution and Decision Reconciliation

## Spread / Slippage
- Rows: `53`
- Traded rows: `7`
- Policy vs live friction delta: `{'n': 7, 'mean': 0.0, 'median': 0.0, 'p90': 0.0, 'max': 0.0}`
- Live total entry friction: `{'n': 7, 'mean': 34.88046401286212, 'median': 32.01062994503852, 'p90': 69.61386939725024, 'max': 83.63692339268339}`

## Backtest / Live Open Decision
- Ledger rows: `53`
- Live traded: `7`
- Replay accepted: `2`
- Decision mismatches: `9`
- Gap classes: `{'match': 44, 'live_accept_replay_reject': 7, 'replay_accept_live_reject': 2}`
- Gap explanations: `{'rank_threshold': 44, 'live_traded': 7, 'live_stale_signal_or_data_gate': 2}`
- Direct rank-gate would open: `10`
- Direct rank-gate mismatches: `3`
- Direct rank-gate gap explanations: `{'match': 50, 'live_stale_signal_or_data_gate': 2, 'rank_threshold': 1}`

Note: decision replay uses live ledger candidates and deployed portfolio-policy gates. It is a final gate parity audit, not a PnL backtest.
