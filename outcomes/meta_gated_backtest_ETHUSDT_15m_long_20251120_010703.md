# Meta-Gated Backtest Report

- Symbol: ETHUSDT
- Exchange: binance
- Timeframe: 15m
- Direction: long
- Execution Mode: full
- Events (labeled): 44589
- Trades (gated): 3570

## Gating Configuration

- Probability Threshold: 0.750
- Use Expected Return: True
- Expected Return Threshold: 0.0015 (fraction)

## Trade-Level Performance (event-time)

- Mean Return per Trade: 1.2127%
- Std Dev per Trade: 0.4099%
- Trade-Level Sharpe (sqrt(N)): 176.777
- Max Drawdown (event-time equity): -8.01%

## Meta-Gating Diagnostics (from meta-labeling step)

- These metrics are computed during the meta-labeling step for the diagnostics gate.
- AUC (OOF meta-model): 0.731
- Mean return per gated trade (diagnostics gate): 1.21%
- Sharpe (diagnostics gated set): 2.92
- Trades gated (diagnostics gate): 3589
- Approximate average trades per day (diagnostics gate): 2.45
