# Meta-Gated Backtest Report

- Symbol: ETHUSDT
- Exchange: binance
- Timeframe: 15m
- Direction: long
- Execution Mode: light
- Events (labeled): 66522
- Trades (gated): 51

## Gating Configuration

- Probability Threshold: 0.550
- Use Expected Return: True
- Expected Return Threshold: 0.0015 (fraction)

## Trade-Level Performance (event-time)

- Mean Return per Trade: 0.1757%
- Std Dev per Trade: 1.1514%
- Trade-Level Sharpe (sqrt(N)): 1.089
- Max Drawdown (event-time equity): -3.48%

## Meta-Gating Diagnostics (from meta-labeling step)

- Gate definition (diagnostics): probability 3 0.75 and expected return 3 0.0015 (fraction, approx. transaction cost)
- AUC (OOF meta-model): 0.760
- Mean return per gated trade: 1.00%
- Sharpe (gated set): 1.86
- Trades gated (diagnostics gate): 6,289
- Approximate average gated trades per day: 4.4  
  (based on 135,775 15m bars 3 1,414 days)
