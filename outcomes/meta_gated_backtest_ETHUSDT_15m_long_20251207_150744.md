# Meta-Gated Backtest Report

- Symbol: ETHUSDT
- Exchange: binance
- Timeframe: 15m
- Direction: long
- Execution Mode: full
- Events (labeled, evaluation set): 13041
- Events (labeled, total): 43471
- Trades (gated): 1701
- Evaluation period: 2024-08-22 → 2025-10-31 (436 days)
- Gated trading period: 2025-03-02 → 2025-10-31 (244 days, ~6.97 trades/day)

## Gating Configuration

- Probability Threshold: 0.550
- Use Expected Return: False

## Trade-Level Performance (event-time)

- Mean Return per Trade: 0.4369%
- Std Dev per Trade: 1.2595%
- Trade-Level Sharpe (sqrt(N)): 14.308
- Max Drawdown (event-time equity): -35.90%
- Hit Rate (gated trades): 61.08%
- Mean Return CI (bootstrap, 95%): [0.3814%, 0.5022%]

## Baseline (Ungated) Event Performance

- Events in evaluation set: 13041
- Mean Return per Event: -0.2751%
- Std Dev per Event: 1.0801%
- Trade-Level Sharpe (sqrt(N)): -29.085
- Max Drawdown (event-time equity): -100.00%
- Hit Rate (events): 33.01%
- Return Quantiles (events): 5%=-1.1102%, 25%=-1.1102%, 50%=-1.0487%, 75%=0.5035%, 95%=1.7514%

## Gated Return Distribution

- Return Quantiles (gated trades): 5%=-1.1102%, 25%=-1.1102%, 50%=0.8828%, 75%=1.7514%, 95%=1.7514%

## Temporal Stability (event-time segments)

| Segment | Trades | Mean Return | Sharpe (trade) |
|---------|--------|------------|----------------|
| 1 | 341 | 0.4789% | 7.169 |
| 2 | 341 | 0.6592% | 10.040 |
| 3 | 341 | 0.4682% | 6.932 |
| 4 | 341 | 0.4198% | 5.994 |
| 5 | 337 | 0.1554% | 2.255 |

## Per-Regime Performance (gated trades)

| Regime | Trades | Mean Return | Sharpe (trade) |
|--------|--------|------------|----------------|
| 1.0 | 186 | 0.9848% | 16.469 |
| 4.0 | 893 | 0.2434% | 5.573 |
| 2.0 | 209 | 0.3213% | 3.377 |
| 3.0 | 199 | 0.8757% | 11.855 |
| 0.0 | 148 | 0.5347% | 5.567 |
| -1.0 | 66 | 0.3353% | 2.072 |

## Transaction Cost Stress Test

Multiplier refers to scaling of baseline transaction_cost used in labeling.

| Cost Multiplier | Mean Return | Sharpe (trade) |
|----------------|------------|----------------|
| 1.0 | 0.4369% | 14.308 |
| 2.0 | 0.1369% | 4.484 |
| 3.0 | -0.1631% | -5.339 |

## Meta-Gating Diagnostics (from meta-labeling step)

- These metrics are computed during the meta-labeling step for the diagnostics gate.
- AUC (OOF meta-model): 0.737
- Mean return per gated trade (diagnostics gate): 0.38%
- Sharpe (diagnostics gated set): 10.09
- Trades gated (diagnostics gate): 1094
- Approximate average trades per day (diagnostics gate): 0.75
