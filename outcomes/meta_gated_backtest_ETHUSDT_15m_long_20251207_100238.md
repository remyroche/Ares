# Meta-Gated Backtest Report

- Symbol: ETHUSDT
- Exchange: binance
- Timeframe: 15m
- Direction: long
- Execution Mode: full
- Events (labeled, evaluation set): 13042
- Events (labeled, total): 43472
- Trades (gated): 1046
- Evaluation period: 2024-08-22 → 2025-10-31 (436 days)
- Gated trading period: 2025-03-02 → 2025-10-31 (244 days, ~4.29 trades/day)

## Gating Configuration

- Probability Threshold: 0.550
- Use Expected Return: False

## Trade-Level Performance (event-time)

- Mean Return per Trade: 0.7280%
- Std Dev per Trade: 1.2322%
- Trade-Level Sharpe (sqrt(N)): 19.107
- Max Drawdown (event-time equity): -13.55%
- Hit Rate (gated trades): 71.22%
- Mean Return CI (bootstrap, 95%): [0.6577%, 0.7942%]

## Baseline (Ungated) Event Performance

- Events in evaluation set: 13042
- Mean Return per Event: -0.2759%
- Std Dev per Event: 1.0976%
- Trade-Level Sharpe (sqrt(N)): -28.703
- Max Drawdown (event-time equity): -100.00%
- Hit Rate (events): 33.93%
- Return Quantiles (events): 5%=-1.1706%, 25%=-1.1706%, 50%=-0.9743%, 75%=0.4683%, 95%=1.8400%

## Gated Return Distribution

- Return Quantiles (gated trades): 5%=-1.1706%, 25%=-1.1613%, 50%=1.2742%, 75%=1.8400%, 95%=1.8400%

## Temporal Stability (event-time segments)

| Segment | Trades | Mean Return | Sharpe (trade) |
|---------|--------|------------|----------------|
| 1 | 210 | 0.6848% | 8.076 |
| 2 | 210 | 0.9351% | 11.606 |
| 3 | 210 | 0.5903% | 6.679 |
| 4 | 210 | 0.9874% | 12.816 |
| 5 | 206 | 0.4368% | 4.883 |

## Per-Regime Performance (gated trades)

| Regime | Trades | Mean Return | Sharpe (trade) |
|--------|--------|------------|----------------|
| 1.0 | 126 | 1.1768% | 18.123 |
| 4.0 | 512 | 0.6634% | 11.769 |
| 2.0 | 135 | 0.2798% | 2.213 |
| 3.0 | 138 | 0.9639% | 10.616 |
| 0.0 | 96 | 0.7966% | 7.175 |
| -1.0 | 39 | 0.6735% | 3.129 |

## Transaction Cost Stress Test

Multiplier refers to scaling of baseline transaction_cost used in labeling.

| Cost Multiplier | Mean Return | Sharpe (trade) |
|----------------|------------|----------------|
| 1.0 | 0.7280% | 19.107 |
| 2.0 | 0.4280% | 11.233 |
| 3.0 | 0.1280% | 3.359 |

## Meta-Gating Diagnostics (from meta-labeling step)

- These metrics are computed during the meta-labeling step for the diagnostics gate.
- AUC (OOF meta-model): 0.737
- Mean return per gated trade (diagnostics gate): 0.56%
- Sharpe (diagnostics gated set): 11.19
- Trades gated (diagnostics gate): 671
- Approximate average trades per day (diagnostics gate): 0.46
