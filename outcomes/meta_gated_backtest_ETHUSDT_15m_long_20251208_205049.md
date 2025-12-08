# Meta-Gated Backtest Report

- Symbol: ETHUSDT
- Exchange: binance
- Timeframe: 15m
- Direction: long
- Execution Mode: full
- Events (labeled, evaluation set): 13041
- Events (labeled, total): 43470
- Trades (gated): 13041
- Evaluation period: 2024-08-22 → 2025-10-31 (436 days)
- Gated trading period: 2024-08-22 → 2025-10-31 (436 days, ~29.91 trades/day)

## Gating Configuration

- Probability Threshold: 0.500
- Use Expected Return: False

## Trade-Level Performance (event-time)

- Mean Return per Trade: -0.2679%
- Std Dev per Trade: 1.2045%
- Trade-Level Sharpe (sqrt(N)): -25.397
- Max Drawdown (event-time equity): -100.00%
- Hit Rate (gated trades): 33.67%
- Mean Return CI (bootstrap, 95%): [-0.2898%, -0.2498%]

## Baseline (Ungated) Event Performance

- Events in evaluation set: 13041
- Mean Return per Event: -0.2679%
- Std Dev per Event: 1.2045%
- Trade-Level Sharpe (sqrt(N)): -25.397
- Max Drawdown (event-time equity): -100.00%
- Hit Rate (events): 33.67%
- Return Quantiles (events): 5%=-1.2380%, 25%=-1.2380%, 50%=-1.0093%, 75%=0.5984%, 95%=1.9059%

## Gated Return Distribution

- Return Quantiles (gated trades): 5%=-1.2380%, 25%=-1.2380%, 50%=-1.0093%, 75%=0.5984%, 95%=1.9059%

## Temporal Stability (event-time segments)

| Segment | Trades | Mean Return | Sharpe (trade) |
|---------|--------|------------|----------------|
| 1 | 2609 | -0.2797% | -12.181 |
| 2 | 2609 | -0.2672% | -11.219 |
| 3 | 2609 | -0.2324% | -9.563 |
| 4 | 2609 | -0.2742% | -11.527 |
| 5 | 2605 | -0.2859% | -12.413 |

## Per-Regime Performance (gated trades)

| Regime | Trades | Mean Return | Sharpe (trade) |
|--------|--------|------------|----------------|
| -1.0 | 456 | -0.1742% | -2.998 |
| 2.0 | 728 | -0.0810% | -1.518 |
| 4.0 | 3536 | -0.2997% | -14.259 |
| 0.0 | 1324 | -0.2848% | -8.587 |
| 3.0 | 1486 | -0.3010% | -9.802 |
| 1.0 | 3273 | -0.2524% | -12.780 |

## Transaction Cost Stress Test

Multiplier refers to scaling of baseline transaction_cost used in labeling.

| Cost Multiplier | Mean Return | Sharpe (trade) |
|----------------|------------|----------------|
| 1.0 | -0.2679% | -25.397 |
| 2.0 | -0.5679% | -53.839 |
| 3.0 | -0.8679% | -82.280 |

## Meta-Gating Diagnostics (from meta-labeling step)

- These metrics are computed during the meta-labeling step for the diagnostics gate.
- AUC (OOF meta-model): 0.737
- Mean return per gated trade (diagnostics gate): 0.38%
- Sharpe (diagnostics gated set): 10.09
- Trades gated (diagnostics gate): 1094
- Approximate average trades per day (diagnostics gate): 0.75
