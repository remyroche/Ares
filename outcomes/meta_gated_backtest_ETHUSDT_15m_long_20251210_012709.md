# Meta-Gated Backtest Report

- Symbol: ETHUSDT
- Exchange: binance
- Timeframe: 15m
- Direction: long
- Execution Mode: full
- Events (labeled, evaluation set): 13041
- Events (labeled, total): 43471
- Trades (gated): 3335
- Evaluation period: 2024-08-22 → 2025-10-31 (436 days)
- Gated trading period: 2024-08-24 → 2025-10-31 (434 days, ~7.68 trades/day)

## Gating Configuration

- Probability Threshold: 0.600
- Use Expected Return: False

## Trade-Level Performance (event-time)

- Mean Return per Trade: -0.3346%
- Std Dev per Trade: 1.0652%
- Trade-Level Sharpe (sqrt(N)): -18.143
- Max Drawdown (event-time equity): -100.00%
- Hit Rate (gated trades): 27.71%
- Mean Return CI (bootstrap, 95%): [-0.3653%, -0.2990%]

## Baseline (Ungated) Event Performance

- Events in evaluation set: 13041
- Mean Return per Event: -0.2737%
- Std Dev per Event: 1.0331%
- Trade-Level Sharpe (sqrt(N)): -30.256
- Max Drawdown (event-time equity): -100.00%
- Hit Rate (events): 30.81%
- Return Quantiles (events): 5%=-1.0122%, 25%=-1.0122%, 50%=-0.9999%, 75%=0.3631%, 95%=1.7662%

## Gated Return Distribution

- Return Quantiles (gated trades): 5%=-1.0122%, 25%=-1.0122%, 50%=-1.0122%, 75%=0.5379%, 95%=1.7662%

## Temporal Stability (event-time segments)

| Segment | Trades | Mean Return | Sharpe (trade) |
|---------|--------|------------|----------------|
| 1 | 667 | -0.2700% | -6.330 |
| 2 | 667 | -0.2424% | -5.637 |
| 3 | 667 | -0.3609% | -9.046 |
| 4 | 667 | -0.3956% | -9.547 |
| 5 | 667 | -0.4043% | -10.424 |

## Per-Regime Performance (gated trades)

| Regime | Trades | Mean Return | Sharpe (trade) |
|--------|--------|------------|----------------|
| 4.0 | 1374 | -0.3475% | -11.939 |
| 0.0 | 416 | -0.4534% | -9.434 |
| 3.0 | 474 | -0.3474% | -7.233 |
| 1.0 | 593 | -0.3036% | -7.297 |
| 2.0 | 189 | -0.0738% | -0.795 |
| -1.0 | 146 | -0.3519% | -3.964 |

## Transaction Cost Stress Test

Multiplier refers to scaling of baseline transaction_cost used in labeling.

| Cost Multiplier | Mean Return | Sharpe (trade) |
|----------------|------------|----------------|
| 1.0 | -0.3346% | -18.143 |
| 2.0 | -0.6346% | -34.408 |
| 3.0 | -0.9346% | -50.674 |

## Meta-Gating Diagnostics (from meta-labeling step)

- These metrics are computed during the meta-labeling step for the diagnostics gate.
- AUC (OOF meta-model): 0.737
- Mean return per gated trade (diagnostics gate): 0.51%
- Sharpe (diagnostics gated set): 9.66
- Trades gated (diagnostics gate): 675
- Approximate average trades per day (diagnostics gate): 0.46
