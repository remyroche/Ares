# Meta-Gated Backtest Report

- Symbol: ETHUSDT
- Exchange: binance
- Timeframe: 15m
- Direction: long
- Execution Mode: full
- Events (labeled, evaluation set): 2491
- Events (labeled, total): 8304
- Trades (gated): 395
- Evaluation period: 2022-07-04 → 2025-11-29 (1245 days)
- Gated trading period: 2022-07-05 → 2025-10-13 (1197 days, ~0.33 trades/day)

## Gating Configuration

- Probability Threshold: 0.650
- Use Expected Return: False

## Trade-Level Performance (event-time)

- Mean Return per Trade: 0.5389%
- Std Dev per Trade: 0.6472%
- Trade-Level Sharpe (sqrt(N)): 16.549
- Max Drawdown (event-time equity): -3.41%
- Hit Rate (gated trades): 78.23%
- Mean Return CI (bootstrap, 95%): [0.4842%, 0.6007%]

## Baseline (Ungated) Event Performance

- Events in evaluation set: 2491
- Mean Return per Event: -0.1178%
- Std Dev per Event: 0.6512%
- Trade-Level Sharpe (sqrt(N)): -9.030
- Max Drawdown (event-time equity): -95.29%
- Hit Rate (events): 34.16%
- Return Quantiles (events): 5%=-0.7288%, 25%=-0.6516%, 50%=-0.5028%, 75%=0.7993%, 95%=0.9389%

## Gated Return Distribution

- Return Quantiles (gated trades): 5%=-0.7159%, 25%=0.7993%, 50%=0.8090%, 75%=0.8998%, 95%=1.0757%

## Temporal Stability (event-time segments)

| Segment | Trades | Mean Return | Sharpe (trade) |
|---------|--------|------------|----------------|
| 1 | 79 | 0.5849% | 8.104 |
| 2 | 79 | 0.5803% | 8.821 |
| 3 | 79 | 0.4156% | 5.138 |
| 4 | 79 | 0.5256% | 6.736 |
| 5 | 79 | 0.5881% | 8.930 |

## Per-Regime Performance (gated trades)

| Regime | Trades | Mean Return | Sharpe (trade) |
|--------|--------|------------|----------------|
| 3.0 | 41 | 0.7788% | 36.477 |
| 2.0 | 44 | 0.7828% | 15.511 |
| 0.0 | 93 | 0.7901% | 20.730 |
| 4.0 | 122 | 0.4370% | 6.561 |
| 1.0 | 81 | 0.1474% | 1.665 |
| -1.0 | 14 | 0.5542% | 2.998 |

## Transaction Cost Stress Test

Multiplier refers to scaling of baseline transaction_cost used in labeling.

| Cost Multiplier | Mean Return | Sharpe (trade) |
|----------------|------------|----------------|
| 1.0 | 0.5389% | 16.549 |
| 2.0 | 0.3889% | 11.943 |
| 3.0 | 0.2389% | 7.336 |

## Meta-Gating Diagnostics (from meta-labeling step)

- These metrics are computed during the meta-labeling step for the diagnostics gate.
- AUC (OOF meta-model): 0.689
- Mean return per gated trade (diagnostics gate): 0.66%
- Sharpe (diagnostics gated set): 1.32
- Trades gated (diagnostics gate): 353
- Approximate average trades per day (diagnostics gate): 0.24
