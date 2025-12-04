# Meta-Gated Backtest Report

- Symbol: ETHUSDT
- Exchange: binance
- Timeframe: 15m
- Direction: long
- Execution Mode: full
- Events (labeled, evaluation set): 2491
- Events (labeled, total): 8304
- Trades (gated): 240
- Evaluation period: 2022-07-04 → 2025-11-29 (1245 days)
- Gated trading period: 2022-07-05 → 2025-10-13 (1197 days, ~0.20 trades/day)

## Gating Configuration

- Probability Threshold: 0.550
- Use Expected Return: True
- Expected Return Threshold: 0.0045 (fraction)

## Trade-Level Performance (event-time)

- Mean Return per Trade: 0.7382%
- Std Dev per Trade: 0.4492%
- Trade-Level Sharpe (sqrt(N)): 25.457
- Max Drawdown (event-time equity): -1.39%
- Hit Rate (gated trades): 91.25%
- Mean Return CI (bootstrap, 95%): [0.6835%, 0.7852%]

## Baseline (Ungated) Event Performance

- Events in evaluation set: 2491
- Mean Return per Event: -0.1178%
- Std Dev per Event: 0.6512%
- Trade-Level Sharpe (sqrt(N)): -9.030
- Max Drawdown (event-time equity): -95.29%
- Hit Rate (events): 34.16%
- Return Quantiles (events): 5%=-0.7288%, 25%=-0.6516%, 50%=-0.5028%, 75%=0.7993%, 95%=0.9389%

## Gated Return Distribution

- Return Quantiles (gated trades): 5%=-0.6729%, 25%=0.7993%, 50%=0.8396%, 75%=0.9083%, 95%=1.0512%

## Temporal Stability (event-time segments)

| Segment | Trades | Mean Return | Sharpe (trade) |
|---------|--------|------------|----------------|
| 1 | 48 | 0.7547% | 10.350 |
| 2 | 48 | 0.7486% | 13.752 |
| 3 | 48 | 0.6317% | 7.987 |
| 4 | 48 | 0.7725% | 11.795 |
| 5 | 48 | 0.7833% | 16.568 |

## Per-Regime Performance (gated trades)

| Regime | Trades | Mean Return | Sharpe (trade) |
|--------|--------|------------|----------------|
| 3.0 | 28 | 0.7733% | 30.647 |
| 2.0 | 29 | 0.8643% | 71.050 |
| 0.0 | 68 | 0.8772% | 80.934 |
| 4.0 | 74 | 0.7334% | 12.405 |
| -1.0 | 10 | 0.7485% | 4.544 |
| 1.0 | 31 | 0.2913% | 2.108 |

## Transaction Cost Stress Test

Multiplier refers to scaling of baseline transaction_cost used in labeling.

| Cost Multiplier | Mean Return | Sharpe (trade) |
|----------------|------------|----------------|
| 1.0 | 0.7382% | 25.457 |
| 2.0 | 0.5882% | 20.284 |
| 3.0 | 0.4382% | 15.111 |

## Meta-Gating Diagnostics (from meta-labeling step)

- These metrics are computed during the meta-labeling step for the diagnostics gate.
- AUC (OOF meta-model): 0.689
- Mean return per gated trade (diagnostics gate): 0.66%
- Sharpe (diagnostics gated set): 1.32
- Trades gated (diagnostics gate): 353
- Approximate average trades per day (diagnostics gate): 0.24
