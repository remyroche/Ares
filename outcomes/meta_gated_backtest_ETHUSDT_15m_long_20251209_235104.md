# Meta-Gated Backtest Report

- Symbol: ETHUSDT
- Exchange: binance
- Timeframe: 15m
- Direction: long
- Execution Mode: full
- Events (labeled, evaluation set): 13041
- Events (labeled, total): 43470
- Trades (gated): 1118
- Evaluation period: 2024-08-22 → 2025-10-31 (436 days)
- Gated trading period: 2025-03-02 → 2025-10-31 (244 days, ~4.58 trades/day)

## Gating Configuration

- Probability Threshold: 0.550
- Use Expected Return: False

## Trade-Level Performance (event-time)

- Mean Return per Trade: 0.7221%
- Std Dev per Trade: 1.3182%
- Trade-Level Sharpe (sqrt(N)): 18.316
- Max Drawdown (event-time equity): -11.69%
- Hit Rate (gated trades): 71.11%
- Mean Return CI (bootstrap, 95%): [0.6420%, 0.7935%]

## Baseline (Ungated) Event Performance

- Events in evaluation set: 13041
- Mean Return per Event: -0.2679%
- Std Dev per Event: 1.2045%
- Trade-Level Sharpe (sqrt(N)): -25.397
- Max Drawdown (event-time equity): -100.00%
- Hit Rate (events): 33.67%
- Return Quantiles (events): 5%=-1.2380%, 25%=-1.2380%, 50%=-1.0093%, 75%=0.5984%, 95%=1.9059%

## Gated Return Distribution

- Return Quantiles (gated trades): 5%=-1.2380%, 25%=-1.2365%, 50%=1.1521%, 75%=1.9059%, 95%=1.9059%

## Temporal Stability (event-time segments)

| Segment | Trades | Mean Return | Sharpe (trade) |
|---------|--------|------------|----------------|
| 1 | 224 | 1.0223% | 12.825 |
| 2 | 224 | 0.8710% | 9.804 |
| 3 | 224 | 0.4128% | 4.623 |
| 4 | 224 | 0.6887% | 7.841 |
| 5 | 222 | 0.6145% | 6.839 |

## Per-Regime Performance (gated trades)

| Regime | Trades | Mean Return | Sharpe (trade) |
|--------|--------|------------|----------------|
| 1.0 | 132 | 1.3054% | 24.443 |
| 4.0 | 548 | 0.5549% | 9.301 |
| 2.0 | 152 | 0.2398% | 1.955 |
| 3.0 | 135 | 1.2011% | 13.634 |
| 0.0 | 102 | 0.9352% | 8.418 |
| -1.0 | 49 | 0.7524% | 4.069 |

## Transaction Cost Stress Test

Multiplier refers to scaling of baseline transaction_cost used in labeling.

| Cost Multiplier | Mean Return | Sharpe (trade) |
|----------------|------------|----------------|
| 1.0 | 0.7221% | 18.316 |
| 2.0 | 0.4221% | 10.706 |
| 3.0 | 0.1221% | 3.096 |

## Meta-Gating Diagnostics (from meta-labeling step)

- These metrics are computed during the meta-labeling step for the diagnostics gate.
- AUC (OOF meta-model): 0.737
- Mean return per gated trade (diagnostics gate): 0.38%
- Sharpe (diagnostics gated set): 10.09
- Trades gated (diagnostics gate): 1094
- Approximate average trades per day (diagnostics gate): 0.75
