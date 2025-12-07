# Meta-Gated Backtest Report

- Symbol: ETHUSDT
- Exchange: binance
- Timeframe: 15m
- Direction: long
- Execution Mode: full
- Events (labeled, evaluation set): 13042
- Events (labeled, total): 43472
- Trades (gated): 7294
- Evaluation period: 2024-08-22 → 2025-10-31 (436 days)
- Gated trading period: 2024-08-23 → 2025-10-31 (435 days, ~16.77 trades/day)

## Gating Configuration

- Probability Threshold: 0.500
- Use Expected Return: False

## Trade-Level Performance (event-time)

- Mean Return per Trade: -0.2956%
- Std Dev per Trade: 1.0839%
- Trade-Level Sharpe (sqrt(N)): -23.293
- Max Drawdown (event-time equity): -100.00%
- Hit Rate (gated trades): 33.70%
- Mean Return CI (bootstrap, 95%): [-0.3167%, -0.2709%]

## Baseline (Ungated) Event Performance

- Events in evaluation set: 13042
- Mean Return per Event: -0.2759%
- Std Dev per Event: 1.0976%
- Trade-Level Sharpe (sqrt(N)): -28.703
- Max Drawdown (event-time equity): -100.00%
- Hit Rate (events): 33.93%
- Return Quantiles (events): 5%=-1.1706%, 25%=-1.1706%, 50%=-0.9743%, 75%=0.4683%, 95%=1.8400%

## Gated Return Distribution

- Return Quantiles (gated trades): 5%=-1.1706%, 25%=-1.1706%, 50%=-1.0165%, 75%=0.4093%, 95%=1.8400%

## Temporal Stability (event-time segments)

| Segment | Trades | Mean Return | Sharpe (trade) |
|---------|--------|------------|----------------|
| 1 | 1459 | -0.3267% | -11.844 |
| 2 | 1459 | -0.2957% | -10.004 |
| 3 | 1459 | -0.2453% | -8.313 |
| 4 | 1459 | -0.2837% | -10.130 |
| 5 | 1458 | -0.3267% | -12.042 |

## Per-Regime Performance (gated trades)

| Regime | Trades | Mean Return | Sharpe (trade) |
|--------|--------|------------|----------------|
| -1.0 | 298 | -0.2706% | -4.253 |
| 2.0 | 624 | -0.1130% | -2.091 |
| 4.0 | 2718 | -0.3089% | -14.408 |
| 0.0 | 817 | -0.3355% | -9.315 |
| 3.0 | 943 | -0.2987% | -8.937 |
| 1.0 | 1059 | -0.2853% | -9.538 |

## Transaction Cost Stress Test

Multiplier refers to scaling of baseline transaction_cost used in labeling.

| Cost Multiplier | Mean Return | Sharpe (trade) |
|----------------|------------|----------------|
| 1.0 | -0.2956% | -23.293 |
| 2.0 | -0.5956% | -46.932 |
| 3.0 | -0.8956% | -70.572 |

## Meta-Gating Diagnostics (from meta-labeling step)

- These metrics are computed during the meta-labeling step for the diagnostics gate.
- AUC (OOF meta-model): 0.737
- Mean return per gated trade (diagnostics gate): -0.28%
- Sharpe (diagnostics gated set): -0.27
- Trades gated (diagnostics gate): 30430
- Approximate average trades per day (diagnostics gate): 20.80
