# Meta-Gated Backtest Report

- Symbol: ETHUSDT
- Exchange: binance
- Timeframe: 15m
- Direction: long
- Execution Mode: full
- Events (labeled, evaluation set): 16951
- Events (labeled, total): 56502
- Trades (gated): 2877

## Gating Configuration

- Probability Threshold: 0.750
- Use Expected Return: True
- Expected Return Threshold: 0.0015 (fraction)

## Trade-Level Performance (event-time)

- Mean Return per Trade: 1.3860%
- Std Dev per Trade: 1.0857%
- Trade-Level Sharpe (sqrt(N)): 68.468
- Max Drawdown (event-time equity): -16.70%
- Hit Rate (gated trades): 84.57%
- Mean Return CI (bootstrap, 95%): [1.3451%, 1.4276%]

## Baseline (Ungated) Event Performance

- Events in evaluation set: 16951
- Mean Return per Event: -0.1042%
- Std Dev per Event: 1.1579%
- Trade-Level Sharpe (sqrt(N)): -11.719
- Max Drawdown (event-time equity): -100.00%
- Hit Rate (events): 28.21%
- Return Quantiles (events): 5%=-0.9392%, 25%=-0.8662%, 50%=-0.7409%, 75%=0.4588%, 95%=2.1673%

## Gated Return Distribution

- Return Quantiles (gated trades): 5%=-0.9392%, 25%=0.8314%, 50%=1.7811%, 75%=2.0566%, 95%=2.6530%

## Temporal Stability (event-time segments)

| Segment | Trades | Mean Return | Sharpe (trade) |
|---------|--------|------------|----------------|
| 1 | 576 | 1.4606% | 33.288 |
| 2 | 576 | 1.5620% | 39.537 |
| 3 | 576 | 1.2061% | 25.733 |
| 4 | 576 | 1.3102% | 27.631 |
| 5 | 573 | 1.3909% | 29.665 |

## Transaction Cost Stress Test

Multiplier refers to scaling of baseline transaction_cost used in labeling.

| Cost Multiplier | Mean Return | Sharpe (trade) |
|----------------|------------|----------------|
| 1.0 | 1.3860% | 68.468 |
| 2.0 | 1.2360% | 61.058 |
| 3.0 | 1.0860% | 53.648 |

## Meta-Gating Diagnostics (from meta-labeling step)

- These metrics are computed during the meta-labeling step for the diagnostics gate.
- AUC (OOF meta-model): 0.832
- Mean return per gated trade (diagnostics gate): 1.32%
- Sharpe (diagnostics gated set): 1.18
- Trades gated (diagnostics gate): 7494
