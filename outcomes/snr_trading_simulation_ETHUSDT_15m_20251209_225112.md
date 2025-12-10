# Trading Simulation Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1462 days
- Valid samples: 21176

## Model Calibration
- Brier Score: 0.1466
- Expected Calibration Error (ECE): 0.0170
- Max Calibration Error (MCE): 0.1482

### Calibration Interpretation
- Brier ≤ 0.18 → Well-calibrated probabilities.
- ECE ≤ 0.05 → Well-calibrated model.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 748 (0.51/day)
- **Mean Return/Trade**: 0.4511%
- **PnL/Day**: 0.2308%
- **Win Rate**: 49.3%
- **Sharpe Ratio**: 8.991
- **Max Drawdown**: -36.56%
- **Final Equity**: 27.0315
- **Max Consecutive Losses**: 13
- **Avg Consecutive Losses**: 3.38
- **Win-Rate Stability**: 0.842

### Threshold 0.60
- **Trades**: 675 (0.46/day)
- **Mean Return/Trade**: 0.5063%
- **PnL/Day**: 0.2337%
- **Win Rate**: 51.4%
- **Sharpe Ratio**: 9.657
- **Max Drawdown**: -31.40%
- **Final Equity**: 28.4016
- **Max Consecutive Losses**: 14
- **Avg Consecutive Losses**: 3.25
- **Win-Rate Stability**: 0.852

### Threshold 0.65
- **Trades**: 594 (0.41/day)
- **Mean Return/Trade**: 0.5654%
- **PnL/Day**: 0.2297%
- **Win Rate**: 53.9%
- **Sharpe Ratio**: 10.284
- **Max Drawdown**: -25.79%
- **Final Equity**: 27.0088
- **Max Consecutive Losses**: 12
- **Avg Consecutive Losses**: 3.11
- **Win-Rate Stability**: 0.862

### Threshold 0.70
- **Trades**: 531 (0.36/day)
- **Mean Return/Trade**: 0.5849%
- **PnL/Day**: 0.2124%
- **Win Rate**: 55.2%
- **Sharpe Ratio**: 10.157
- **Max Drawdown**: -22.38%
- **Final Equity**: 21.1223
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 2.98
- **Win-Rate Stability**: 0.887

### Threshold 0.75
- **Trades**: 469 (0.32/day)
- **Mean Return/Trade**: 0.6383%
- **PnL/Day**: 0.2047%
- **Win Rate**: 57.4%
- **Sharpe Ratio**: 10.527
- **Max Drawdown**: -20.60%
- **Final Equity**: 18.9876
- **Max Consecutive Losses**: 9
- **Avg Consecutive Losses**: 2.82
- **Win-Rate Stability**: 0.918

### Threshold 0.80
- **Trades**: 398 (0.27/day)
- **Mean Return/Trade**: 0.6377%
- **PnL/Day**: 0.1736%
- **Win Rate**: 59.8%
- **Sharpe Ratio**: 9.786
- **Max Drawdown**: -18.19%
- **Final Equity**: 12.1407
- **Max Consecutive Losses**: 8
- **Avg Consecutive Losses**: 2.62
- **Win-Rate Stability**: 0.899

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 748 | 0.51 | 0.451% | 0.231% | 49.3% | 8.99 | -36.6% | 13 |
| 0.60 | 675 | 0.46 | 0.506% | 0.234% | 51.4% | 9.66 | -31.4% | 14 |
| 0.65 | 594 | 0.41 | 0.565% | 0.230% | 53.9% | 10.28 | -25.8% | 12 |
| 0.70 | 531 | 0.36 | 0.585% | 0.212% | 55.2% | 10.16 | -22.4% | 10 |
| 0.75 | 469 | 0.32 | 0.638% | 0.205% | 57.4% | 10.53 | -20.6% | 9 |
| 0.80 | 398 | 0.27 | 0.638% | 0.174% | 59.8% | 9.79 | -18.2% | 8 |

## Recommended Gating Threshold (from Trading Simulation)

- **Probability threshold**: 0.60
- **Trades**: 675 (0.462/day)
- **Mean return/trade**: 0.5063%
- **PnL/day**: 0.2337%
- **Sharpe (trades)**: 9.657
- **Max drawdown**: -31.40%
- **Final equity**: 28.4016

## Regime-Specific Recommended Thresholds

- **Regime** `hmm_-1.0`:
  - prob_threshold = 0.60
  - trades/day ≈ 0.068
  - mean_return ≈ 0.9605%
  - Sharpe ≈ 3.769
  - n_trades = 23
- **Regime** `hmm_0.0`:
  - prob_threshold = 0.75
  - trades/day ≈ 0.133
  - mean_return ≈ 1.1881%
  - Sharpe ≈ 9.683
  - n_trades = 45
- **Regime** `hmm_1.0`:
  - prob_threshold = 0.55
  - trades/day ≈ 0.579
  - mean_return ≈ 0.5487%
  - Sharpe ≈ 5.683
  - n_trades = 195
- **Regime** `hmm_2.0`:
  - prob_threshold = 0.60
  - trades/day ≈ 0.201
  - mean_return ≈ 0.2479%
  - Sharpe ≈ 1.410
  - n_trades = 72
- **Regime** `hmm_3.0`:
  - prob_threshold = 0.80
  - trades/day ≈ 0.098
  - mean_return ≈ 0.8588%
  - Sharpe ≈ 4.800
  - n_trades = 33
- **Regime** `hmm_4.0`:
  - prob_threshold = 0.55
  - trades/day ≈ 0.833
  - mean_return ≈ 0.4453%
  - Sharpe ≈ 5.554
  - n_trades = 299