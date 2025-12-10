# Trading Simulation Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1462 days
- Valid samples: 21177

## Model Calibration
- Brier Score: 0.1606
- Expected Calibration Error (ECE): 0.0285
- Max Calibration Error (MCE): 0.5611

### Calibration Interpretation
- Brier ≤ 0.18 → Well-calibrated probabilities.
- ECE ≤ 0.05 → Well-calibrated model.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 8840 (6.05/day)
- **Mean Return/Trade**: -0.3190%
- **PnL/Day**: -1.9286%
- **Win Rate**: 24.1%
- **Sharpe Ratio**: -27.735
- **Max Drawdown**: -100.00%
- **Final Equity**: 0.0000
- **Max Consecutive Losses**: 71
- **Avg Consecutive Losses**: 7.17
- **Win-Rate Stability**: 0.898

### Threshold 0.60
- **Trades**: 8149 (5.57/day)
- **Mean Return/Trade**: -0.3187%
- **PnL/Day**: -1.7763%
- **Win Rate**: 24.1%
- **Sharpe Ratio**: -26.602
- **Max Drawdown**: -100.00%
- **Final Equity**: 0.0000
- **Max Consecutive Losses**: 70
- **Avg Consecutive Losses**: 7.04
- **Win-Rate Stability**: 0.896

### Threshold 0.65
- **Trades**: 7408 (5.07/day)
- **Mean Return/Trade**: -0.3222%
- **PnL/Day**: -1.6328%
- **Win Rate**: 24.1%
- **Sharpe Ratio**: -25.698
- **Max Drawdown**: -100.00%
- **Final Equity**: 0.0000
- **Max Consecutive Losses**: 64
- **Avg Consecutive Losses**: 6.99
- **Win-Rate Stability**: 0.899

### Threshold 0.70
- **Trades**: 6613 (4.52/day)
- **Mean Return/Trade**: -0.3221%
- **PnL/Day**: -1.4568%
- **Win Rate**: 24.0%
- **Sharpe Ratio**: -24.263
- **Max Drawdown**: -100.00%
- **Final Equity**: 0.0000
- **Max Consecutive Losses**: 62
- **Avg Consecutive Losses**: 7.05
- **Win-Rate Stability**: 0.902

### Threshold 0.75
- **Trades**: 5698 (3.90/day)
- **Mean Return/Trade**: -0.3211%
- **PnL/Day**: -1.2516%
- **Win Rate**: 23.9%
- **Sharpe Ratio**: -22.449
- **Max Drawdown**: -100.00%
- **Final Equity**: 0.0000
- **Max Consecutive Losses**: 59
- **Avg Consecutive Losses**: 6.83
- **Win-Rate Stability**: 0.907

### Threshold 0.80
- **Trades**: 4735 (3.24/day)
- **Mean Return/Trade**: -0.3236%
- **PnL/Day**: -1.0481%
- **Win Rate**: 24.0%
- **Sharpe Ratio**: -20.667
- **Max Drawdown**: -100.00%
- **Final Equity**: 0.0000
- **Max Consecutive Losses**: 76
- **Avg Consecutive Losses**: 6.74
- **Win-Rate Stability**: 0.910

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 8840 | 6.05 | -0.319% | -1.929% | 24.1% | -27.74 | -100.0% | 71 |
| 0.60 | 8149 | 5.57 | -0.319% | -1.776% | 24.1% | -26.60 | -100.0% | 70 |
| 0.65 | 7408 | 5.07 | -0.322% | -1.633% | 24.1% | -25.70 | -100.0% | 64 |
| 0.70 | 6613 | 4.52 | -0.322% | -1.457% | 24.0% | -24.26 | -100.0% | 62 |
| 0.75 | 5698 | 3.90 | -0.321% | -1.252% | 23.9% | -22.45 | -100.0% | 59 |
| 0.80 | 4735 | 3.24 | -0.324% | -1.048% | 24.0% | -20.67 | -100.0% | 76 |

## Regime-Specific Recommended Thresholds

- **Regime** `hmm_-1.0`:
  - prob_threshold = 0.55
  - trades/day ≈ 0.293
  - mean_return ≈ 0.0955%
  - Sharpe ≈ 0.751
  - n_trades = 99
- **Regime** `hmm_2.0`:
  - prob_threshold = 0.80
  - trades/day ≈ 0.153
  - mean_return ≈ 0.1396%
  - Sharpe ≈ 0.779
  - n_trades = 55