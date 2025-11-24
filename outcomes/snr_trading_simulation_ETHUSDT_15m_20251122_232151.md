# Trading Simulation Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1 days
- Valid samples: 31972

## Model Calibration
- Brier Score: 0.1674
- Expected Calibration Error (ECE): 0.0258
- Max Calibration Error (MCE): 0.1741

### Calibration Interpretation
- Brier ≤ 0.18 → Well-calibrated probabilities.
- ECE ≤ 0.05 → Well-calibrated model.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 10668 (10668.00/day)
- **Mean Return/Trade**: 1.0136%
- **PnL/Day**: 10813.1582%
- **Win Rate**: 83.3%
- **Sharpe Ratio**: 81.792
- **Max Drawdown**: -26.14%
- **Final Equity**: 22432442177062489178181814493121661581677035520.0000
- **Max Consecutive Losses**: 22
- **Avg Consecutive Losses**: 4.76
- **Win-Rate Stability**: 0.831

### Threshold 0.60
- **Trades**: 9180 (9180.00/day)
- **Mean Return/Trade**: 1.1540%
- **PnL/Day**: 10593.4662%
- **Win Rate**: 90.6%
- **Sharpe Ratio**: 90.361
- **Max Drawdown**: -26.14%
- **Final Equity**: 2818107061893985696258161412142617482890313728.0000
- **Max Consecutive Losses**: 16
- **Avg Consecutive Losses**: 3.69
- **Win-Rate Stability**: 0.888

### Threshold 0.65
- **Trades**: 8302 (8302.00/day)
- **Mean Return/Trade**: 1.2288%
- **PnL/Day**: 10201.7868%
- **Win Rate**: 95.0%
- **Sharpe Ratio**: 94.575
- **Max Drawdown**: -26.14%
- **Final Equity**: 61311047164197078602288147106848091379073024.0000
- **Max Consecutive Losses**: 14
- **Avg Consecutive Losses**: 3.04
- **Win-Rate Stability**: 0.923

### Threshold 0.70
- **Trades**: 7840 (7840.00/day)
- **Mean Return/Trade**: 1.2691%
- **PnL/Day**: 9949.5834%
- **Win Rate**: 97.4%
- **Sharpe Ratio**: 96.887
- **Max Drawdown**: -26.14%
- **Final Equity**: 5172670206240348987974454284360168318697472.0000
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 2.61
- **Win-Rate Stability**: 0.951

### Threshold 0.75
- **Trades**: 7518 (7518.00/day)
- **Mean Return/Trade**: 1.2972%
- **PnL/Day**: 9752.5456%
- **Win Rate**: 99.1%
- **Sharpe Ratio**: 98.329
- **Max Drawdown**: -26.14%
- **Final Equity**: 745929359275024135625913036557615282257920.0000
- **Max Consecutive Losses**: 7
- **Avg Consecutive Losses**: 2.13
- **Win-Rate Stability**: 0.978

### Threshold 0.80
- **Trades**: 7266 (7266.00/day)
- **Mean Return/Trade**: 1.2980%
- **PnL/Day**: 9431.2911%
- **Win Rate**: 99.7%
- **Sharpe Ratio**: 96.478
- **Max Drawdown**: -26.14%
- **Final Equity**: 31069225538633575275715098042844359688192.0000
- **Max Consecutive Losses**: 2
- **Avg Consecutive Losses**: 1.73
- **Win-Rate Stability**: 0.991

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 10668 | 10668.00 | 1.014% | 10813.158% | 83.3% | 81.79 | -26.1% | 22 |
| 0.60 | 9180 | 9180.00 | 1.154% | 10593.466% | 90.6% | 90.36 | -26.1% | 16 |
| 0.65 | 8302 | 8302.00 | 1.229% | 10201.787% | 95.0% | 94.57 | -26.1% | 14 |
| 0.70 | 7840 | 7840.00 | 1.269% | 9949.583% | 97.4% | 96.89 | -26.1% | 10 |
| 0.75 | 7518 | 7518.00 | 1.297% | 9752.546% | 99.1% | 98.33 | -26.1% | 7 |
| 0.80 | 7266 | 7266.00 | 1.298% | 9431.291% | 99.7% | 96.48 | -26.1% | 2 |