# Trading Simulation Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1489 days
- Valid samples: 4187

## Model Calibration
- Brier Score: 0.1978
- Expected Calibration Error (ECE): 0.0665
- Max Calibration Error (MCE): 0.3679

### Calibration Interpretation
- Brier 0.18-0.25 → Moderate calibration.
- ECE 0.05-0.15 → Moderate calibration error.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 925 (0.62/day)
- **Mean Return/Trade**: 0.3633%
- **PnL/Day**: 0.2257%
- **Win Rate**: 61.7%
- **Sharpe Ratio**: 15.507
- **Max Drawdown**: -8.69%
- **Final Equity**: 27.9579
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 2.20
- **Win-Rate Stability**: 0.862

### Threshold 0.60
- **Trades**: 797 (0.54/day)
- **Mean Return/Trade**: 0.4103%
- **PnL/Day**: 0.2196%
- **Win Rate**: 65.7%
- **Sharpe Ratio**: 16.691
- **Max Drawdown**: -7.43%
- **Final Equity**: 25.6496
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 2.01
- **Win-Rate Stability**: 0.872

### Threshold 0.65
- **Trades**: 677 (0.45/day)
- **Mean Return/Trade**: 0.4836%
- **PnL/Day**: 0.2199%
- **Win Rate**: 71.2%
- **Sharpe Ratio**: 19.226
- **Max Drawdown**: -5.86%
- **Final Equity**: 25.8385
- **Max Consecutive Losses**: 13
- **Avg Consecutive Losses**: 1.77
- **Win-Rate Stability**: 0.900

### Threshold 0.70
- **Trades**: 557 (0.37/day)
- **Mean Return/Trade**: 0.5517%
- **PnL/Day**: 0.2064%
- **Win Rate**: 75.8%
- **Sharpe Ratio**: 21.276
- **Max Drawdown**: -4.21%
- **Final Equity**: 21.1996
- **Max Consecutive Losses**: 7
- **Avg Consecutive Losses**: 1.50
- **Win-Rate Stability**: 0.931

### Threshold 0.75
- **Trades**: 453 (0.30/day)
- **Mean Return/Trade**: 0.6242%
- **PnL/Day**: 0.1899%
- **Win Rate**: 80.6%
- **Sharpe Ratio**: 23.881
- **Max Drawdown**: -2.72%
- **Final Equity**: 16.6442
- **Max Consecutive Losses**: 4
- **Avg Consecutive Losses**: 1.29
- **Win-Rate Stability**: 0.944

### Threshold 0.80
- **Trades**: 359 (0.24/day)
- **Mean Return/Trade**: 0.7122%
- **PnL/Day**: 0.1717%
- **Win Rate**: 86.4%
- **Sharpe Ratio**: 29.669
- **Max Drawdown**: -2.01%
- **Final Equity**: 12.7289
- **Max Consecutive Losses**: 3
- **Avg Consecutive Losses**: 1.17
- **Win-Rate Stability**: 0.956

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 925 | 0.62 | 0.363% | 0.226% | 61.7% | 15.51 | -8.7% | 10 |
| 0.60 | 797 | 0.54 | 0.410% | 0.220% | 65.7% | 16.69 | -7.4% | 10 |
| 0.65 | 677 | 0.45 | 0.484% | 0.220% | 71.2% | 19.23 | -5.9% | 13 |
| 0.70 | 557 | 0.37 | 0.552% | 0.206% | 75.8% | 21.28 | -4.2% | 7 |
| 0.75 | 453 | 0.30 | 0.624% | 0.190% | 80.6% | 23.88 | -2.7% | 4 |
| 0.80 | 359 | 0.24 | 0.712% | 0.172% | 86.4% | 29.67 | -2.0% | 3 |