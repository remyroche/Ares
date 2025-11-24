# Trading Simulation Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1 days
- Valid samples: 31974

## Model Calibration
- Brier Score: 0.1675
- Expected Calibration Error (ECE): 0.0278
- Max Calibration Error (MCE): 0.3250

### Calibration Interpretation
- Brier ≤ 0.18 → Well-calibrated probabilities.
- ECE ≤ 0.05 → Well-calibrated model.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 10733 (10733.00/day)
- **Mean Return/Trade**: 1.0101%
- **PnL/Day**: 10841.8142%
- **Win Rate**: 82.8%
- **Sharpe Ratio**: 82.095
- **Max Drawdown**: -25.97%
- **Final Equity**: 29937978755377191958696587085855137447408566272.0000
- **Max Consecutive Losses**: 21
- **Avg Consecutive Losses**: 4.78
- **Win-Rate Stability**: 0.826

### Threshold 0.60
- **Trades**: 9204 (9204.00/day)
- **Mean Return/Trade**: 1.1563%
- **PnL/Day**: 10643.0035%
- **Win Rate**: 90.1%
- **Sharpe Ratio**: 91.263
- **Max Drawdown**: -22.71%
- **Final Equity**: 4638236635893756133312625821161923066969718784.0000
- **Max Consecutive Losses**: 25
- **Avg Consecutive Losses**: 3.90
- **Win-Rate Stability**: 0.872

### Threshold 0.65
- **Trades**: 8301 (8301.00/day)
- **Mean Return/Trade**: 1.2495%
- **PnL/Day**: 10371.8081%
- **Win Rate**: 95.0%
- **Sharpe Ratio**: 97.679
- **Max Drawdown**: -22.71%
- **Final Equity**: 334632300207246927368161064643012619650727936.0000
- **Max Consecutive Losses**: 15
- **Avg Consecutive Losses**: 3.52
- **Win-Rate Stability**: 0.920

### Threshold 0.70
- **Trades**: 7806 (7806.00/day)
- **Mean Return/Trade**: 1.2982%
- **PnL/Day**: 10133.6119%
- **Win Rate**: 97.9%
- **Sharpe Ratio**: 101.333
- **Max Drawdown**: -22.71%
- **Final Equity**: 32604155612525662260637394069305863117471744.0000
- **Max Consecutive Losses**: 11
- **Avg Consecutive Losses**: 2.59
- **Win-Rate Stability**: 0.955

### Threshold 0.75
- **Trades**: 7494 (7494.00/day)
- **Mean Return/Trade**: 1.3177%
- **PnL/Day**: 9874.7198%
- **Win Rate**: 99.3%
- **Sharpe Ratio**: 101.959
- **Max Drawdown**: -22.71%
- **Final Equity**: 2542901407252045674542681447071352285560832.0000
- **Max Consecutive Losses**: 5
- **Avg Consecutive Losses**: 2.04
- **Win-Rate Stability**: 0.980

### Threshold 0.80
- **Trades**: 7278 (7278.00/day)
- **Mean Return/Trade**: 1.3198%
- **PnL/Day**: 9605.7829%
- **Win Rate**: 99.8%
- **Sharpe Ratio**: 100.378
- **Max Drawdown**: -22.71%
- **Final Equity**: 177536270149402265383287034438954954784768.0000
- **Max Consecutive Losses**: 4
- **Avg Consecutive Losses**: 2.00
- **Win-Rate Stability**: 0.989

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 10733 | 10733.00 | 1.010% | 10841.814% | 82.8% | 82.09 | -26.0% | 21 |
| 0.60 | 9204 | 9204.00 | 1.156% | 10643.003% | 90.1% | 91.26 | -22.7% | 25 |
| 0.65 | 8301 | 8301.00 | 1.249% | 10371.808% | 95.0% | 97.68 | -22.7% | 15 |
| 0.70 | 7806 | 7806.00 | 1.298% | 10133.612% | 97.9% | 101.33 | -22.7% | 11 |
| 0.75 | 7494 | 7494.00 | 1.318% | 9874.720% | 99.3% | 101.96 | -22.7% | 5 |
| 0.80 | 7278 | 7278.00 | 1.320% | 9605.783% | 99.8% | 100.38 | -22.7% | 4 |