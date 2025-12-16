# Trading Simulation Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 360 days
- Valid samples: 2064

## Model Calibration
- Brier Score: 0.1947
- Expected Calibration Error (ECE): 0.0512
- Max Calibration Error (MCE): 0.3455

### Calibration Interpretation
- Brier 0.18-0.25 → Moderate calibration.
- ECE 0.05-0.15 → Moderate calibration error.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 32 (0.09/day)
- **Mean Return/Trade**: -0.1277%
- **PnL/Day**: -0.0114%
- **Win Rate**: 34.4%
- **Sharpe Ratio**: -2.717
- **Max Drawdown**: -4.55%
- **Final Equity**: 0.9598
- **Max Consecutive Losses**: 8
- **Avg Consecutive Losses**: 3.00
- **Win-Rate Stability**: 0.887

### Threshold 0.60
- **Trades**: 18 (0.05/day)
- **Mean Return/Trade**: -0.0391%
- **PnL/Day**: -0.0020%
- **Win Rate**: 50.0%
- **Sharpe Ratio**: -0.585
- **Max Drawdown**: -1.37%
- **Final Equity**: 0.9929
- **Max Consecutive Losses**: 3
- **Avg Consecutive Losses**: 1.80
- **Win-Rate Stability**: nan

### Threshold 0.65
- Insufficient data (7 trades)

### Threshold 0.70
- Insufficient data (3 trades)

### Threshold 0.75
- Insufficient data (0 trades)

### Threshold 0.80
- Insufficient data (0 trades)

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 32 | 0.09 | -0.128% | -0.011% | 34.4% | -2.72 | -4.6% | 8 |
| 0.60 | 18 | 0.05 | -0.039% | -0.002% | 50.0% | -0.59 | -1.4% | 3 |