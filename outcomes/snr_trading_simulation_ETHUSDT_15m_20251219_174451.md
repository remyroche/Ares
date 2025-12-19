# Trading Simulation Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## Overview
- Date range: 287 days
- Valid samples: 2784

## Model Calibration
- Brier Score: nan
- Expected Calibration Error (ECE): nan
- Max Calibration Error (MCE): nan

### Calibration Interpretation
- Brier score not available.
- ECE not available.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 354 (1.23/day)
- **Mean Return/Trade**: -0.3199%
- **PnL/Day**: -0.3946%
- **Win Rate**: 0.0%
- **Sharpe Ratio**: -5.836
- **Max Drawdown**: -68.15%
- **Final Equity**: 0.3156
- **Max Consecutive Losses**: 0
- **Avg Consecutive Losses**: 0.00
- **Win-Rate Stability**: 1.000

### Threshold 0.60
- **Trades**: 222 (0.77/day)
- **Mean Return/Trade**: -0.2993%
- **PnL/Day**: -0.2315%
- **Win Rate**: 0.0%
- **Sharpe Ratio**: -4.136
- **Max Drawdown**: -49.63%
- **Final Equity**: 0.5074
- **Max Consecutive Losses**: 0
- **Avg Consecutive Losses**: 0.00
- **Win-Rate Stability**: 1.000

### Threshold 0.65
- **Trades**: 123 (0.43/day)
- **Mean Return/Trade**: -0.3479%
- **PnL/Day**: -0.1491%
- **Win Rate**: 0.0%
- **Sharpe Ratio**: -3.430
- **Max Drawdown**: -35.50%
- **Final Equity**: 0.6463
- **Max Consecutive Losses**: 0
- **Avg Consecutive Losses**: 0.00
- **Win-Rate Stability**: 1.000

### Threshold 0.70
- **Trades**: 64 (0.22/day)
- **Mean Return/Trade**: -0.3931%
- **PnL/Day**: -0.0877%
- **Win Rate**: 0.0%
- **Sharpe Ratio**: -2.832
- **Max Drawdown**: -25.20%
- **Final Equity**: 0.7741
- **Max Consecutive Losses**: 0
- **Avg Consecutive Losses**: 0.00
- **Win-Rate Stability**: 1.000

### Threshold 0.75
- **Trades**: 34 (0.12/day)
- **Mean Return/Trade**: -0.4195%
- **PnL/Day**: -0.0497%
- **Win Rate**: 0.0%
- **Sharpe Ratio**: -2.141
- **Max Drawdown**: -15.04%
- **Final Equity**: 0.8649
- **Max Consecutive Losses**: 0
- **Avg Consecutive Losses**: 0.00
- **Win-Rate Stability**: 1.000

### Threshold 0.80
- **Trades**: 16 (0.06/day)
- **Mean Return/Trade**: -0.6056%
- **PnL/Day**: -0.0338%
- **Win Rate**: 0.0%
- **Sharpe Ratio**: -2.196
- **Max Drawdown**: -12.16%
- **Final Equity**: 0.9065
- **Max Consecutive Losses**: 0
- **Avg Consecutive Losses**: 0.00
- **Win-Rate Stability**: nan

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 354 | 1.23 | -0.320% | -0.395% | 0.0% | -5.84 | -68.2% | 0 |
| 0.60 | 222 | 0.77 | -0.299% | -0.232% | 0.0% | -4.14 | -49.6% | 0 |
| 0.65 | 123 | 0.43 | -0.348% | -0.149% | 0.0% | -3.43 | -35.5% | 0 |
| 0.70 | 64 | 0.22 | -0.393% | -0.088% | 0.0% | -2.83 | -25.2% | 0 |
| 0.75 | 34 | 0.12 | -0.420% | -0.050% | 0.0% | -2.14 | -15.0% | 0 |
| 0.80 | 16 | 0.06 | -0.606% | -0.034% | 0.0% | -2.20 | -12.2% | 0 |

## Regime-Specific Recommended Thresholds

- **Regime** `vol_low`:
  - prob_threshold = 0.65
  - trades/day ≈ 0.072
  - mean_return ≈ 0.0236%
  - Sharpe ≈ 0.132
  - n_trades = 16
