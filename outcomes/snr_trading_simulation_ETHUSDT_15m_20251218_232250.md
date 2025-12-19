# Trading Simulation Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## Overview
- Date range: 287 days
- Valid samples: 536

## Model Calibration
- Brier Score: 0.2660
- Expected Calibration Error (ECE): 0.1569
- Max Calibration Error (MCE): 0.6332

### Calibration Interpretation
- Brier > 0.25 → Poorly calibrated probabilities.
- ECE > 0.15 → Significant calibration error.

## Trading Metrics by Probability Threshold

### Threshold 0.60
- **Trades**: 142 (0.49/day)
- **Mean Return/Trade**: -0.2108%
- **PnL/Day**: -0.1043%
- **Win Rate**: 39.4%
- **Sharpe Ratio**: -1.531
- **Max Drawdown**: -44.37%
- **Final Equity**: 0.7271
- **Max Consecutive Losses**: 12
- **Avg Consecutive Losses**: 5.06
- **Win-Rate Stability**: 0.790

### Threshold 0.65
- **Trades**: 70 (0.24/day)
- **Mean Return/Trade**: -0.2999%
- **PnL/Day**: -0.0732%
- **Win Rate**: 37.1%
- **Sharpe Ratio**: -1.526
- **Max Drawdown**: -27.29%
- **Final Equity**: 0.8027
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 4.40
- **Win-Rate Stability**: 0.758

### Threshold 0.70
- **Trades**: 28 (0.10/day)
- **Mean Return/Trade**: -0.6095%
- **PnL/Day**: -0.0595%
- **Win Rate**: 28.6%
- **Sharpe Ratio**: -1.937
- **Max Drawdown**: -14.84%
- **Final Equity**: 0.8394
- **Max Consecutive Losses**: 6
- **Avg Consecutive Losses**: 3.33
- **Win-Rate Stability**: 0.789

### Threshold 0.75
- **Trades**: 16 (0.06/day)
- **Mean Return/Trade**: -0.4058%
- **PnL/Day**: -0.0226%
- **Win Rate**: 37.5%
- **Sharpe Ratio**: -1.012
- **Max Drawdown**: -7.58%
- **Final Equity**: 0.9351
- **Max Consecutive Losses**: 5
- **Avg Consecutive Losses**: 2.50
- **Win-Rate Stability**: nan

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.60 | 142 | 0.49 | -0.211% | -0.104% | 39.4% | -1.53 | -44.4% | 12 |
| 0.65 | 70 | 0.24 | -0.300% | -0.073% | 37.1% | -1.53 | -27.3% | 10 |
| 0.70 | 28 | 0.10 | -0.610% | -0.059% | 28.6% | -1.94 | -14.8% | 6 |
| 0.75 | 16 | 0.06 | -0.406% | -0.023% | 37.5% | -1.01 | -7.6% | 5 |

## Regime-Specific Recommended Thresholds

- **Regime** `vol_low`:
  - prob_threshold = 0.60
  - trades/day ≈ 0.156
  - mean_return ≈ 0.6813%
  - Sharpe ≈ 3.477
  - n_trades = 44
