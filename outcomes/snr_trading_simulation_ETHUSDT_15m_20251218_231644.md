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
- Brier Score: 0.2680
- Expected Calibration Error (ECE): 0.1275
- Max Calibration Error (MCE): 0.7732

### Calibration Interpretation
- Brier > 0.25 → Poorly calibrated probabilities.
- ECE 0.05-0.15 → Moderate calibration error.

## Trading Metrics by Probability Threshold

### Threshold 0.60
- **Trades**: 132 (0.46/day)
- **Mean Return/Trade**: -0.4959%
- **PnL/Day**: -0.2281%
- **Win Rate**: 31.1%
- **Sharpe Ratio**: -3.483
- **Max Drawdown**: -54.60%
- **Final Equity**: 0.5097
- **Max Consecutive Losses**: 13
- **Avg Consecutive Losses**: 5.69
- **Win-Rate Stability**: 0.865

### Threshold 0.65
- **Trades**: 71 (0.25/day)
- **Mean Return/Trade**: -0.6022%
- **PnL/Day**: -0.1490%
- **Win Rate**: 28.2%
- **Sharpe Ratio**: -2.983
- **Max Drawdown**: -36.40%
- **Final Equity**: 0.6446
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 5.67
- **Win-Rate Stability**: 0.844

### Threshold 0.70
- **Trades**: 28 (0.10/day)
- **Mean Return/Trade**: -0.8526%
- **PnL/Day**: -0.0832%
- **Win Rate**: 21.4%
- **Sharpe Ratio**: -2.690
- **Max Drawdown**: -23.23%
- **Final Equity**: 0.7837
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 4.40
- **Win-Rate Stability**: 0.862

### Threshold 0.75
- **Trades**: 23 (0.08/day)
- **Mean Return/Trade**: -0.8361%
- **PnL/Day**: -0.0670%
- **Win Rate**: 21.7%
- **Sharpe Ratio**: -2.263
- **Max Drawdown**: -20.54%
- **Final Equity**: 0.8214
- **Max Consecutive Losses**: 11
- **Avg Consecutive Losses**: 4.50
- **Win-Rate Stability**: 0.811

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.60 | 132 | 0.46 | -0.496% | -0.228% | 31.1% | -3.48 | -54.6% | 13 |
| 0.65 | 71 | 0.25 | -0.602% | -0.149% | 28.2% | -2.98 | -36.4% | 10 |
| 0.70 | 28 | 0.10 | -0.853% | -0.083% | 21.4% | -2.69 | -23.2% | 10 |
| 0.75 | 23 | 0.08 | -0.836% | -0.067% | 21.7% | -2.26 | -20.5% | 11 |

## Regime-Specific Recommended Thresholds

- **Regime** `vol_low`:
  - prob_threshold = 0.60
  - trades/day ≈ 0.089
  - mean_return ≈ 0.6408%
  - Sharpe ≈ 2.398
  - n_trades = 25
