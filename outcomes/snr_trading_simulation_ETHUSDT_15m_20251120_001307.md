# Trading Simulation Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1462 days
- Valid samples: 22658

## Model Calibration
- Brier Score: 0.2096
- Expected Calibration Error (ECE): 0.0585
- Max Calibration Error (MCE): 0.2112

### Calibration Interpretation
- Brier 0.18-0.25 → Moderate calibration.
- ECE 0.05-0.15 → Moderate calibration error.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 6836 (4.68/day)
- **Mean Return/Trade**: 0.7119%
- **PnL/Day**: 3.3287%
- **Win Rate**: 74.5%
- **Sharpe Ratio**: 61.568
- **Max Drawdown**: -25.66%
- **Final Equity**: 843053470919304806400.0000
- **Max Consecutive Losses**: 22
- **Avg Consecutive Losses**: 3.84
- **Win-Rate Stability**: 0.802

### Threshold 0.60
- **Trades**: 5602 (3.83/day)
- **Mean Return/Trade**: 0.8574%
- **PnL/Day**: 3.2853%
- **Win Rate**: 81.1%
- **Sharpe Ratio**: 73.827
- **Max Drawdown**: -22.69%
- **Final Equity**: 478333861427060736000.0000
- **Max Consecutive Losses**: 17
- **Avg Consecutive Losses**: 3.41
- **Win-Rate Stability**: 0.827

### Threshold 0.65
- **Trades**: 4620 (3.16/day)
- **Mean Return/Trade**: 1.0177%
- **PnL/Day**: 3.2159%
- **Win Rate**: 88.4%
- **Sharpe Ratio**: 95.426
- **Max Drawdown**: -25.36%
- **Final Equity**: 183601233967937880064.0000
- **Max Consecutive Losses**: 19
- **Avg Consecutive Losses**: 3.13
- **Win-Rate Stability**: 0.866

### Threshold 0.70
- **Trades**: 4044 (2.77/day)
- **Mean Return/Trade**: 1.1254%
- **PnL/Day**: 3.1130%
- **Win Rate**: 93.2%
- **Sharpe Ratio**: 122.696
- **Max Drawdown**: -14.76%
- **Final Equity**: 42220174133234917376.0000
- **Max Consecutive Losses**: 16
- **Avg Consecutive Losses**: 2.97
- **Win-Rate Stability**: 0.902

### Threshold 0.75
- **Trades**: 3589 (2.45/day)
- **Mean Return/Trade**: 1.2106%
- **PnL/Day**: 2.9720%
- **Win Rate**: 97.0%
- **Sharpe Ratio**: 174.827
- **Max Drawdown**: -8.01%
- **Final Equity**: 5540287724817795072.0000
- **Max Consecutive Losses**: 9
- **Avg Consecutive Losses**: 2.42
- **Win-Rate Stability**: 0.947

### Threshold 0.80
- **Trades**: 3213 (2.20/day)
- **Mean Return/Trade**: 1.2523%
- **PnL/Day**: 2.7521%
- **Win Rate**: 98.6%
- **Sharpe Ratio**: 235.358
- **Max Drawdown**: -8.01%
- **Final Equity**: 228718024775217120.0000
- **Max Consecutive Losses**: 9
- **Avg Consecutive Losses**: 1.76
- **Win-Rate Stability**: 0.966

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 6836 | 4.68 | 0.712% | 3.329% | 74.5% | 61.57 | -25.7% | 22 |
| 0.60 | 5602 | 3.83 | 0.857% | 3.285% | 81.1% | 73.83 | -22.7% | 17 |
| 0.65 | 4620 | 3.16 | 1.018% | 3.216% | 88.4% | 95.43 | -25.4% | 19 |
| 0.70 | 4044 | 2.77 | 1.125% | 3.113% | 93.2% | 122.70 | -14.8% | 16 |
| 0.75 | 3589 | 2.45 | 1.211% | 2.972% | 97.0% | 174.83 | -8.0% | 9 |
| 0.80 | 3213 | 2.20 | 1.252% | 2.752% | 98.6% | 235.36 | -8.0% | 9 |