# Trading Simulation Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1462 days
- Valid samples: 21177

## Model Calibration
- Brier Score: 0.1405
- Expected Calibration Error (ECE): 0.0147
- Max Calibration Error (MCE): 0.1582

### Calibration Interpretation
- Brier ≤ 0.18 → Well-calibrated probabilities.
- ECE ≤ 0.05 → Well-calibrated model.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 1094 (0.75/day)
- **Mean Return/Trade**: 0.3792%
- **PnL/Day**: 0.2838%
- **Win Rate**: 49.1%
- **Sharpe Ratio**: 10.089
- **Max Drawdown**: -24.92%
- **Final Equity**: 57.7905
- **Max Consecutive Losses**: 19
- **Avg Consecutive Losses**: 3.46
- **Win-Rate Stability**: 0.861

### Threshold 0.60
- **Trades**: 1015 (0.69/day)
- **Mean Return/Trade**: 0.3960%
- **PnL/Day**: 0.2750%
- **Win Rate**: 49.8%
- **Sharpe Ratio**: 10.193
- **Max Drawdown**: -23.96%
- **Final Equity**: 51.1406
- **Max Consecutive Losses**: 17
- **Avg Consecutive Losses**: 3.42
- **Win-Rate Stability**: 0.863

### Threshold 0.65
- **Trades**: 930 (0.64/day)
- **Mean Return/Trade**: 0.4263%
- **PnL/Day**: 0.2712%
- **Win Rate**: 51.3%
- **Sharpe Ratio**: 10.557
- **Max Drawdown**: -26.45%
- **Final Equity**: 48.7086
- **Max Consecutive Losses**: 16
- **Avg Consecutive Losses**: 3.28
- **Win-Rate Stability**: 0.858

### Threshold 0.70
- **Trades**: 834 (0.57/day)
- **Mean Return/Trade**: 0.4700%
- **PnL/Day**: 0.2681%
- **Win Rate**: 53.1%
- **Sharpe Ratio**: 11.144
- **Max Drawdown**: -23.62%
- **Final Equity**: 46.9700
- **Max Consecutive Losses**: 16
- **Avg Consecutive Losses**: 3.05
- **Win-Rate Stability**: 0.861

### Threshold 0.75
- **Trades**: 753 (0.52/day)
- **Mean Return/Trade**: 0.5042%
- **PnL/Day**: 0.2597%
- **Win Rate**: 55.4%
- **Sharpe Ratio**: 11.451
- **Max Drawdown**: -22.72%
- **Final Equity**: 41.7943
- **Max Consecutive Losses**: 16
- **Avg Consecutive Losses**: 3.05
- **Win-Rate Stability**: 0.861

### Threshold 0.80
- **Trades**: 641 (0.44/day)
- **Mean Return/Trade**: 0.5296%
- **PnL/Day**: 0.2322%
- **Win Rate**: 57.6%
- **Sharpe Ratio**: 11.244
- **Max Drawdown**: -19.69%
- **Final Equity**: 28.2404
- **Max Consecutive Losses**: 13
- **Avg Consecutive Losses**: 2.99
- **Win-Rate Stability**: 0.871

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 1094 | 0.75 | 0.379% | 0.284% | 49.1% | 10.09 | -24.9% | 19 |
| 0.60 | 1015 | 0.69 | 0.396% | 0.275% | 49.8% | 10.19 | -24.0% | 17 |
| 0.65 | 930 | 0.64 | 0.426% | 0.271% | 51.3% | 10.56 | -26.4% | 16 |
| 0.70 | 834 | 0.57 | 0.470% | 0.268% | 53.1% | 11.14 | -23.6% | 16 |
| 0.75 | 753 | 0.52 | 0.504% | 0.260% | 55.4% | 11.45 | -22.7% | 16 |
| 0.80 | 641 | 0.44 | 0.530% | 0.232% | 57.6% | 11.24 | -19.7% | 13 |

## Recommended Gating Threshold (from Trading Simulation)

- **Probability threshold**: 0.55
- **Trades**: 1094 (0.748/day)
- **Mean return/trade**: 0.3792%
- **PnL/day**: 0.2838%
- **Sharpe (trades)**: 10.089
- **Max drawdown**: -24.92%
- **Final equity**: 57.7905

## Regime-Specific Recommended Thresholds

- **Regime** `hmm_-1.0`:
  - prob_threshold = 0.65
  - trades/day ≈ 0.104
  - mean_return ≈ 0.9595%
  - Sharpe ≈ 4.950
  - n_trades = 35
- **Regime** `hmm_0.0`:
  - prob_threshold = 0.55
  - trades/day ≈ 0.308
  - mean_return ≈ 0.6388%
  - Sharpe ≈ 6.049
  - n_trades = 104
- **Regime** `hmm_1.0`:
  - prob_threshold = 0.55
  - trades/day ≈ 0.935
  - mean_return ≈ 0.3716%
  - Sharpe ≈ 5.448
  - n_trades = 315
- **Regime** `hmm_2.0`:
  - prob_threshold = 0.60
  - trades/day ≈ 0.220
  - mean_return ≈ 0.3168%
  - Sharpe ≈ 2.075
  - n_trades = 79
- **Regime** `hmm_3.0`:
  - prob_threshold = 0.80
  - trades/day ≈ 0.234
  - mean_return ≈ 0.2078%
  - Sharpe ≈ 1.522
  - n_trades = 79
- **Regime** `hmm_4.0`:
  - prob_threshold = 0.75
  - trades/day ≈ 0.861
  - mean_return ≈ 0.5568%
  - Sharpe ≈ 8.072
  - n_trades = 309