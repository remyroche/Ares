# SNR Diagnostics Analysis - 2025-12-19

## Overview
This report analyzes the results of the standalone `snr_diagnostics` run on ETHUSDT (15m, long, 2025-12-19).

## Key Findings (Confirmation of Failure)
The standalone diagnostics confirm the findings from the HPO run. The model shows zero predictive power and catastrophic trading performance.

- **Win Rate:** 0.0% at all probability thresholds (0.55 - 0.80).
- **Sharpe Ratio:** -15.44.
- **Mean Return:** -0.2804% per trade.
- **Estimated Label Noise:** 100%.

## Analysis of the "Zero Win Rate" Anomaly
The fact that *every single trade* (2,787 trades) resulted in a loss is statistically impossible for a random signal in a balanced market. This systematic failure points to:
1.  **Signal Inversion:** The model might be perfectly predicting the *wrong* direction (e.g., predicting "up" when price goes "down").
2.  **Execution Simulation Error:** The backtesting logic might be incorrectly calculating PnL (e.g., applying a massive spread/fee that exceeds volatility, or treating every exit as a stop-loss).
3.  **Target Mismatch:** The model is trained on a different target (e.g., 2h horizon) than what is being simulated (e.g., 15m horizon with tight stop).

## Recommendations
- **Immediate Debugging:** Do not proceed with further training until the PnL calculation and signal direction are verified.
- **Inspect `snr_diagnostics.py`:** Check how it loads `oof_preds` and `market_data`. Ensure timestamps align.
- **Check Costs:** Verify if transaction costs (slip + comm) in the simulation are set unrealistically high (e.g., > 0.5%).

## Conclusion
The current iteration is non-functional. The consistent nature of the failure (-0.28% return per trade for ALL trades) strongly identifies a deterministic error in the evaluation pipeline rather than just "bad modeling".
