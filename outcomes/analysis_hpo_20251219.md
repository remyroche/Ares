# HPO Analysis Report - 2025-12-19

## Overview
This report analyzes the results of the `meta_labeling_hpo_sample_weighted` step execution on ETHUSDT (15m timeframe, blank mode).

## Key Findings

### Critical Performance Failure
The trading simulation results indicate a catastrophic failure of the current model/strategy configuration:
- **Win Rate:** 0.0% across all probability thresholds (0.60 - 0.75).
- **Trades:** 2,787 executed trades.
- **Mean Return:** -0.2804% per trade (consistently).
- **Max Drawdown:** -99.97%.
- **Sharpe Ratio:** -15.44.

**Observation:** The consistency of the loss (every single trade lost) and the high volume of trades suggests a potential systematic error, such as:
1.  **Label/Signal Inversion:** The model might be predicting the opposite of the profitable direction.
2.  **Execution Logic Bug:** The realized return calculation might be flawed or using an incorrect price (e.g., entering at High instead of Close).
3.  **Data Issue:** Future leakage or incorrect target alignment.

### Model Robustness & Diagnostics
The SNR (Signal-to-Noise Ratio) diagnostics confirm the poor performance:
- **Score:** 0.000 / 1.0 (Rating: Bad).
- **Pseudo-R^2:** -3.4138 (Model performs worse than a naive baseline).
- **Estimated Label Noise:** 100% (Confident Learning suggests all confident predictions are mislabeled).
- **Residual Autocorrelation:** 0.5761 (High), indicating significant unmodeled temporal structure.

### Layer 2 & 3 Metrics
- **Layer 2 Optimization:** Produced geometries, but subsequent OOF evaluation likely failed to validate them.
- **Layer 3 Probe:** Failed to find any stable signal.

## Recommendations
1.  **Investigate Signal Inversion:** dramatically consistently poor performance often implies the signal is informative but reversed. Check `label_based_layer_2` and `label_based_layer_3` target definitions.
2.  **Debug Realized Returns:** Verify `compute_realized_returns` logic in `feature_generation_meta_labeling_step.py` or `LabelBasedLayerX`.
3.  **Review Feature Alignment:** Ensure features are not lagged incorrectly against the target.

## Next Steps
- Run standalone SNR diagnostics to double-check the findings (planned).
- Manually inspect a few specific events to trace why they were labeled as opportunities but resulted in losses.
